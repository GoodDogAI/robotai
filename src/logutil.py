import os
import re
import hashlib
import logging
from numpy import isin

from sqlalchemy import Column, Integer, String, JSON, create_engine, select, delete
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.orm.attributes import flag_modified

from typing import List, Dict, BinaryIO, Tuple
from functools import total_ordering
from capnp.lib import capnp
from cereal import log
from typing import Optional


logger = logging.getLogger(__name__)
Base = declarative_base()

LOGNAME_RE = re.compile(r"(?P<logname>[a-z]+)-(?P<runname>[a-f0-9]+)-(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})-(?P<hour>\d{1,2})_(?P<minute>\d{1,2}).log")


def sha256(filename: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


# Just makes sure that every event in the log file can be parsed, doesn't handle modelValidation events
def quick_validate_log(f: BinaryIO) -> bool:
    try:
        events = log.Event.read_multiple(f)
        
        for evt in events:
            evt.which()

        return True
    except capnp.KjException:
        return False


def check_log_monotonic(f: BinaryIO) -> bool:
    last_log_mono_time = None

    events = log.Event.read_multiple(f)

    for evt in events:
        if last_log_mono_time is not None and evt.logMonoTime < last_log_mono_time:
            return False

        last_log_mono_time = evt.logMonoTime

    return True


def resort_log_monotonic(input: BinaryIO, output: BinaryIO):
    events = log.Event.read_multiple(input)
    events = sorted(events, key=lambda evt: evt.logMonoTime)

    for evt in events:
        evt.as_builder().write(output)


def get_runname(filename: str) -> str:
    match = LOGNAME_RE.match(filename)
    if match:
        return match.group("logname") + "-" + match.group("runname")
    else:
        return ""


@total_ordering
class LogSummary(Base):
    __tablename__ = "logs"

    filename = Column(String, primary_key=True)
    sha256 = Column(String)
    last_modified = Column(Integer)
    orig_sha256 = Column(String, nullable=True)

    meta = Column(JSON, default={})

    def __le__(self, other):
        return self.filename < other.filename

    def get_runname(self):
        return get_runname(self.filename)

    def __repr__(self) -> str:
        return f"LogSummary(filename={self.filename}, sha256={self.sha256}, last_modified={self.last_modified}, orig_sha256={self.orig_sha256})"


# Allows for quick and cached access to the SHA256 hash of a bunch of log files
class LogHashes:
    dir: str
    extension: str = ".log"

    def __init__(self, dir):
        self.dir = dir
        self.engine = create_engine(f"sqlite:///{os.path.join(dir, '_hashes.sqlite')}", echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.update()

    def update(self, original_hashes: Dict[str, str] = {}):
        with Session(self.engine) as session:
            found_files = set()

            for file in os.listdir(self.dir):
                if not file.endswith(self.extension):
                    continue

                found_files.add(file)
                filepath = os.path.join(self.dir, file)
                mtime = round(os.path.getmtime(filepath) * 1e9)

                existing = session.execute(select(LogSummary).where(LogSummary.filename == file)).scalar_one_or_none()
                if existing is not None and existing.last_modified == mtime:
                    continue

                logger.warning(f"Hashing {filepath}")
        
                new_sha = sha256(filepath)

                if file in original_hashes:
                    orig_sha = original_hashes[file]
                elif existing is not None:
                    orig_sha = existing.orig_sha256
                else:
                    orig_sha = new_sha 

                session.merge(LogSummary(filename=file, sha256=new_sha, orig_sha256=orig_sha, last_modified=mtime))
                session.commit()

            # Remove any files that are no longer in the directory
            session.execute(delete(LogSummary).where(LogSummary.filename.not_in(found_files)))
            session.commit()


    def values(self) -> List[LogSummary]:
        with Session(self.engine) as session:
            return session.execute(select(LogSummary)).scalars().all()

    def hash_exists(self, hash:str) -> bool:
        with Session(self.engine) as session:
            return session.execute(select(LogSummary).where((LogSummary.sha256 == hash) | (LogSummary.orig_sha256 == hash))).scalar_one_or_none() is not None

    def filename_exists(self, filename:str) -> bool:
        with Session(self.engine) as session:
            return session.execute(select(LogSummary).where(LogSummary.filename == filename)).scalar_one_or_none() is not None

    def get_logsummary(self, filename: str) -> Optional[LogSummary]:
        with Session(self.engine) as session:
            return session.execute(select(LogSummary).where(LogSummary.filename == filename)).scalar_one_or_none()

    def update_metadata(self, filename: str, **kwargs):
        with Session(self.engine) as session:
            existing = session.execute(select(LogSummary).where(LogSummary.filename == filename)).scalar_one_or_none()
            if existing is not None:
                if not isinstance(existing.meta, dict):
                    existing.meta = {}
                existing.meta.update(kwargs)
                flag_modified(existing, "meta")
                session.commit()
            else:
                raise KeyError(f"Log {filename} not found")

    def __str__(self):
        return f"LogHashes(dir={self.dir})"

    def _sort_by_time(self, log: LogSummary) -> Tuple:
        m = LOGNAME_RE.match(log.filename)

        if m:
            return (int(m["year"]), int(m["month"]), int(m["day"]), int(m["hour"]), int(m["minute"]))
        else:
            return (0, 0, 0, 0, 0)    

    def group_logs(self) -> List[List[LogSummary]]:
        groups = []
        cur_group = []
        
        last_d = None

        for log in sorted(self.values(), key=self._sort_by_time):
            m = LOGNAME_RE.match(log.filename)
            
            if m:
                d = m["runname"]
            else:
                d = None
        
            if (last_d is None or d != last_d) and cur_group != []:
                groups.append(cur_group)
                cur_group = []

            cur_group.append(log)
            last_d = d

        if cur_group != []:
            groups.append(cur_group)

        return sorted(groups, key=lambda x: "".join(x[0].filename.split("-")[2:]))
