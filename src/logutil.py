import os
import hashlib
import json
import logging

from typing import List, Dict, BinaryIO
from functools import total_ordering
from capnp.lib import capnp
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, parse_file_as
from cereal import log
from typing import Optional


logger = logging.getLogger(__name__)
DATA_FILE = "_hashes.json"


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

@total_ordering
class LogSummary(BaseModel):
    filename: str
    orig_sha256: Optional[str]
    sha256: str
    last_modified: int

    def __init__(self, **data):
        super().__init__(**data)

    def __le__(self, other):
        return self.filename < other.filename


# Allows for quick and cached access to the SHA256 hash of a bunch of log files
class LogHashes:
    dir: str
    extension: str = ".log"
    files: Dict[str, LogSummary] = {}

    def __init__(self, dir):
        self.dir = dir
        self.update()

    def update(self, original_hashes: Dict[str, str] = {}):
        path = os.path.join(self.dir, DATA_FILE)
        if os.path.exists(path):
            self.existing_files = parse_file_as(Dict[str, LogSummary], path)
        else:
            self.existing_files = {}

        self.files = {}

        for file in os.listdir(self.dir):
            if not file.endswith(self.extension):
                continue
            filepath = os.path.join(self.dir, file)
            mtime = round(os.path.getmtime(filepath) * 1e9)

            if file in self.existing_files and self.existing_files[file].last_modified == mtime:
                self.files[file] = self.existing_files[file]
                continue

            logger.warning(f"Hashing {filepath}")
    
            new_sha = sha256(filepath)

            if file in original_hashes:
                orig_sha = original_hashes[file]
            elif file in self.existing_files:
                orig_sha = self.existing_files[file].orig_sha256
            else:
                orig_sha = new_sha

            self.files[file] = LogSummary(filename=file, sha256=new_sha, orig_sha256=orig_sha, last_modified=mtime)

        with open(path + "_temp", "w") as f:
            json.dump(jsonable_encoder(self.files), f)

        if os.path.exists(path):
            os.remove(path)
        os.rename(path + "_temp", path)

    def values(self) -> List[LogSummary]:
        return list(self.files.values())

    def hash_exists(self, hash:str) -> bool:
        return hash in {f.sha256 for f in self.files.values()} | {f.orig_sha256 for f in self.files.values()}

    def filename_exists(self, filename:str) -> bool:
        return filename in self.files

    def __str__(self):
        return str(self.files)
