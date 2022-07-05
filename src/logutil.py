import os
import hashlib
import json
import logging

from typing import List, Dict

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, parse_file_as

logger = logging.getLogger(__name__)
DATA_FILE = "_hashes.json"


def sha256(filename: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


class LogSummary(BaseModel):
    filename: str
    sha256: str
    last_modified: int


# Allows for quick and cached access to the SHA256 hash of a bunch of log files
class LogHashes:
    dir: str
    extension: str=".log"
    hashes: Dict[str, LogSummary] = {}

    def __init__(self, dir):
        self.dir = dir
        self.update()

    def update(self):
        path = os.path.join(self.dir, DATA_FILE)
        if os.path.exists(path):
            self.hashes = parse_file_as(Dict[str, LogSummary], path)
        else:
            self.hashes = {}

        for file in os.listdir(self.dir):
            if not file.endswith(self.extension):
                continue
            filepath = os.path.join(self.dir, file)
            mtime = round(os.path.getmtime(filepath) * 1e9)

            if file in self.hashes and self.hashes[file].last_modified == mtime:
                continue

            logger.info(f"Hashing {filepath}")
            self.hashes[file] = LogSummary(filename=file, sha256=sha256(filepath), last_modified=mtime)

        with open(path + "_temp", "w") as f:
            json.dump(jsonable_encoder(self.hashes), f)

        if os.path.exists(path):
            os.remove(path)
        os.rename(path + "_temp", path)

    def values(self) -> List[LogSummary]:
        return self.hashes.values()

    def hash_exists(self, hash:str) -> bool:
        return hash in self.hashes