import os
import pandas as pd
import hashlib

DATAFRAME_FILE = "_hashes.json"
DATAFRAME_COLUMNS = ["filename", "last_modified", "sha256"]

def sha256(filename: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


# Allows for quick and cached access to the SHA256 hash of a bunch of log files
class LogHashes:
    dir: str
    extension: str=".log"
    hashes: pd.DataFrame

    def __init__(self, dir):
        self.dir = dir
        self.update()

    def update(self):
        path = os.path.join(self.dir, DATAFRAME_FILE)
        if os.path.exists(path):
            self.hashes = pd.read_json(path)
        else:
            self.hashes = pd.DataFrame(columns=DATAFRAME_COLUMNS)

        for file in os.listdir(self.dir):
            if not file.endswith(self.extension):
                continue

            self.hashes[file] = sha256(os.path.join(self.dir, file))
