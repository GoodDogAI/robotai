import os
import hashlib
import json

DATA_FILE = "_hashes.json"

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
    hashes: dict={}

    def __init__(self, dir):
        self.dir = dir
        self.update()

    def update(self):
        path = os.path.join(self.dir, DATA_FILE)
        if os.path.exists(path):
            self.hashes = json.load(path)
        else:
            self.hashes = {}

        for file in os.listdir(self.dir):
            if not file.endswith(self.extension):
                continue
            filepath = os.path.join(self.dir, file)

            self.hashes[file] = {
                "last_modified": round(os.path.getmtime(filepath) * 1e9),
                "sha256": sha256(filepath),
            }

        with open(path, "w") as f:
            json.dump(self.hashes, f)
