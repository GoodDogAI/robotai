import os
import pyarrow
import hashlib

from pathlib import Path

from config import HOST_CONFIG
from src.logutil import LogHashes, LogSummary

# This class takes in a directory, and runs models on all of the video frames, saving
# the results to an arrow cache file. The keys are the videofilename-frameindex.
class ArrowModelCache():
    def __init__(self, dir: str, model_fullname: str):
        self.lh = LogHashes(dir)
        self.model_fullname = model_fullname
        Path(HOST_CONFIG.CACHE_DIR, "arrow", self.model_fullname).mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, log: LogSummary):
        return os.path.join(HOST_CONFIG.CACHE_DIR, "arrow", self.model_fullname, log.filename + ".arrow")

    def build_cache(self):
        for group in self.lh.group_logs():
            print(f"Building cache for {group}")

    