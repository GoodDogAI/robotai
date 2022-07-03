import os
import pandas as pd

DATAFRAME_FILE = "_hashes.json"


# Allows for quick and cached access to the SHA256 hash of a bunch of log files
class LogUtil:
    dir: str
    extension: str=".log"
    hashes: pd.DataFrame

    def __init__(self, dir):
        self.dir = dir
        self.update()

    def update(self):
        hashes = pd.read_json(os.path.join(self.dir, DATAFRAME_FILE))
