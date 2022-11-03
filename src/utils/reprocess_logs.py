import glob
import os
from src.config import HOST_CONFIG
from src.logutil import resort_log_monotonic, check_log_monotonic


for file in glob.glob(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "*.log")):
    with open(file, "rb") as i, open(file + ".fixed", "w+b") as o:
        resort_log_monotonic(i, o)

        o.flush()
        o.seek(0)
        if not check_log_monotonic(o):
            print("Failed to fix " + file)
        else:
            print("Fixed " + file)