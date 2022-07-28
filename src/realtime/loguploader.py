import re
import os
import time
import requests
import logging

from typing import Dict, Any
from inotify_simple import INotify, flags
from src.logutil import LogHashes
from src.include.config import load_realtime_config

# Watches for newly completed log files in the logging directory, and uploads them

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
CONFIG = load_realtime_config()

def sync(lh: LogHashes) -> bool:
    try:
        all_logs = requests.get(CONFIG["LOG_SERVICE"] + "/logs")
        if all_logs.status_code != 200:
            logger.warn("Warning, unable to connect to logservice")
            return False
        all_hashes = set(x["sha256"] for x in all_logs.json())

        for ls in lh.values():
            if ls.sha256 not in all_hashes:
                with open(os.path.join(lh.dir, ls.filename), "rb") as f:
                    result = requests.post(CONFIG["LOG_SERVICE"] + "/logs", files={"logfile": f})

                if result.status_code != 200:
                    logger.warn(f"Warning, unable to upload {ls} response code {result.status_code}")
                    return False
                
                logger.info(f"Uploaded {ls} successfully")

        return True
    except requests.ConnectionError:
        logger.error(f"Could not connect to {CONFIG['LOG_SERVICE']} in order to sync logs, will try again later")
    finally:
        return False

def main():
    lh = LogHashes(CONFIG["LOG_PATH"])

    sync(lh)

    inotify = INotify()
    watch_flags = flags.CREATE | flags.DELETE | flags.MOVED_TO | flags.MOVED_FROM
    wd = inotify.add_watch(lh.dir, watch_flags)

    while True:
        if any(e.name.endswith(lh.extension) for e in inotify.read(timeout=1000, read_delay=100)):
            logger.warn(f"Got inotify updating lhes")
            lh.update()
            sync(lh)


    inotify.close()
    logger.warn("DONE")

if __name__ == "__main__":
    main()