import re
import os
import time
import requests
import logging

from typing import Dict, Any
from inotify_simple import INotify, flags
from src.logutil import LogHashes

# Watches for newly completed log files in the logging directory, and uploads them

logger = logging.getLogger(__name__)


def load_realtime_config() -> Dict[str, Any]:
    config_vars = {}
    with open(os.path.join(os.path.dirname(__file__), "..", "include", "config.h")) as f:
        b = re.findall(r"\#define +([a-zA-Z0-9_]+) +\"?([0-9a-zA-Z_/\-\'\:\.]+)\"?", f.read())

        for match in b:
            config_vars[match[0]] = match[1]

    return config_vars


CONFIG = load_realtime_config()

def main():
    req = requests.get(CONFIG["LOG_SERVICE"] + "/logs")
    if req.status_code != 200:
        logger.warn("Warning, unable to connect to logservice")

    lh = LogHashes(CONFIG["LOG_PATH"])

    inotify = INotify()
    watch_flags = flags.CREATE | flags.DELETE | flags.MOVED_TO | flags.MOVED_FROM
    wd = inotify.add_watch(lh.dir, watch_flags)

    while True:
        if any(e.name.endswith(lh.extension) for e in inotify.read(timeout=1000, read_delay=100)):
            print(f"Got inotify updating lhes")
            lh.update()


    inotify.close()
    print("DONE")

if __name__ == "__main__":
    main()