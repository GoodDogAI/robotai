import re
import os
import time
import requests

from typing import Dict, Any

# Watches for newly completed log files in the logging directory, and uploads them

from src.logutil import LogHashes

def load_realtime_config() -> Dict[str, Any]:
    config_vars = {}
    with open(os.path.join(os.path.dirname(__file__), "..", "include", "config.h")) as f:
        b = re.findall(r"\#define +([a-zA-Z0-9_]+) +\"?([0-9a-zA-Z_/\-\'\:]+)\"?", f.read())

        for match in b:
            config_vars[match[0]] = match[1]

    return config_vars


CONFIG = load_realtime_config()

lh = LogHashes(CONFIG["LOG_PATH"])

print(lh)

def main():
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()