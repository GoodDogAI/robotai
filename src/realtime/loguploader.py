import re
import os
import requests

# Watches for newly completed log files in the logging directory, and uploads them

from src.logutil import LogHashes

config_vars = {}
with open(os.path.join(os.path.dirname(__file__), "..", "include", "config.h")) as f:
    b = re.findall(r"\#define +([a-zA-Z0-9_]+) +\"?([0-9a-zA-Z_/\-\'\:]+)\"?", f.read())

    for match in b:
        config_vars[match[0]] = match[1]

print(config_vars)

lh = LogHashes(config_vars["LOG_PATH"])

print(lh)