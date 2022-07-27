
import re
import os

from typing import Dict, Any

def load_realtime_config() -> Dict[str, Any]:
    config_vars = {}
    with open(os.path.join(os.path.dirname(__file__), "..", "include", "config.h")) as f:
        b = re.findall(r"\#define +([a-zA-Z0-9_]+) +\"?([0-9a-zA-Z_/\-\'\:\.]+)\"?", f.read())

        for match in b:
            config_vars[match[0]] = match[1]

    return config_vars
