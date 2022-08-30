import os

from src.config.config import DEVICE_CONFIG

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "config.h"), "w") as f:
        f.write("#pragma once\n\n")
        f.write("// Warning: This file is autogenerated from config.py, do not edit it!\n\n")
        
        for key, value in DEVICE_CONFIG.items():
            if isinstance(value, str):
                f.write(f"#define {key} \"{value}\"\n")
            elif isinstance(value, int):
                f.write(f"#define {key} {value}\n")
            elif isinstance(value, float):
                f.write(f"#define {key} {value:.12f}f\n")
            else:
                pass