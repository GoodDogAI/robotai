import os
import subprocess


class ManagerProcess:
    name: str = ""
    p: subprocess.Popen

    def __init__(self, name):
        self.name = name


class PythonProcess(ManagerProcess):
    pass

class NativeProcess(ManagerProcess):
    def __init__(self, name):
        super().__init__(name)

procs = [
    NativeProcess("camerad"), 
    NativeProcess("encoderd"),
    NativeProcess("loggerd"),
]

if __name__ == "__main__":
    print("Starting manager...")

    for proc in procs:
        print(f"Starting {proc.name}...")
        proc.p = subprocess.Popen(os.path.abspath(f"build/{proc.name}"))

    for proc in procs:
        proc.p.wait()


