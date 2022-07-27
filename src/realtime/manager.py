import os
import asyncio


class ManagerProcess:
    name: str = ""
    p = None

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

async def main():
    print("Starting manager...")

    for proc in procs:
        print(f"Starting {proc.name}...")
        proc.p = await asyncio.create_subprocess_exec(os.path.abspath(f"build/{proc.name}"))

    await asyncio.wait([proc.p.wait() for proc in procs], return_when=asyncio.FIRST_COMPLETED)

    for proc in procs:
        try:
            await proc.p.kill()
            print(f"Killed {proc.name}")
        except:
            print(f"Could not kill {proc.name}")

    print("Manager finished")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

