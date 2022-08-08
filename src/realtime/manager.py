import os
import asyncio
import importlib
import logging
import multiprocessing
from setproctitle import setproctitle 

# Manager script to schedule and run a bunch of workers with various configurations
# Loosely based on: https://github.com/commaai/openpilot/blob/master/selfdrive/manager
logging.basicConfig()
logger = logging.getLogger(__name__)

def pythonlauncher(name: str, module: str, func: str, nice: int=0):
    # import the process
    logger.info(f"Importing {name} {module}")
    mod = importlib.import_module(module)

    # rename the process
    setproctitle(name)

    # set the niceness level
    os.nice(nice)

    # exec the process
    logger.info(f"Launching {name} {module}.{func}")
    getattr(mod, func)()

class ManagerProcess:
    name: str = ""
    running: bool = False
    p = None
    
    def __init__(self, name: str):
        self.name = name

    async def start(self):
        raise NotImplementedError()

    async def join(self):
        raise NotImplementedError()

    def kill(self):
        raise NotImplementedError()

class PythonProcess(ManagerProcess):
    module: str = ""
    func: str = ""
    nice: int = 0

    def __init__(self, name: str, module: str, func: str="main", nice: int=0):
        super().__init__(name)
        self.module = module
        self.func = func
        self.nice = nice

    async def start(self):
        self.p = multiprocessing.Process(name=self.name, target=pythonlauncher, args=(self.name, self.module, self.func, self.nice))
        self.running = True
        self.p.start()

    async def join(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._do_join)

    def _do_join(self):
        self.p.join()
        self.running = False
        
    def kill(self):
        self.p.terminate()
        self.running = False

class NativeProcess(ManagerProcess):
    def __init__(self, name: str):
        super().__init__(name)

    async def start(self):
        self.p = await asyncio.create_subprocess_exec(os.path.abspath(f"build/{self.name}"))
        self.running = True

    async def join(self):
        await self.p.wait()
        self.running = False

    def kill(self):
        self.p.kill()
        self.running = False

procs = [
    NativeProcess("camerad"), 
    NativeProcess("encoderd"),
    NativeProcess("loggerd"),
    NativeProcess("micd"),
    NativeProcess("odrived"),
    NativeProcess("simplebgcd"),
    PythonProcess("loguploader", "src.realtime.loguploader", nice=15), # Keep it lower priority
]

async def main():
    logger.warning("Starting manager...")

    for proc in procs:
        logger.warning(f"Starting {proc.name}...")
        await proc.start()

    done, pending = await asyncio.wait([proc.join() for proc in procs], return_when=asyncio.FIRST_COMPLETED)

    logger.warning(f"Done: {done}")

    for proc in procs:
        try:
            if proc.running:
                print("killing ", proc.name)
                proc.kill()
                logger.info(f"Killed {proc.name}")
        except Exception as e:
            logger.exception(f"Could not kill {proc.name}")

    logger.info("Manager finished")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

