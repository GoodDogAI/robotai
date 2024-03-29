import os
import argparse
import psutil
import asyncio
import importlib
import logging
import signal
import multiprocessing
from setproctitle import setproctitle 

from typing import List, Dict

from src.config import DEVICE_CONFIG
from src.realtime.setup import prepare_brain_models


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
    args: List[str] = []
    running: bool = False
    delay: float = 0.0
    p = None
    
    def __init__(self, name: str, args: List[str] = [], delay: float = 0.0):
        self.name = name
        self.args = args
        self.delay = delay

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

    def __init__(self, name: str, module: str, func: str="main",  args: List[str] = [], delay: float=0.0, nice: int=0):
        super().__init__(name, args, delay)
        self.module = module
        self.func = func
        self.nice = nice

    async def start(self):
        await asyncio.sleep(self.delay)

        logger.warning(f"Starting {self.name}...")
        self.p = multiprocessing.Process(name=self.name, target=pythonlauncher, args=(self.name, self.module, self.func, self.nice))
        self.running = True
        self.p.start()

    async def join(self):
        asyncio.current_task().set_name(self.name)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._do_join)

    def _do_join(self):
        self.p.join()
        self.running = False
        
    def kill(self):
        self.p.terminate()
        self.running = False

class NativeProcess(ManagerProcess):
    nice: int = 0

    def __init__(self, name: str,  args: List[str] = [], delay: float=0.0, nice: int=0, affinity: List[int]=[]):
        super().__init__(name, args, delay)
        self.nice = nice
        self.affinity = affinity

    async def start(self):
        await asyncio.sleep(self.delay)
        
        logger.warning(f"Starting {self.name}...")
        self.p = await asyncio.create_subprocess_exec(os.path.abspath(f"build/{self.name}"), *self.args)

        setnice = psutil.Process(self.p.pid)
        setnice.nice(self.nice)

        # if self.nice < 0:
        #     os.sched_setscheduler(self.p.pid, os.SCHED_FIFO, os.sched_param(53))  # pylint: disable=no-member

        if len(self.affinity) > 0:
            os.sched_setaffinity(self.p.pid, self.affinity)

        self.running = True

    async def join(self):
        asyncio.current_task().set_name(self.name)
        await self.p.wait()
        self.running = False

    def kill(self):
        self.p.terminate()
        self.running = False

def get_procs(models: Dict[str,str]) -> List[ManagerProcess]:
    return [
        NativeProcess("camerad_v4l", nice=-10, delay=5.0), 
        NativeProcess("encoderd", ["head_color", "--maxqp", str(DEVICE_CONFIG.ENCODER_HEAD_COLOR_QP)]),
        NativeProcess("loggerd"),
        NativeProcess("micd"),
        NativeProcess("odrived"),
        NativeProcess("simplebgcd"),
        NativeProcess("appcontrold"),
        NativeProcess("braind", ["--config", os.path.join(DEVICE_CONFIG.MODEL_STORAGE_PATH, "brain_config.json"),
                                 "--vision_model", models["vision"],
                                 "--brain_model", models["brain"]]),
    ]


async def cancel_me():
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        raise
    finally:
        print("Cancellation complete")


async def brain_main():
    logger.warning("Setting up brain and models...")

    models = prepare_brain_models()
    procs = get_procs(models)

    logger.warning("Starting manager...")

    cancelation_task = asyncio.create_task(cancel_me())
    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, cancelation_task.cancel)
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, cancelation_task.cancel)

    # Start everything
    await asyncio.wait([proc.start() for proc in procs], return_when=asyncio.ALL_COMPLETED)

    # Wait for first to exit, or for the cancellation task
    done, pending = await asyncio.wait([cancelation_task] + [proc.join() for proc in procs], return_when=asyncio.FIRST_COMPLETED)

    print("Task finished", done)
    asyncio.get_running_loop().remove_signal_handler(signal.SIGINT)
    asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM)

    for proc in procs:
        try:
            if proc.running:
                print("killing ", proc.name)
                proc.kill()
                logger.info(f"Killed {proc.name}")
        except Exception as e:
            logger.exception(f"Could not kill {proc.name}")

    print("Waiting for rest of tasks to finish")
    cancelation_task.cancel()
    done, pending = await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)

    for proc in procs:
        print('{0: <16} ret {1}'.format(proc.name, proc.p.returncode))


async def uploader_main():
    # Log uploader runs seperately, until it finishes, also keep it lower priority
    log_uploader_proc = PythonProcess("loguploader", "src.realtime.loguploader", func="sync_once")

    # Run the log uploader until it is done, but not during realtime stuff
    logger.warning("Starting log uploader...")
    await log_uploader_proc.start()
    logger.warning("Waiting for log uploader to finish...")
    await log_uploader_proc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", action="store_true", help="repeat the brain_main loop indefinitely")
    opt = parser.parse_args()

    loop = asyncio.get_event_loop()   

    while True:
        loop.run_until_complete(brain_main())

        if not opt.repeat:
            break

    try:
        loop.run_until_complete(uploader_main())
    except KeyboardInterrupt:
        logger.warning("Got CTRL+C on Uploader")

    loop.close()


