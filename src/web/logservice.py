import os
import random
import string
from typing import List

from .config import RECORD_DIR
from fastapi import FastAPI, Depends, UploadFile, HTTPException
from pydantic import BaseModel
from src.logutil import LogHashes, LogSummary, asha256

app = FastAPI(title="RobotAI Log Service")
_loghashes = None


def get_loghashes() -> LogHashes:
    global _loghashes
    if _loghashes is None:
        _loghashes = LogHashes(RECORD_DIR)
    return _loghashes


@app.get("/logs")
async def list_logs(lh: LogHashes = Depends(get_loghashes)) -> List[LogSummary]:
    return lh.values()


@app.get("/logs/exists/{sha256}")
async def log_exists(sha256: str, lh: LogHashes = Depends(get_loghashes)):
    return lh.hash_exists(sha256)


@app.post("/logs")
async def post_log(logfile: UploadFile, lh: LogHashes = Depends(get_loghashes)):
    # Make sure the hash doesn't exist already
    sha256 = await asha256(logfile)

    if lh.hash_exists(sha256):
        raise HTTPException(status_code=500, detail="Log hash already exists")

    # Reset the logfile back to the beginning
    await logfile.seek(0)

    # Determine a new filename
    newfilename = os.path.join(lh.dir, logfile.filename)

    while os.path.exists(newfilename) or not newfilename.endswith(lh.extension):
        root, ext = os.path.splitext(logfile.filename)
        extra = ''.join(random.choices(string.ascii_letters, k=5))
        newfilename = os.path.join(lh.dir, f"{root}_{extra}{lh.extension}")

    # Copy over to the final location
    with open(newfilename, "wb") as fp:
        while byte_block := await logfile.read(65536):
            fp.write(byte_block)

    lh.update()


@app.get("/logs/{logfile}")
async def get_log(logfile: str, lh: LogHashes = Depends(get_loghashes)):
    return {}
