import os
import random
import string
from typing import List

from .config import RECORD_DIR
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from src.logutil import LogHashes, LogSummary, asha256

app = FastAPI(title="RobotAI Log Service")
loghashes = LogHashes(RECORD_DIR)


@app.get("/logs")
async def list_logs() -> List[LogSummary]:
    return loghashes.values()


@app.get("/logs/exists/{sha256}")
async def log_exists(sha256: str):
    return loghashes.hash_exists(sha256)


@app.post("/logs")
async def put_log(logfile: UploadFile):
    # Make sure the hash doesn't exist already
    sha256 = await asha256(logfile)

    if loghashes.hash_exists(sha256):
        raise HTTPException(status_code=500, detail="Log hash already exists")

    # Reset the logfile back to the beginning
    await logfile.seek(0)

    # Determine a new filename
    newfilename = os.path.join(RECORD_DIR, logfile.filename)

    while os.path.exists(newfilename):
        root, ext = os.path.splitext(logfile.filename)
        extra = ''.join(random.choices(string.ascii_letters, k=5))
        newfilename = os.path.join(RECORD_DIR, f"{root}_{extra}{ext}")

    # Copy over to the final location
    with open(newfilename, "wb") as fp:
        while byte_block := await logfile.read(65536):
            fp.write(byte_block)

    loghashes.update()


@app.get("/logs/{logfile}")
async def get_log(logfile: str):
    return {}
