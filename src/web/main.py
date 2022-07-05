import os
from typing import List

from .config import RECORD_DIR
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from src.logutil import LogHashes

app = FastAPI()
loghashes = LogHashes(RECORD_DIR)



@app.get("/logs")
async def list_logs() -> List[LogSummary]:
    return loghashes.values()


@app.get("/logs/exists/{sha256}")
async def log_exists(sha256: str):
    return loghashes.hash_exists(sha256)

@app.put("/logs/{logfile}")
async def put_log(logfile: UploadFile) -> bool:
    # Make sure the hash doesn't exist already

    # Copy it over to the log location, adding new name if necessary
    return False

@app.get("/logs/{logfile}")
async def get_log(logfile: str):
    return {}
