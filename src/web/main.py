import os
from .config import RECORD_DIR
from fastapi import FastAPI

app = FastAPI()



@app.get("/logs")
async def list_logs():
    result = []

    for file in os.listdir(RECORD_DIR):
        if not file.endswith(".log"):
            continue

        result.append({
            "filename": os.path.basename(file),
            "hash": "0",
        })

    return result


@app.get("/logs/exists/{sha256}")
async def log_exists(sha256: str):
    return False

@app.put("/logs/{logfile}")
async def put_log(logfile: str) -> bool:
    return False

@app.get("/logs/{logfile}")
async def get_log(logfile: str):
    return {}
