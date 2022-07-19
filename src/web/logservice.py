import io
import os
import random
import string
import png

from typing import List

from fastapi import FastAPI, Depends, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from cereal import log
import cereal.messaging as messaging

from .config import RECORD_DIR
from src.logutil import LogHashes, LogSummary, asha256, validate_log
from .video import load_image

app = FastAPI(title="RobotAI Log Service")
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://jake-training-box:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    logfile.file.seek(0, io.SEEK_END)
    if logfile.file.tell() < 1:
        raise HTTPException(status_code=400, detail="Empty file")

    # Check that you can read all messages
    logfile.file.seek(0)
    if not validate_log(logfile.file):
        raise HTTPException(status_code=400, detail="Log file is not a serialized capnp buffer")

    # Determine a new filename
    newfilename = os.path.join(lh.dir, logfile.filename)

    while os.path.exists(newfilename) or not newfilename.endswith(lh.extension):
        root, ext = os.path.splitext(logfile.filename)
        extra = ''.join(random.choices(string.ascii_letters, k=5))
        newfilename = os.path.join(lh.dir, f"{root}_{extra}{lh.extension}")

    # Copy over to the final location
    logfile.file.seek(0)
    with open(newfilename, "wb") as fp:
        while byte_block := await logfile.read(65536):
            fp.write(byte_block)

    lh.update()


@app.get("/logs/{logfile}")
async def get_log(logfile: str, lh: LogHashes = Depends(get_loghashes)) -> JSONResponse:
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    result = []

    with open(os.path.join(lh.dir, logfile), "rb") as f:
        events = log.Event.read_multiple(f)

        for evt in events:
            result.append(evt.to_dict())

    # Don't try to encode raw data fields in the json,
    # it will just get interpreted as utf-8 text and you will have a bad time
    return JSONResponse(jsonable_encoder(result,
                        custom_encoder={bytes: lambda data_obj: None}))

@app.get("/logs/{logfile}/frame/{frameid}")
async def get_log_frame(logfile: str, frameid: int, lh: LogHashes = Depends(get_loghashes)):
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    rgb = load_image(os.path.join(RECORD_DIR, logfile), frameid)
    img = png.from_array(rgb, 'RGB')
    img_data = io.BytesIO()
    img.write(img_data)
    print(img.info, img.rows)
    return Response(content=img_data.getvalue(), media_type="image/png")