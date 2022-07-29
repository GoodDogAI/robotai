import io
import os
import random
import string
import png
import time
import hashlib

from typing import List

from fastapi import FastAPI, Depends, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from cereal import log
import cereal.messaging as messaging

from .config_web import RECORD_DIR
from src.logutil import LogHashes, LogSummary, validate_log
from .video import load_image

app = FastAPI(title="RobotAI Log Service")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|jake-training-box)(:[0-9]+)?",
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


async def asha256(fp: UploadFile) -> str:
    sha256_hash = hashlib.sha256()
    while byte_block := await fp.read(65536):
        sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


@app.get("/logs")
async def list_logs(lh: LogHashes = Depends(get_loghashes)) -> List[LogSummary]:
    return sorted(lh.values())


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
    await logfile.close()


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

    start = time.perf_counter()
    rgb = load_image(os.path.join(RECORD_DIR, logfile), frameid)
    img = png.from_array(rgb, 'RGB', info={'bitdepth': 8})
    img_data = io.BytesIO()
    img.write(img_data)
    response = Response(content=img_data.getvalue(), media_type="image/png")
    #print(f"Took {round(1000 * (time.perf_counter() - start))} ms")

    return response

@app.get("/logs/{logfile}/thumbnail")
async def get_log_thumbnail(logfile: str, lh: LogHashes = Depends(get_loghashes)):
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    with open(os.path.join(RECORD_DIR, logfile), "rb") as f:
        events = log.Event.read_multiple(f)

        for i, evt in enumerate(events):
            if evt.which() == "headEncodeData":
                if evt.headEncodeData.idx.flags & 0x8:
                    packet = evt.headEncodeData.data
                    return Response(content=packet, media_type="image/heic")

    raise HTTPException(status_code=500, detail="Unable to find thumbnail")
