import io
import os
import random
import string
import png
import time
import hashlib
import re

from typing import List

from fastapi import APIRouter, Depends, Form, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response

from cereal import log
from src.config import HOST_CONFIG

from src.web.dependencies import get_loghashes
from src.logutil import LogHashes, LogSummary, quick_validate_log
from src.video import get_image_packets, decode_last_frame
import src.PyNvCodec as nvc

router = APIRouter(prefix="/logs",
    tags=["logs"],
)


async def asha256(fp: UploadFile) -> str:
    sha256_hash = hashlib.sha256()
    while byte_block := await fp.read(65536):
        sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


@router.get("/")
async def list_logs(lh: LogHashes = Depends(get_loghashes)) -> List[List[LogSummary]]:
    groups = []
    cur_group = []
    logname_re = re.compile(r"([a-z]+)-([a-f0-9]+)-(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2})_(\d{1,2}).log")
    last_d = None

    for log in sorted(lh.values()):
        m = logname_re.match(log.filename)
        
        if m:
            d = m[2]
        else:
            d = None
    
        if (last_d is None or d != last_d) and cur_group != []:
            groups.append(cur_group)
            cur_group = []

        cur_group.append(log)
        last_d = d

    if cur_group != []:
        groups.append(cur_group)

    return groups


@router.get("/exists/{sha256}")
async def log_exists(sha256: str, lh: LogHashes = Depends(get_loghashes)):
    return lh.hash_exists(sha256)


@router.post("/")
async def post_log(logfile: UploadFile, sha256: str=Form(), lh: LogHashes = Depends(get_loghashes)):
    # Make sure the hash doesn't exist already
    local_hash = await asha256(logfile)

    if lh.hash_exists(local_hash):
        raise HTTPException(status_code=500, detail="Log hash already exists")

    if sha256 != local_hash:
        raise HTTPException(status_code=400, detail="SHA256 did not match expected value")

    logfile.file.seek(0, io.SEEK_END)
    if logfile.file.tell() < 1:
        raise HTTPException(status_code=400, detail="Empty file")

    # Check that you can read all messages
    logfile.file.seek(0)
    if not quick_validate_log(logfile.file):
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


@router.get("/{logfile}")
async def get_log(logfile: str, lh: LogHashes = Depends(get_loghashes)) -> JSONResponse:
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    result = []

    with open(os.path.join(lh.dir, logfile), "rb") as f:
        events = log.Event.read_multiple(f)

        for evt in events:
            data = evt.to_dict()
            
            # Cut out some hard datafields in certain message types that would
            # otherwise make for huge downloads
            which = evt.which()
            if which == "micData":
                del data["micData"]["data"]

            # Add in some sizing metadata
            data["_total_size_bytes"] = evt.total_size.word_count * 8
            result.append(data)

    # Don't try to encode raw data fields in the json,
    # it will just get interpreted as utf-8 text and you will have a bad time
    return JSONResponse(jsonable_encoder(result,
                        custom_encoder={bytes: lambda data_obj: None}))

@router.get("/{logfile}/frame/{frameid}")
async def get_log_frame(logfile: str, frameid: int, lh: LogHashes = Depends(get_loghashes)):
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    start = time.perf_counter()

    try:
        packets = get_image_packets(os.path.join(HOST_CONFIG.RECORD_DIR, logfile), frameid)
    except KeyError:
        raise HTTPException(status_code=404, detail="Frame not found")
        
    rgb = decode_last_frame(packets, pixel_format=nvc.PixelFormat.RGB)
    img = png.from_array(rgb, 'RGB', info={'bitdepth': 8})
    img_data = io.BytesIO()
    img.write(img_data)
    response = Response(content=img_data.getvalue(), media_type="image/png")
    #print(f"Took {round(1000 * (time.perf_counter() - start))} ms")

    return response

@router.get("/{logfile}/thumbnail")
async def get_log_thumbnail(logfile: str, lh: LogHashes = Depends(get_loghashes)):
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    with open(os.path.join(HOST_CONFIG.RECORD_DIR, logfile), "rb") as f:
        events = log.Event.read_multiple(f)

        for i, evt in enumerate(events):
            if evt.which() == "headEncodeData":
                if evt.headEncodeData.idx.flags & 0x8:
                    packet = evt.headEncodeData.data
                    return Response(content=packet, media_type="image/heic")

    raise HTTPException(status_code=500, detail="Unable to find thumbnail")
