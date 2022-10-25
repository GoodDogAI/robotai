import io
import os
import subprocess
import tempfile
import hashlib
import shutil
import numpy as np

from typing import List

from fastapi import APIRouter, Depends, Form, UploadFile, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse, Response, HTMLResponse
from fastapi.templating import Jinja2Templates


from src.config import HOST_CONFIG, MODEL_CONFIGS, DEVICE_CONFIG

from src.web.dependencies import get_loghashes
from src.logutil import LogHashes, LogSummary, quick_validate_log
from src.video import get_image_packets, decode_last_frame
from src.train import log_validation

from cereal import log

router = APIRouter()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


@router.get("/", response_class=HTMLResponse)
async def homepage(request: Request, lh: LogHashes = Depends(get_loghashes)):
    logs = lh.group_logs()
    return templates.TemplateResponse("index.html", {"request": request,
                                                     "log_groups": logs})

@router.get("/page/{logfile}")
async def homepage(request: Request, logfile: str, lh: LogHashes = Depends(get_loghashes)):
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    logdata = []

    with open(os.path.join(lh.dir, logfile), "rb") as f:
        events = log.Event.read_multiple(f)

        for index, evt in enumerate(events):
            which = evt.which()

            logdata.append({
                "index": index,
                "which": which,
                "total_size_bytes": evt.total_size.word_count * 8
            })

    return templates.TemplateResponse("log.html", {"request": request,
                                                     "logdata": logdata})
