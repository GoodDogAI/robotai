import io
import os
import random
import string
import png
import time
import hashlib
import re

from typing import List
from datetime import datetime, timedelta

from fastapi import FastAPI, Depends, Form, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from cereal import log

from .config_web import RECORD_DIR
from src.logutil import LogHashes, LogSummary, validate_log
from ..video import load_image

app = FastAPI(title="RobotAI Model Service")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|jake-training-box)(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/models")
async def get_all_models() -> List[str]:
    return {
    "yolov7-tiny-s53": {
        "checkpoint": "/home/jake/robotai/_checkpoints/yolov7-tiny.pt",

        # Input dimensions must be divisible by the stride
        # In current situations, the image will be cropped to the nearest multiple of the stride
        "dimension_stride": 32,

        "intermediate_layer": "input.236",
        "intermediate_slice": 53,
    }
}
