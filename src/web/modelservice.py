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

from fastapi import APIRouter, Depends, Form, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response

from cereal import log

from src.config import HOST_CONFIG, MODEL_CONFIGS, BRAIN_CONFIGS
from src.logutil import LogHashes, LogSummary, validate_log
from ..video import load_image

router = APIRouter(prefix="/models",
    tags=["models"],
)

@router.get("/")
async def get_all_models() -> JSONResponse:
    return MODEL_CONFIGS

@router.get("/brain/default")
async def get_default_brain_model() -> str:
    brain = HOST_CONFIG.DEFAULT_BRAIN_CONFIG

    return {brain: BRAIN_CONFIGS[brain]}

@router.get("/{model_name}")
async def get_model(model_name: str) -> JSONResponse:
    return MODEL_CONFIGS[model_name]