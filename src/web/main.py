import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from . import logservice, modelservice

app = FastAPI(title="RobotAI Log Service")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|jake-training-box)(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(logservice.router)
app.include_router(modelservice.router)