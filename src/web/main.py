from fastapi import FastAPI, Depends, Form, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from . import logservice, modelservice

app = FastAPI(title="RobotAI Log Service")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|jake-training-box)(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(logservice.router)
app.include_router(modelservice.router)