import os

from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from src.train.modelloader import create_and_validate_onnx

from src.config import HOST_CONFIG, MODEL_CONFIGS, BRAIN_CONFIGS


router = APIRouter(prefix="/models",
    tags=["models"],
)

@router.get("/")
async def get_all_models() -> JSONResponse:
    return MODEL_CONFIGS

@router.get("/brain/default")
async def get_default_brain_model() -> str:
    brain = HOST_CONFIG.DEFAULT_BRAIN_CONFIG

    return { brain: BRAIN_CONFIGS[brain] }

@router.get("/{model_name}/config")
async def get_model_config(model_name: str) -> JSONResponse:
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail="Model not found")

    return MODEL_CONFIGS[model_name]

@router.get("/{model_name}/onnx")
async def get_model_onnx(model_name: str) -> FileResponse:
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail="Model not found")

    onnx_path = create_and_validate_onnx(model_name)
    return FileResponse(path=onnx_path, media_type='application/octet-stream', filename=os.path.basename(onnx_path))
