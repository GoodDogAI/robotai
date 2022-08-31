import os
import onnxruntime
import numpy as np
import io
from numpy.random import default_rng

from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response

from src.train.modelloader import create_and_validate_onnx

from src.config import HOST_CONFIG, MODEL_CONFIGS, BRAIN_CONFIGS


router = APIRouter(prefix="/models",
    tags=["models"],
)

REFERENCE_MODEL_RNG_SEED = 17


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
    return FileResponse(path=onnx_path, media_type="application/octet-stream", filename=os.path.basename(onnx_path))

@router.get("/{model_name}/reference_input/{input_name}")
async def get_model_reference_input(model_name: str, input_name: str) -> FileResponse:
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail="Model not found")

    onnx_path = create_and_validate_onnx(model_name)
    ort_sess = onnxruntime.InferenceSession(onnx_path)

    all_inputs = {x.name:x.shape for x in ort_sess.get_inputs()}

    if input_name not in all_inputs:
        raise HTTPException(status_code=404, detail="Input name not found in model ONNX")

    shape = all_inputs[input_name]
    rng = default_rng(REFERENCE_MODEL_RNG_SEED)
    data = rng.random(shape)

    file_data = io.BytesIO()
    np.save(file_data, data)

    return Response(content=file_data.getvalue(), media_type="application/octet-stream")
    
    
@router.get("/{model_name}/reference_output/{output_name}")
async def get_model_reference_input(model_name: str, output_name: str) -> FileResponse:
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail="Model not found")
