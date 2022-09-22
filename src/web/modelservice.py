import os
import onnxruntime
import numpy as np
import io
import copy
from numpy.random import default_rng

from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response

from src.train.modelloader import create_and_validate_onnx, onnx_to_numpy_dtype, model_fullname, brain_fullname

from src.config import HOST_CONFIG, MODEL_CONFIGS, BRAIN_CONFIGS


router = APIRouter(prefix="/models",
    tags=["models"],
)

REFERENCE_MODEL_RNG_SEED = 17

def _get_reference_input(shape, dtype):
    rng = default_rng(REFERENCE_MODEL_RNG_SEED)

    if dtype == np.uint8:
        return rng.integers(low=0, high=255, size=shape, dtype=dtype)
    elif dtype == np.float32 or dtype == np.float64:
        return rng.random(size=shape, dtype=dtype)
    else:
        raise NotImplementedError()


@router.get("/")
async def get_all_models() -> JSONResponse:
    return { name: model | {"_fullname": model_fullname(model)} for name, model in
        MODEL_CONFIGS.items()
    }

@router.get("/brain/default")
async def get_default_brain_model() -> str:
    brain = HOST_CONFIG.DEFAULT_BRAIN_CONFIG
    
    config = copy.deepcopy(BRAIN_CONFIGS[brain])
    config["_fullname"] = brain_fullname(brain)
    
    for type, model in config["models"].items():
        config["models"][type] = {"basename": model, "_fullname": model_fullname(MODEL_CONFIGS[model])}

    return { brain: config }

@router.get("/{model_name}/config")
async def get_model_config(model_name: str) -> JSONResponse:
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail="Model not found")

    return MODEL_CONFIGS[model_name] | {"_fullname": model_fullname(MODEL_CONFIGS[model_name])}

@router.get("/{model_name}/onnx")
async def get_model_onnx(model_name: str) -> FileResponse:
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail="Model not found")

    config = MODEL_CONFIGS[model_name]
    onnx_path = create_and_validate_onnx(config)
    return FileResponse(path=onnx_path, media_type="application/octet-stream", filename=os.path.basename(onnx_path),
                        headers={"X-ModelFullname": model_fullname(config)})

@router.get("/{model_name}/reference_input/{input_name}")
async def get_model_reference_input(model_name: str, input_name: str) -> FileResponse:
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail="Model not found")

    config = MODEL_CONFIGS[model_name]
    onnx_path = create_and_validate_onnx(config)
    ort_sess = onnxruntime.InferenceSession(onnx_path)

    all_input_shapes = {x.name:x.shape for x in ort_sess.get_inputs()}
    all_input_dtypes = {x.name:onnx_to_numpy_dtype(x.type) for x in ort_sess.get_inputs()}

    if input_name not in all_input_shapes:
        raise HTTPException(status_code=404, detail="Input name not found in model ONNX")

    shape = all_input_shapes[input_name]
    dtype = all_input_dtypes[input_name]
   
    data = _get_reference_input(shape, dtype)

    file_data = io.BytesIO()
    np.save(file_data, data)

    return Response(content=file_data.getvalue(),
                    media_type="application/octet-stream",
                    headers={"X-ModelFullname": model_fullname(config)})
    
    
@router.get("/{model_name}/reference_output/{output_name}")
async def get_model_reference_input(model_name: str, output_name: str) -> FileResponse:
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail="Model not found")

    config = MODEL_CONFIGS[model_name]
    onnx_path = create_and_validate_onnx(config)
    ort_sess = onnxruntime.InferenceSession(onnx_path)

    feed_dict = {}
    for input in ort_sess.get_inputs():
        feed_dict[input.name] = _get_reference_input(input.shape, onnx_to_numpy_dtype(input.type))

    try:
        ort_outputs = ort_sess.run([output_name], feed_dict)
    except onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument:
        raise HTTPException(status_code=404, detail="Output name not found in model ONNX")

    assert len(ort_outputs) == 1

    file_data = io.BytesIO()
    np.save(file_data, ort_outputs[0])

    return Response(content=file_data.getvalue(),
                media_type="application/octet-stream",
                headers={"X-ModelFullname": model_fullname(config)})

