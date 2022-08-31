import torch
import onnx
import onnxruntime
import os
import time
import importlib
import hashlib
import numpy as np
from pathlib import Path

from itertools import chain
import polygraphy
import polygraphy.backend.trt

from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, EngineFromBytes, SaveEngine, TrtRunner
from polygraphy.cuda import DeviceView


from src.config import DEVICE_CONFIG, HOST_CONFIG, MODEL_CONFIGS



MODEL_MATCH_RTOL = 1e-4
MODEL_MATCH_ATOL = 1e-4

def onnx_to_numpy_dtype(onnx_type: str) -> np.dtype:
    if onnx_type == "tensor(float)":
        return np.float32
    elif onnx_type == "tensor(int32)":
        return np.int32
    elif onnx_type == "tensor(int64)":
        return np.int64
    else:
        raise ValueError(f"Unsupported ONNX type {onnx_type}")


# Returns a unique path that includes a hash of the model config and checkpoint
def model_basename(config_name: str) -> str:
    config = MODEL_CONFIGS[config_name]
    assert config is not None, "Unable to find config"
    sha256_hash = hashlib.sha256()

    with open(config["checkpoint"], "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)

    # Also include the hash of the config itself
    sha256_hash.update(repr(config).encode("utf-8"))
    model_sha = sha256_hash.hexdigest()[:16]

    model_basename = os.path.basename(config["checkpoint"]).replace(".pt", "") + "-" + model_sha + ""
    return model_basename


def validate_pt_onnx(pt_model: torch.nn.Module, onnx_path: str) -> bool:
    ort_sess = onnxruntime.InferenceSession(onnx_path)
    assert len(ort_sess.get_inputs()) == 1, "ONNX model must have a single input"
    ort_shape = ort_sess.get_inputs()[0].shape
    random_input = torch.FloatTensor(*ort_shape).uniform_(0.0, 1.0)

    torch_output = pt_model(random_input.to("cuda"))
    ort_outputs = ort_sess.run(None, {'input.1': random_input.cpu().numpy()})

    # Check that the outputs are the same
    for i, torch_output in enumerate(chain(*torch_output)):
        torch_output = torch_output.detach().cpu().numpy()
        ort_output = ort_outputs[i]

        matches = np.isclose(ort_output, torch_output, rtol=MODEL_MATCH_RTOL, atol=MODEL_MATCH_ATOL).sum()
        print(f"PT-ONNX Output {i} matches: {matches / torch_output.size:.3%}")

        assert np.allclose(ort_output, torch_output, rtol=MODEL_MATCH_RTOL, atol=MODEL_MATCH_ATOL), f"Output mismatch {i}"

    return True   

def validate_onnx_trt(onnx_path: str, trt_path: str) -> bool:
    with open(trt_path, "rb") as f:
        build_engine = EngineFromBytes(f.read())

    ort_sess = onnxruntime.InferenceSession(onnx_path)
    assert len(ort_sess.get_inputs()) == 1, "ONNX model must have a single input"
    assert ort_sess.get_inputs()[0].type == "tensor(float)", "ONNX model must have a single input of type float"
    ort_shape = ort_sess.get_inputs()[0].shape

    random_input = torch.FloatTensor(*ort_shape).uniform_(0.0, 1.0).to("cuda")
    ort_outputs = ort_sess.run(None, {'input.1': random_input.cpu().numpy()})

    with TrtRunner(build_engine) as runner:
        assert len(runner.get_input_metadata()) == 1, "TRT model must have a single input"
        trt_input_name = next(iter(runner.get_input_metadata()))
        trt_input_shape = runner.get_input_metadata()[trt_input_name].shape
        assert ort_shape == trt_input_shape, "Input shape mismatch"
        assert runner.get_input_metadata()[trt_input_name].dtype == np.float32, "TRT model must have a single input of type float"

        start = time.perf_counter()

        for i in range(100):
            trt_outputs = runner.infer({
                trt_input_name: DeviceView(random_input.data_ptr(), random_input.shape, np.float32)
            })

        print(f"TRT inference time: {(time.perf_counter() - start)/100:.3f}s")

        # Check that the outputs are the same
        for index, ort_output_metadata in enumerate(ort_sess.get_outputs()):
            ort_output = ort_outputs[index]
            trt_output = trt_outputs[ort_output_metadata.name]

            matches = np.isclose(ort_output, trt_output, rtol=MODEL_MATCH_RTOL, atol=MODEL_MATCH_ATOL).sum()
            print(f"ONNX-TRT Output {index} matches: {matches / trt_output.size:.3%}")

            assert np.allclose(ort_output, trt_output, rtol=MODEL_MATCH_RTOL, atol=MODEL_MATCH_ATOL), f"Output mismatch {index}"

    return True

# Returns a path to an onnx model that matches the given model configuration
def create_and_validate_onnx(config_name: str) -> str:
    config = MODEL_CONFIGS[config_name]
    assert config is not None, "Unable to find config"
    assert config["type"] == "vision", "Config must be a vision model"

    Path(HOST_CONFIG.CACHE_DIR, "models").mkdir(parents=True, exist_ok=True)

    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = False

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False

    # TODO The first version will only do batch_size 1, but for later speed in recalculating the cache, we should increase the size
    batch_size = 1
    # slice down the image size to the nearest multiple of the stride
    img_size = (DEVICE_CONFIG.CAMERA_HEIGHT // config["dimension_stride"] * config["dimension_stride"],
                DEVICE_CONFIG.CAMERA_WIDTH // config["dimension_stride"] * config["dimension_stride"])
    device = "cuda:0"

    # Load the original pytorch model

    # Import and call function by string from the config
    load_module = importlib.import_module(".".join(config["load_fn"].split(".")[:-1]))
    load_fn = getattr(load_module, config["load_fn"].split(".")[-1])
    model = load_fn(config["checkpoint"])

    img = torch.zeros(batch_size, 3, *img_size).to(device)  # image size(1,3,height,width) 

    y = model(img)  # dry run

    # Convert that to ONNX
    onnx_path = os.path.join(HOST_CONFIG.CACHE_DIR, "models", f"{model_basename(config_name)}.onnx")
    onnx_exists_and_validates = False

    if os.path.exists(onnx_path):
        print(f"Found cached ONNX model {onnx_path}")
        onnx_exists_and_validates = validate_pt_onnx(model, onnx_path)
    
    if not onnx_exists_and_validates:
        torch.onnx.export(model, img, onnx_path, verbose=False, opset_version=12, dynamic_axes=None)

        onnx_model = onnx.load(onnx_path)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print("Confirmed ONNX model is valid")
        onnx_exists_and_validates = validate_pt_onnx(model, onnx_path)

    assert onnx_exists_and_validates, "Validation of pytorch and onnx outputs failed"

    return onnx_path

# Returns a path to a TRT model that matches the given model configuration
def create_and_validate_trt(config_name: str) -> str:
    config = MODEL_CONFIGS[config_name]
    assert config is not None, "Unable to find config"
    assert config["type"] == "vision", "Config must be a vision model"

    Path(HOST_CONFIG.CACHE_DIR, "models").mkdir(parents=True, exist_ok=True) 

    # TRT models must be made from ONNX files
    onnx_path = create_and_validate_onnx(config_name)


    # Build the tensorRT engine
    trt_path = os.path.join(HOST_CONFIG.CACHE_DIR, "models", f"{model_basename(config_name)}.engine")
    trt_exists_and_validates = False

    if os.path.exists(trt_path):
        print("Loading existing engine")
        trt_exists_and_validates = validate_onnx_trt(onnx_path, trt_path)

    if not trt_exists_and_validates:
        build_engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_path), config=CreateConfig(fp16=False)) 
        build_engine = SaveEngine(build_engine, path=trt_path)

        with TrtRunner(build_engine) as runner:
            print("Created TRT engine")

        trt_exists_and_validates = validate_onnx_trt(onnx_path, trt_path)

    assert trt_exists_and_validates, "Validation of onnx and trt outputs failed"


# Loads a preconfigured model from a pytorch checkpoint,
# if needed, it builds a new tensorRT engine, and verifies that the model results are identical
def load_vision_model(config: str) -> polygraphy.backend.trt.TrtRunner:
    trt_path = create_and_validate_trt(config)

