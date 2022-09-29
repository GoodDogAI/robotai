import torch
import onnx
import onnxruntime
import os
import copy
import time
import importlib
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Literal, Union, List, Tuple, Iterable, BinaryIO, Dict, Any
from contextlib import contextmanager, ExitStack

import polygraphy
import polygraphy.backend.trt
import onnx_graphsurgeon

from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, EngineFromBytes, SaveEngine, TrtRunner
from polygraphy.cuda import DeviceView
from cereal import log

from src.train.onnx_yuv import NV12MToRGB, CenterCrop, ConvertCropVision, int8_from_uint8
from src.train.reward import ConvertCropVisionReward, ThresholdNMS, SumCenteredObjectsPresentReward
from src.config import DEVICE_CONFIG, HOST_CONFIG, MODEL_CONFIGS
from src.config.config import BRAIN_CONFIGS



MODEL_MATCH_RTOL = 1e-4
MODEL_MATCH_ATOL = 1e-4

def update_model_config_caches():
    for model_name, model_config in MODEL_CONFIGS.items():
        config_cache_path = os.path.join(HOST_CONFIG.CACHE_DIR, "models", f"{model_fullname(model_config)}_config.json")

        if not os.path.exists(config_cache_path):
            with open(config_cache_path, "w") as f:
                json.dump(model_config, f, indent=4)


def onnx_to_numpy_dtype(onnx_type: str) -> np.dtype:
    if onnx_type == "tensor(float)":
        return np.float32
    elif onnx_type == "tensor(int32)":
        return np.int32
    elif onnx_type == "tensor(int64)":
        return np.int64
    elif onnx_type == "tensor(uint8)":
        return np.uint8
    elif onnx_type == "tensor(int8)":
        return np.int8
    else:
        raise ValueError(f"Unsupported ONNX type {onnx_type}")

def get_reference_input(shape: List[int], dtype: np.dtype, model_type: Literal["vision"]) -> np.ndarray:
    if model_type == "vision" or model_type == "reward":
        if dtype == np.uint8:
            return np.random.randint(low=16, high=235, size=shape, dtype=np.uint8)
        elif dtype == np.float32:
            return np.random.randint(low=16, high=235, size=shape, dtype=np.uint8).astype(np.float32)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

def get_pt_feeddict(ort_sess: onnxruntime.InferenceSession, model_type: Literal["vision"]) -> dict:
    pt_feed_dict = {}
    for ort_input in ort_sess.get_inputs():
        ort_shape = ort_input.shape
        ort_dtype = onnx_to_numpy_dtype(ort_input.type)

        pt_feed_dict[ort_input.name] = torch.from_numpy(get_reference_input(ort_shape, ort_dtype, model_type)).to(device="cuda")

    return pt_feed_dict


def _flatten_deep(x: Union[List, Tuple]) -> Iterable:
    if isinstance(x, (list, tuple)):
        for y in x:
            yield from _flatten_deep(y)
    else:
        yield x

# Returns a unique path that includes a hash of the model config and checkpoint
def model_fullname(config: Dict[str, Any]) -> str:
    assert config is not None, "Unable to find config"
    sha256_hash = hashlib.sha256()

    with open(config["checkpoint"], "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)

    # Also include the hash of the config itself
    sha256_hash.update(repr(config).encode("utf-8"))
    model_sha = sha256_hash.hexdigest()[:16]

    model_fullname = os.path.basename(config["checkpoint"]).replace(".pt", "") + "-" + model_sha + ""
    return model_fullname

def brain_fullname(brain_name: str) -> str:
    brain = BRAIN_CONFIGS[brain_name]
    assert brain is not None, "Unable to find brain config"

    sha256_hash = hashlib.sha256()
    sha256_hash.update(repr(brain).encode("utf-8"))
    config_sha = sha256_hash.hexdigest()[:16]

    brain_fullname = brain_name + "-" + config_sha
    return brain_fullname


def validate_pt_onnx(pt_model: torch.nn.Module, onnx_path: str, model_type:Literal["vision"]) -> bool:
    ort_sess = onnxruntime.InferenceSession(onnx_path)
  
    # NB. You need to make a copy of the feed_dict now, otherwise the tensors will be modified in-place
    pt_feed_dict = get_pt_feeddict(ort_sess, model_type)
    ort_feed_dict = {k: v.cpu().numpy() for k, v in pt_feed_dict.items()}

    torch_outputs = pt_model(**pt_feed_dict)
    output_names = [output.name for output in ort_sess.get_outputs()]
    ort_outputs = ort_sess.run(output_names, ort_feed_dict)
    

    # Check that the outputs are the same
    for i, torch_output in enumerate(_flatten_deep(torch_outputs)):
        torch_output = torch_output.detach().cpu().numpy()
        ort_output = ort_outputs[i]

        if model_type == "reward" and output_names[i] != "raw_detections":
            print("SKIPPING REWARD OUTPUT")
            continue

        matches = np.isclose(ort_output, torch_output, rtol=MODEL_MATCH_RTOL, atol=MODEL_MATCH_ATOL).sum()
        print(f"PT-ONNX Output {i} matches: {matches / torch_output.size:.3%}")

        diffs = np.abs(ort_output - torch_output)

        assert np.allclose(ort_output, torch_output, rtol=MODEL_MATCH_RTOL, atol=MODEL_MATCH_ATOL), f"Output mismatch {i}"

    return True   

def validate_onnx_trt(onnx_path: str, trt_path: str, model_type:Literal["vision"]) -> bool:
    with open(trt_path, "rb") as f:
        build_engine = EngineFromBytes(f.read())

    ort_sess = onnxruntime.InferenceSession(onnx_path)

    pt_feed_dict = get_pt_feeddict(ort_sess, model_type)
    ort_feed_dict = {k: v.cpu().numpy() for k, v in pt_feed_dict.items()}

    ort_outputs = ort_sess.run(None, ort_feed_dict)

    with TrtRunner(build_engine) as runner:
        start = time.perf_counter()

        for i in range(100):
            trt_outputs = runner.infer({name: DeviceView(data.data_ptr(), data.shape, np.float32) for name, data in pt_feed_dict.items()})

        print(f"TRT inference time: {(time.perf_counter() - start)/100:.3f}s")

        # Check that the outputs are the same
        for index, ort_output_metadata in enumerate(ort_sess.get_outputs()):
            ort_output = ort_outputs[index]
            trt_output = trt_outputs[ort_output_metadata.name]

            matches = np.isclose(ort_output, trt_output, rtol=MODEL_MATCH_RTOL, atol=MODEL_MATCH_ATOL).sum()
            print(f"ONNX-TRT Output {index} matches: {matches / trt_output.size:.3%}")

            if model_type == "reward" and ort_output_metadata.name != "raw_detections":
                # We check to see that at least the raw_detections match exactly on reward-type models, the NMS algorithms are slighly mismatched between PT and ONNX
                pass
            else:
                assert np.allclose(ort_output, trt_output, rtol=MODEL_MATCH_RTOL, atol=MODEL_MATCH_ATOL), f"Output mismatch {index}"

    return True

def create_pt_model(config: Dict[str, Any]) -> torch.nn.Module:
    model_type = config["type"]
    assert model_type in {"vision", "reward"}, "Config must be a vision model"

    # Load the original pytorch model
    # Import and call function by string from the config
    load_module = importlib.import_module(".".join(config["load_fn"].split(".")[:-1]))
    load_fn = getattr(load_module, config["load_fn"].split(".")[-1])
    vision_model = load_fn(config["checkpoint"])

    internal_size = (DEVICE_CONFIG.CAMERA_HEIGHT // config["dimension_stride"] * config["dimension_stride"],
                     DEVICE_CONFIG.CAMERA_WIDTH // config["dimension_stride"] * config["dimension_stride"])

    # Make a module that is going to include the input format conversion and any required cropping
    if model_type == "vision":
        full_model = ConvertCropVision(NV12MToRGB(), CenterCrop(internal_size), vision_model)
    elif model_type == "reward":
        assert config["reward_module"] == "src.train.reward.SumCenteredObjectsPresentReward", "Other modules not implemented yet"
        full_model = ConvertCropVisionReward(NV12MToRGB(), CenterCrop(internal_size),
                                             vision_model,
                                             ThresholdNMS(iou_threshold=config["iou_threshold"], max_detections=config["max_detections"]),
                                             SumCenteredObjectsPresentReward(width=internal_size[1], height=internal_size[0], 
                                                                             class_names=config["class_names"],
                                                                             class_weights=config["reward_kwargs"]["class_weights"],
                                                                             reward_scale=config["reward_kwargs"]["reward_scale"],
                                                                             center_epsilon=config["reward_kwargs"]["center_epsilon"]))
    else:
        raise NotImplementedError()

    return full_model

# Returns a path to an onnx model that matches the given model configuration
def create_and_validate_onnx(config: Dict[str, Any], skip_cache: bool=False) -> str:
    assert config is not None, "Unable to find config"
    model_type = config["type"]
    assert model_type in {"vision", "reward"}, "Config must be a vision model"

    assert config["input_format"] == "rgb", "Only NV12M to RGB conversion is supported"

    Path(HOST_CONFIG.CACHE_DIR, "models").mkdir(parents=True, exist_ok=True)

    # Save off the config that will be used to generate this model, so that we can regerenate TRT models if the config ever changes in the future
    config_cache_path = os.path.join(HOST_CONFIG.CACHE_DIR, "models", f"{model_fullname(config)}_config.json")

    with open(config_cache_path, "w") as f:
        json.dump(config, f, indent=4)

    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = False

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False

    # The first version will only do batch_size 1, but for later speed in recalculating the cache, we should increase the size
    # However, I did some testing, and increasing to an optimal batch size will increase throughput maybe 2x, but at vast increase to code complexity
    batch_size = 1
    device = "cuda:0"
    # slice down the image size to the nearest multiple of the stride
    inputs = {
        "y": torch.from_numpy(get_reference_input((batch_size, 1, DEVICE_CONFIG.CAMERA_HEIGHT, DEVICE_CONFIG.CAMERA_WIDTH), np.float32, model_type)).to(device=device),
        "uv": torch.from_numpy(get_reference_input((batch_size, 1, DEVICE_CONFIG.CAMERA_HEIGHT // 2, DEVICE_CONFIG.CAMERA_WIDTH), np.float32, model_type)).to(device=device),
    }

    full_model = create_pt_model(config)

    _ = full_model(**inputs)  # dry run

    # Convert and validate the full base model to ONNX
    orig_onnx_path = os.path.join(HOST_CONFIG.CACHE_DIR, "models", f"{model_fullname(config)}_orig.onnx")
    final_onnx_path = os.path.join(HOST_CONFIG.CACHE_DIR, "models", f"{model_fullname(config)}_final.onnx")

    onnx_exists_and_validates = False

    if os.path.exists(orig_onnx_path) and not skip_cache:
        print(f"Found cached ONNX model {orig_onnx_path}")
        onnx_exists_and_validates = validate_pt_onnx(full_model, orig_onnx_path, model_type)
    
    if not onnx_exists_and_validates:
        output_names = None

        if model_type == "reward":
            output_names = ["reward", "bboxes", "raw_detections"]

        torch.onnx.export(copy.deepcopy(full_model), inputs, orig_onnx_path, input_names=list(inputs), output_names=output_names,
                          verbose=False, opset_version=12, dynamic_axes=None)

        onnx_model = onnx.load(orig_onnx_path)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print("Confirmed ONNX model is valid")

        # Use graph surgeon to simplify the model, but make no other changes for now
        graph = onnx_graphsurgeon.import_onnx(onnx_model)
        graph.toposort()
        graph.fold_constants()

        for op in graph.nodes:
            # The NonMaxSupression op is supported in both ONNX and TRT, but we need to manually configure the IOU threshold
            # and set it to a fixed max output size
            if op.op == "NonMaxSuppression":
                op.inputs[2].values = np.array([config["max_detections"]], dtype=np.int64) # Number of detections per class
                op.inputs[3].values = np.array([config["iou_threshold"]], dtype=np.float32) # IOU threshold

        graph.cleanup()

        onnx.save(onnx_graphsurgeon.export_onnx(graph), orig_onnx_path)

        onnx_exists_and_validates = validate_pt_onnx(full_model, orig_onnx_path, model_type)

    if not onnx_exists_and_validates:
        os.remove(orig_onnx_path)
        raise AssertionError("Validation of pytorch and onnx outputs failed")

    # Now, load that onnx, and cut it down to only the desired output
    onnx_model = onnx.load(orig_onnx_path)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    graph = onnx_graphsurgeon.import_onnx(onnx_model)
    tensors = graph.tensors()

    if model_type == "vision":
        # Extract the intermediate output
        intermediate_output = tensors[config["intermediate_layer"]]

        # Now, do a concatenation and slicing
        
        # inputs are [tensor, new_shape]
        reshaped = graph.layer(inputs=[intermediate_output, np.array([0, -1])], outputs=["intermediate_reshape"], op="Reshape") 

        # inputs are [tensor, starts, ends, axes, steps]
        final_output = onnx_graphsurgeon.Variable("intermediate", dtype=np.float32)
        graph.layer(inputs=reshaped + [np.array([0]), np.array([-1]), np.array([1]), np.array([config["intermediate_slice"]])], outputs=[final_output], op="Slice")

        graph.outputs = [final_output]
        graph.cleanup()
    elif model_type == "reward":
        pass

    # Save the final onnx model, but do a shape inference one last time on it for convience in debugging
    onnx.save(onnx_graphsurgeon.export_onnx(graph), final_onnx_path)
    onnx_model = onnx.load(final_onnx_path)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, final_onnx_path)

    return final_onnx_path

# Returns a path to a TRT model that matches the given model configuration
def create_and_validate_trt(onnx_path: str, skip_cache: bool=False) -> str:
    json_config_path = onnx_path.replace("_final.onnx", "_config.json")
    with open(json_config_path, "r") as f:
        config = json.load(f)
    assert config is not None, "Unable to find cached config"
    model_type = config["type"]
    assert model_type in {"vision", "reward"}, "Config must be a vision or reward model"

    Path(HOST_CONFIG.CACHE_DIR, "models").mkdir(parents=True, exist_ok=True) 

    # Build the tensorRT engine
    trt_path = onnx_path.replace("_final.onnx", ".engine")
    trt_exists_and_validates = False

    if os.path.exists(trt_path) and not skip_cache:
        print("Loading existing engine")
        trt_exists_and_validates = validate_onnx_trt(onnx_path, trt_path, model_type)

    if not trt_exists_and_validates:
        build_engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_path), config=CreateConfig(fp16=False, max_workspace_size=1e8)) 
        build_engine = SaveEngine(build_engine, path=trt_path)

        with TrtRunner(build_engine) as runner:
            print("Created TRT engine")

        trt_exists_and_validates = validate_onnx_trt(onnx_path, trt_path, model_type)

    if not trt_exists_and_validates:
        os.remove(trt_path)
        raise AssertionError("Validation of onnx and trt outputs failed")

    return trt_path

# Loads a pre-cached, pre generated vision / reward model
@contextmanager
def load_vision_model(full_name: str) -> polygraphy.backend.trt.TrtRunner:
    trt_path = os.path.join(HOST_CONFIG.CACHE_DIR, "models", f"{full_name}.engine")

    if os.path.exists(trt_path):
        print(f"Loading cached engine {trt_path}")
    else:
        print(f"Engine {trt_path} not found, creating it from ONNX")
        
        config_path = os.path.join(HOST_CONFIG.CACHE_DIR, "models", f"{full_name}_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        onnx_final_path = create_and_validate_onnx(config)
        trt_path = create_and_validate_trt(onnx_final_path)

    with open(trt_path, "rb") as f:
        build_engine = EngineFromBytes(f.read())

    runner = TrtRunner(build_engine)

    try:
        runner.activate()

        yield runner
    finally:
        runner.deactivate()

@contextmanager
def load_all_models_in_log(input: BinaryIO) -> Dict[str, polygraphy.backend.trt.TrtRunner]:
    input.seek(0)

    models = {}

    events = log.Event.read_multiple(input)   

    with ExitStack() as es:
        for evt in events:
            if evt.which() == "modelValidation" and evt.modelValidation.modelType == log.ModelValidation.ModelType.visionIntermediate:
                if evt.modelValidation.modelFullName not in models:
                    model_context = load_vision_model(evt.modelValidation.modelFullName)
                    models[evt.modelValidation.modelFullName] = es.enter_context(model_context)

        yield models
