import torch
import onnx
import onnxruntime
import os
import numpy as np

from itertools import chain
import polygraphy
import polygraphy.backend.trt

from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, EngineFromBytes, SaveEngine, TrtRunner

from src.logutil import sha256
from src.train.config_train import VISION_CONFIGS, CACHE_DIR
from src.include.config import load_realtime_config

CONFIG = load_realtime_config()
DECODE_WIDTH = int(CONFIG["CAMERA_WIDTH"])
DECODE_HEIGHT = int(CONFIG["CAMERA_HEIGHT"])

MODEL_MATCH_RTOL = 1e-4
MODEL_MATCH_ATOL = 1e-4


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

    print("Validated pytorch and onnx outputs")

    return True   

def validate_onnx_trt(onnx_path: str, trt_path: str) -> bool:
    with open(trt_path, "rb") as f:
        build_engine = EngineFromBytes(f.read())

    ort_sess = onnxruntime.InferenceSession(onnx_path)
    assert len(ort_sess.get_inputs()) == 1, "ONNX model must have a single input"
    assert ort_sess.get_inputs()[0].type == "tensor(float)", "ONNX model must have a single input of type float"
    ort_shape = ort_sess.get_inputs()[0].shape

    random_input = torch.FloatTensor(*ort_shape).uniform_(0.0, 1.0)
    ort_outputs = ort_sess.run(None, {'input.1': random_input.cpu().numpy()})

    with TrtRunner(build_engine) as runner:
        assert len(runner.get_input_metadata()) == 1, "TRT model must have a single input"
        trt_input_name = next(iter(runner.get_input_metadata()))
        trt_input_shape = runner.get_input_metadata()[trt_input_name].shape
        assert ort_shape == trt_input_shape, "Input shape mismatch"
        assert runner.get_input_metadata()[trt_input_name].dtype == np.float32, "TRT model must have a single input of type float"

        trt_outputs = runner.infer({
            trt_input_name: random_input.cpu().numpy()
        })

        # Check that the outputs are the same
        for index, ort_output_metadata in enumerate(ort_sess.get_outputs()):
            ort_output = ort_outputs[index]
            trt_output = trt_outputs[ort_output_metadata.name]

            matches = np.isclose(ort_output, trt_output, rtol=MODEL_MATCH_RTOL, atol=MODEL_MATCH_ATOL).sum()
            print(f"ONNX-TRT Output {index} matches: {matches / trt_output.size:.3%}")

            assert np.allclose(ort_output, trt_output, rtol=MODEL_MATCH_RTOL, atol=MODEL_MATCH_ATOL), f"Output mismatch {index}"

        



# Loads a preconfigured model from a pytorch checkpoint,
# if needed, it builds a new tensorRT engine, and verifies that the model results are identical
def load_vision_model(config: str) -> polygraphy.backend.trt.TrtRunner:
    config = VISION_CONFIGS[config]
    assert config is not None, "Unable to find config"

    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = False

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False

    # TODO The first version will only do batch_size 1, but for later speed in recalculating the cache, we should increase the size
    batch_size = 1
    # slice down the image size to the nearest multiple of the stride
    img_size = (DECODE_HEIGHT // config["dimension_stride"] * config["dimension_stride"],
                DECODE_WIDTH // config["dimension_stride"] * config["dimension_stride"])
    device = "cuda:0"

    # Load the original pytorch model
    model_sha = sha256(config["checkpoint"])
    model_basename = os.path.basename(config["checkpoint"]).replace(".pt", "") + "-" + model_sha
    model = config["load_fn"](config["checkpoint"])

    img = torch.zeros(batch_size, 3, *img_size).to(device)  # image size(1,3,height,width) 

    y = model(img)  # dry run

    # Convert that to ONNX
    onnx_path = os.path.join(CACHE_DIR, f"{model_basename}.onnx")
    print("Starting ONNX export with onnx {onnx.__version__}")

    torch.onnx.export(model, img, onnx_path, verbose=False, opset_version=12, dynamic_axes=None)

    onnx_model = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print("Confirmed ONNX model is valid")

    assert validate_pt_onnx(model, onnx_path), "Validation of pytorch and onnx outputs failed"
    print("Validated pytorch and onnx outputs")

    # Build the tensorRT engine
    trt_path = os.path.join(CACHE_DIR, f"{model_basename}.engine")

    if os.path.exists(trt_path):
        print("Loading existing engine")

        validates = validate_onnx_trt(onnx_path, trt_path)
    # else:
    #     build_engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_path), config=CreateConfig(fp16=False)) 
    #     build_engine = SaveEngine(build_engine, path=trt_path)

    #     # TODO: To help speed things up, we should use a timing cache
    #     # TODO: We can also see if the engine already exists, and if we can successfully load and compare it, just return that

    #     with TrtRunner(build_engine) as runner:
    #         outputs = runner.infer(feed_dict={"input.1": random_input.cpu().numpy()})

    #     print("Create TRT engine")