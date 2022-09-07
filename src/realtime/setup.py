import requests
import onnx
import json
import logging
import os
import numpy as np

from typing import Literal, Dict
from src.config import DEVICE_CONFIG

import polygraphy
import polygraphy.backend.trt

from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, EngineFromBytes, SaveEngine, TrtRunner
from polygraphy.cuda import DeviceView


logger = logging.getLogger(__name__)

MODEL_SERVICE = DEVICE_CONFIG["MODEL_SERVICE"]


def prepare_model_reference(base_name: str, full_name: str, io: str, type: Literal["input", "output"]):
    logger.warning(f"Preparing model reference {type} {io}")
    reference_path = os.path.join(DEVICE_CONFIG.MODEL_STORAGE_PATH, f"{full_name}-{type}-{io}.npy")

    if not os.path.exists(reference_path):
        if type == "input":
            resp = requests.get(f"{MODEL_SERVICE}/models/{base_name}/reference_input/{io}")
        elif type == "output":
            resp = requests.get(f"{MODEL_SERVICE}/models/{base_name}/reference_output/{io}")
        else:
            raise NotImplementedError(f"IO type {type} not implemented")

        assert resp.status_code == 200, f"Failed to reach service for model {base_name}"

        with open(reference_path, "wb") as f:
            f.write(resp.content)

def prepare_device_model(base_name: str, full_name: str):
    logger.warning(f"Preparing model {base_name}")
    onnx_path = os.path.join(DEVICE_CONFIG.MODEL_STORAGE_PATH, f"{full_name}.onnx")
    trt_path = os.path.join(DEVICE_CONFIG.MODEL_STORAGE_PATH, f"{full_name}.engine")

    if not os.path.exists(onnx_path):
        resp = requests.get(f"{MODEL_SERVICE}/models/{base_name}/onnx/")
        assert resp.status_code == 200, f"Failed to reach service for model {base_name}"

        with open(onnx_path, "wb") as f:
            f.write(resp.content)

    model = onnx.load(onnx_path)

    for input in list(model.graph.input):
        prepare_model_reference(base_name, full_name, input.name, "input")

    for output in list(model.graph.output):
        prepare_model_reference(base_name, full_name, output.name, "output")

    if not os.path.exists(trt_path):
        build_engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_path), config=CreateConfig(fp16=False)) 
        build_engine = SaveEngine(build_engine, path=trt_path)

        with TrtRunner(build_engine) as runner:
            logger.info(f"Created TRT engine {full_name}")

    with open(trt_path, "rb") as f:
        run_engine = EngineFromBytes(f.read())

    with TrtRunner(run_engine) as runner:
        feed_dict = {}

        for input in list(model.graph.input):
            feed_dict[input.name] = np.load(os.path.join(DEVICE_CONFIG.MODEL_STORAGE_PATH, f"{full_name}-input-{input.name}.npy"))

        trt_outputs = runner.infer(feed_dict)

        for output_name, output_tensor in trt_outputs.items():
            reference = np.load(os.path.join(DEVICE_CONFIG.MODEL_STORAGE_PATH, f"{full_name}-output-{output_name}.npy"))
            assert np.allclose(output_tensor, reference, rtol=1e-4, atol=1e-4), f"Output {output_name} does not match reference"

    

def prepare_brain_models(brain_name: str=None) -> Dict[str, str]:
    # If brain_config is None, then get the current brain config name & checksum from the server
    # If you can't reach the server, then reuse the last one
    # NB: The config cached from the server will contain a checksum, so you don't have to update it everytime
    #     and also contain checksum of the required submodels

    # Ask the server for all the models in the current brain config
    # For each model, download the onnx, download the reference inputs/outputs, and convert to TensorRT

    # Once everything check out, return and allow the manager to start everything
    if brain_name is None:
        logger.warning("No brain name provided, attempting to locate and use default")
        resp = requests.get(f"{MODEL_SERVICE}/models/brain/default/")

        if resp.status_code == 200:
            with open(os.path.join(DEVICE_CONFIG.MODEL_STORAGE_PATH, "brain_config.json"), "wb") as f:
                f.write(resp.content)

            brain_name = list(resp.json())[0]
            logger.warning(f"Downloaded {brain_name} from server")
        else:
            logger.warning("Could not reach server to get brain config, using cached one")


    with open(os.path.join(DEVICE_CONFIG.MODEL_STORAGE_PATH, "brain_config.json"), "r") as f:
        brain_config = json.load(f)

    if brain_name != list(brain_config)[0]:
        raise RuntimeError(f"Brain name {brain_name} does not match the one in the config {list(brain_config)[0]}")

    brain_config = brain_config[brain_name]

    result = {}

    for model_name in brain_config["models"]:
        prepare_device_model(brain_config["models"][model_name]["basename"],
                             brain_config["models"][model_name]["_fullname"])
        result[model_name] = brain_config["models"][model_name]["_fullname"]                             

    return result
