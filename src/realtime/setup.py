import requests
import onnx
import json
import logging
import os
from src.config import DEVICE_CONFIG

logger = logging.getLogger(__name__)

MODEL_SERVICE = DEVICE_CONFIG["MODEL_SERVICE"]


def prepare_device_model(config_name: str):
    logger.warning(f"Preparing model {config_name}")
    resp = requests.get(f"{MODEL_SERVICE}/models/{config_name}/onnx/")
    assert resp.status_code == 200, f"Failed to reach service for model {config_name}"
    assert "Content-Disposition" in resp.headers, "Missing Content-Disposition header"
    
    onnx_filename = resp.headers.get("Content-Disposition").split("filename=")[1]
    with open(os.path.join(DEVICE_CONFIG.MODEL_STORAGE_PATH, onnx_filename), "wb") as f:
        f.write(resp.content)

    

def prepare_brain(brain_name: str=None):
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

    prepare_device_model(brain_config["vision_model"])

    # TODO Uncomment when the model service is ready
    #prepare_device_model(brain_config["brain_model"])

