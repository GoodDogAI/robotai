import requests
import os
from src.config import DEVICE_CONFIG

def prepare_device_model(config_name: str):
    resp = requests.get(f"/models/{config_name}/onnx/")
    assert resp.status_code == 200, f"Failed to reach service for model {config_name}"
    assert "Content-Disposition" in resp.headers, "Missing Content-Disposition header"
    
    onnx_filename = resp.headers.get("Content-Disposition").split("filename=")[1]
    with open(os.path.join(DEVICE_CONFIG.MODEL_STORAGE_PATH, onnx_filename), "wb") as f:
        f.write(resp.content)

    
