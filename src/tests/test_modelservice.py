import unittest
import tempfile
import hashlib
import io
import numpy as np

from fastapi.testclient import TestClient
from cereal import log
from src.logutil import sha256, LogHashes
from src.tests.utils import artificial_logfile
from src.web.main import app
from src.web.dependencies import get_loghashes


class ModelServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_model_onnx(self):
        resp = self.client.get("/models/brain/default/")
        self.assertEqual(resp.status_code, 200)

        j = resp.json()
        main_key = list(j.keys())[0]

        resp = self.client.get(f"/models/INVALID/onnx/")
        self.assertEqual(resp.status_code, 404)

        resp = self.client.get(f"/models/{j[main_key]['vision_model']}/onnx/")
        self.assertEqual(resp.status_code, 200)

    def test_model_references(self):
        resp = self.client.get("/models/brain/default/")

        j = resp.json()
        main_key = list(j.keys())[0]
        
        resp = self.client.get(f"/models/{j[main_key]['vision_model']}/reference_input/INVALID_NAME")
        self.assertEqual(resp.status_code, 404)
    
        resp = self.client.get(f"/models/{j[main_key]['vision_model']}/reference_input/input.1")
        self.assertEqual(resp.status_code, 200)

        file_data = io.BytesIO(resp.content)
        ref_input = np.load(file_data)

        print(ref_input.shape)

        resp = self.client.get(f"/models/{j[main_key]['vision_model']}/reference_output/INVALID_OUTPUT")
        self.assertEqual(resp.status_code, 404)

        resp = self.client.get(f"/models/{j[main_key]['vision_model']}/reference_output/onnx::Sigmoid_308")
        self.assertEqual(resp.status_code, 200)
    

if __name__ == '__main__':
    unittest.main()
