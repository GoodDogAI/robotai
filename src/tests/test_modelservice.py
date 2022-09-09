import unittest
import onnxruntime
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
        main_key = list(j)[0]

        resp = self.client.get(f"/models/INVALID/onnx/")
        self.assertEqual(resp.status_code, 404)

        vision_model = j[main_key]["models"]["vision_model"]["basename"]

        resp = self.client.get(f"/models/{vision_model}/onnx/")
        self.assertEqual(resp.status_code, 200)

    def test_model_references(self):
        resp = self.client.get("/models/brain/default/")

        j = resp.json()
        main_key = list(j)[0]
        vision_model = j[main_key]["models"]["vision_model"]["basename"]

        resp = self.client.get(f"/models/{vision_model}/onnx/")
        self.assertEqual(resp.status_code, 200)

        ort_sess = onnxruntime.InferenceSession(resp.content)

        inputs = [i.name for i in ort_sess.get_inputs()]
        output_name = "intermediate"
        feed_dict = {}
        
        resp = self.client.get(f"/models/{vision_model}/reference_input/INVALID_NAME")
        self.assertEqual(resp.status_code, 404)
    
        for input in inputs:
            resp = self.client.get(f"/models/{vision_model}/reference_input/{input}")
            self.assertEqual(resp.status_code, 200)

            file_data = io.BytesIO(resp.content)
            ref_input = np.load(file_data)

            feed_dict[input] = ref_input
            print(ref_input.shape)

        resp = self.client.get(f"/models/{vision_model}/reference_output/INVALID_OUTPUT")
        self.assertEqual(resp.status_code, 404)

        resp = self.client.get(f"/models/{vision_model}/reference_output/{output_name}")
        self.assertEqual(resp.status_code, 200)

        file_data = io.BytesIO(resp.content)
        ref_output = np.load(file_data)

        print(ref_output.shape)

        ort_outputs = ort_sess.run([output_name], feed_dict)

        np.testing.assert_almost_equal(ort_outputs[0], ref_output, decimal=5)

    

if __name__ == '__main__':
    unittest.main()
