
import os
import tensorrt as trt
import numpy as np
import unittest
import png

from src.train.onnx_yuv import png_to_nv12m
from src.config import HOST_CONFIG, BRAIN_CONFIGS
from src.train.modelloader import load_vision_model
from src.train.log_validation import full_validate_log, cosine_similarity

class TestModelValidation(unittest.TestCase):
    def test_typical_model_deltas(self):
        # Load an image, and run it through the model
        with open(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "room.png"), "rb") as f1:
            y1, uv1 = png_to_nv12m(f1)

        with open(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "plug.png"), "rb") as f1:
            y2, uv2 = png_to_nv12m(f1)


        with load_vision_model("yolov7-tiny-s53") as engine:
            trt_outputs = engine.infer({"y": np.copy(y1), "uv": np.copy(uv1)})
            trt_intermediate_1 = np.copy(trt_outputs["intermediate"])

            # Now, perturb the y input slightly
            offsets = np.round(np.random.uniform(-2, 2, y1.shape)).astype(np.int8)
            y1 += offsets
            y1 = np.clip(y1, 16, 235)

            trt_outputs = engine.infer({"y": np.copy(y1), "uv": np.copy(uv1)})
            trt_intermediate_1perturb = np.copy(trt_outputs["intermediate"])

            trt_outputs = engine.infer({"y": np.copy(y2), "uv": np.copy(uv2)})
            trt_intermediate_2 = np.copy(trt_outputs["intermediate"])

            # Now, compute the difference between the two
            matches = np.isclose(trt_intermediate_1, trt_intermediate_1perturb, rtol=1e-2, atol=1e-2).sum()
            print(f"Logged Output matches: {matches / trt_intermediate_1perturb.size:.3%}")

            # Cosine similarity to a random vector should be around zero
            self.assertLess(cosine_similarity(trt_intermediate_1[0], np.random.uniform(-1, 1, trt_intermediate_1perturb[0].shape)), 0.01)

            # Cosine similarity to a perturbed image should be quite high
            print(f"Cosine Similarity: {cosine_similarity(trt_intermediate_1[0], trt_intermediate_1perturb[0]):.3%}")
            self.assertGreater(cosine_similarity(trt_intermediate_1[0], trt_intermediate_1perturb[0]), 0.98)
            
            # Cosine similarity to a different image should be quite low
            print(f"Cosine Similarity: {cosine_similarity(trt_intermediate_1[0], trt_intermediate_2[0]):.3%}")
            self.assertLess(cosine_similarity(trt_intermediate_1[0], trt_intermediate_2[0]), 0.60)


    def test_vision_intermediate_video(self):
        #test_path = os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "alphalog-5205d621-2022-9-14-21_57.log")
        test_path = os.path.join(HOST_CONFIG.RECORD_DIR, "alphalog-6eb8d100-2022-9-16-16_21.log")
        #test_path = os.path.join(HOST_CONFIG.RECORD_DIR, "alphalog-6705c9d2-2022-9-16-16_47.log")
        
        with open(test_path, "rb") as f:
            full_validate_log(f)
