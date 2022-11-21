import os
import unittest
import torch

from src.messaging import new_message
from src.train.modelloader import model_fullname, create_pt_model
from src.train.arrowcache import ArrowModelCache
from src.train.rldataset import MsgVecDataset
from src.config import HOST_CONFIG, MODEL_CONFIGS

class StableBaselinesRLTests(unittest.TestCase):
    def test_pytorch_params(self):
        cache = MsgVecDataset(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), MODEL_CONFIGS["basic-brain-test1"])

        datapoint = next(cache.generate_dataset())
        obs = torch.from_numpy(datapoint["obs"]).to("cuda")
        obs = torch.unsqueeze(obs, 0)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        pt_model = create_pt_model(MODEL_CONFIGS["basic-brain-test1"])

        res1 = pt_model(observation=obs)
        print(res1)

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        pt_model = create_pt_model(MODEL_CONFIGS["basic-brain-test1"])

        res2 = pt_model(observation=obs)
        print(res2)

        self.assertTrue(torch.allclose(res1, res2))
