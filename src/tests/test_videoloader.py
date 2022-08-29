import unittest
import time
import torch
from torch.utils.data import DataLoader

from src.train.videoloader import surface_to_tensor, build_datapipe

class VideoLoaderTest(unittest.TestCase):
    max_elements = 1000

    def test_basic(self):
        start = time.perf_counter()
        count = 0

        for x in build_datapipe().header(self.max_elements):
            count += 1

        print(f"Took {time.perf_counter() - start:0.2f} seconds")
        print(f"Loaded {count} frames")
        print(f"{count / (time.perf_counter() - start):0.1f} fps")
  
    def test_train(self):
        datapipe = build_datapipe().header(self.max_elements)
        dl = DataLoader(dataset=datapipe)



if __name__ == '__main__':
    unittest.main()
