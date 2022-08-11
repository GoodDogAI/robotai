import unittest
import time

from src.train.videoloader import surface_to_tensor, build_datapipe

class VideoLoaderTest(unittest.TestCase):
    def test_basic(self):
        start = time.perf_counter()
        count = 0

        for x in build_datapipe():
            count += 1

        print(f"Took {time.perf_counter() - start:0.2f} seconds")
        print(f"Loaded {count} frames")
        print(f"{count / (time.perf_counter() - start):0.1f} fps")
  

if __name__ == '__main__':
    unittest.main()
