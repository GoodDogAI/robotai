import unittest
import tempfile
import os

from src.logutil import LogHashes


class LogUtilTest(unittest.TestCase):
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            logutil = LogHashes(td)

    def test_basic_log(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "test.log"), "wb") as f:
                f.write(b"HelloWorld")

            logutil = LogHashes(td)
            print(logutil.hashes)



if __name__ == '__main__':
    unittest.main()
