import time
import unittest
import tempfile
import os

from src.logutil import LogHashes, validate_log
from src.tests.utils import artificial_logfile


class LogHashesTest(unittest.TestCase):
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            logutil = LogHashes(td)
            self.assertEqual(logutil.files, {})

    def test_basic_log(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "test.log"), "wb") as f:
                f.write(b"HelloWorld")

            logutil = LogHashes(td)
            logutil2 = LogHashes(td)
            self.assertEqual(logutil.files, logutil2.files)

            # File of different extension is not hashed
            with open(os.path.join(td, "test.random"), "wb") as f:
                f.write(b"Test")

            logutil.update()
            self.assertEqual(logutil.files, logutil2.files)

            # Annoying hack, because it uses modification time as a speedup
            time.sleep(0.01)

            # Modifying a file changes the hash
            with open(os.path.join(td, "test.log"), "wb") as f:
                f.write(b"GoodbyeWorld")

            logutil.update()
            self.assertNotEqual(logutil.files, logutil2.files)

    def test_validate_log(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "test.log"), "wb") as f:
                f.write(b"HelloWorld")

            with open(os.path.join(td, "test.log"), "rb") as f:
                self.assertFalse(validate_log(f))

    def test_artificial_logfile(self):
        with artificial_logfile() as f:
            self.assertTrue(validate_log(f))

if __name__ == '__main__':
    unittest.main()
