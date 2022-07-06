import unittest
import tempfile
import os

from src.logutil import LogHashes


class LogHashesTest(unittest.TestCase):
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            logutil = LogHashes(td)

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

            # Modifying a file changes the hash
            with open(os.path.join(td, "test.log"), "wb") as f:
                f.write(b"GoodbyeWorld")

            logutil.update()
            self.assertNotEqual(logutil.files, logutil2.files)


if __name__ == '__main__':
    unittest.main()
