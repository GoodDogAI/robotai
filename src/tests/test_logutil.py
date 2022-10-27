import time
import unittest
import tempfile
import os

from src.logutil import LogHashes, quick_validate_log
from src.tests.utils import artificial_logfile


class LogHashesTest(unittest.TestCase):
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            logutil = LogHashes(td)
            self.assertEqual(logutil.values(), [])

    def test_basic_log(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "test.log"), "wb") as f:
                f.write(b"HelloWorld")

            logutil = LogHashes(td)
            logutil2 = LogHashes(td)
            self.assertEqual([x.filename for x in logutil.values()], [x.filename for x in logutil2.values()])
            
            # File of different extension is not hashed
            with open(os.path.join(td, "test.random"), "wb") as f:
                f.write(b"Test")

            logutil.update()
            self.assertEqual([x.filename for x in logutil.values()], [x.filename for x in logutil2.values()])

            # Annoying hack, because it uses modification time as a speedup
            time.sleep(0.01)

            # Modifying a file changes the hash, but keeps original hash
            with open(os.path.join(td, "test.log"), "wb") as f:
                f.write(b"GoodbyeWorld")

            logutil.update()
            self.assertEqual([(x.filename, x.sha256, x.orig_sha256) for x in logutil.values()],
                            [(x.filename, x.sha256, x.orig_sha256) for x in logutil2.values()])
            

            # Delete a file and it will stay in the database
            os.unlink(os.path.join(td, "test.log"))
            logutil.update()
            self.assertEqual(len(logutil.values()), 1)

    def test_validate_log(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "test.log"), "wb") as f:
                f.write(b"HelloWorld")

            with open(os.path.join(td, "test.log"), "rb") as f:
                self.assertFalse(quick_validate_log(f))

    def test_artificial_logfile(self):
        with artificial_logfile() as f:
            self.assertTrue(quick_validate_log(f))

    def test_logs_sorted_correctly(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "alphalog-6d7f2832-2022-10-11-18_10.log"), "wb") as f:
                f.write(b"HelloWorld")

            with open(os.path.join(td, "alphalog-6d7f2832-2022-10-11-18_9.log"), "wb") as f:
                f.write(b"HelloWorld")

            lh = LogHashes(td)
            self.assertEqual(lh.group_logs()[0][0].filename, "alphalog-6d7f2832-2022-10-11-18_9.log")
            self.assertEqual(lh.group_logs()[0][1].filename, "alphalog-6d7f2832-2022-10-11-18_10.log")

if __name__ == '__main__':
    unittest.main()
