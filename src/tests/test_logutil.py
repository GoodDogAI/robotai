import time
import unittest
import tempfile
import os

from src.logutil import LogHashes, check_log_monotonic, quick_validate_log, resort_log_monotonic
from src.messaging import new_message
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
            

            # Delete a file and it will get removed
            os.unlink(os.path.join(td, "test.log"))
            logutil.update()
            self.assertEqual(len(logutil.values()), 0)

    def test_validate_log(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "test.log"), "wb") as f:
                f.write(b"HelloWorld")

            with open(os.path.join(td, "test.log"), "rb") as f:
                self.assertFalse(quick_validate_log(f))

    def test_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "test.log"), "wb") as f:
                f.write(b"HelloWorld")

            lh = LogHashes(td)
            self.assertEqual(lh.values()[0].meta, {})

            lh.update_metadata("test.log", test="test")
            self.assertEqual(lh.values()[0].meta, {"test": "test"})

            lh.update()
            self.assertEqual(lh.values()[0].meta, {"test": "test"})

            with open(os.path.join(td, "test.log"), "wb") as f:
                f.write(b"GoodbyeWorld")

            lh.update()
            self.assertEqual(lh.values()[0].meta, {"test": "test"})

            lh.update_metadata("test.log", test="test2")
            self.assertEqual(lh.values()[0].meta, {"test": "test2"})

            lh.update_metadata("test.log", validation="passed")
            self.assertEqual(lh.values()[0].meta, {"test": "test2", "validation": "passed"})


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

    def test_logo_monotonic(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "aligned.log"), "w+b") as f:
                msg = new_message("voltage")
                msg.voltage.volts = 12.0
                msg.logMonoTime = 100
                msg.write(f)

                msg = new_message("voltage")
                msg.voltage.volts = 12.0
                msg.logMonoTime = 101
                msg.write(f)

                f.flush()
                f.seek(0)

                self.assertTrue(check_log_monotonic(f))

            with open(os.path.join(td, "notaligned.log"), "w+b") as f:
                msg = new_message("voltage")
                msg.voltage.volts = 12.0
                msg.logMonoTime = 100
                msg.write(f)

                msg = new_message("voltage")
                msg.voltage.volts = 12.0
                msg.logMonoTime = 99
                msg.write(f)

                f.flush()
                f.seek(0)

                self.assertFalse(check_log_monotonic(f))

            with open(os.path.join(td, "alginedequal.log"), "w+b") as f:
                msg = new_message("voltage")
                msg.voltage.volts = 12.0
                msg.logMonoTime = 100
                msg.write(f)

                msg = new_message("voltage")
                msg.voltage.volts = 12.0
                msg.logMonoTime = 100
                msg.write(f)

                msg = new_message("voltage")
                msg.voltage.volts = 12.0
                msg.logMonoTime = 102
                msg.write(f)

                f.flush()
                f.seek(0)

                self.assertTrue(check_log_monotonic(f))

            with open(os.path.join(td, "notaligned.log"), "rb") as i, open(os.path.join(td, "notaligned_fixed.log"), "w+b") as o:
                resort_log_monotonic(i, o)
                
                o.flush()
                o.seek(0)
                self.assertTrue(check_log_monotonic(o))
            
if __name__ == '__main__':
    unittest.main()
