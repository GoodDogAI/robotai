import unittest
import tempfile
import hashlib
import os

from fastapi.testclient import TestClient
from cereal import log
from src.logutil import sha256, LogHashes
from src.tests.utils import artificial_logfile
from src.web.main import app
from src.web.dependencies import get_loghashes
from src.config import HOST_CONFIG

class LogServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.lh = LogHashes(self.td.name)
        app.dependency_overrides[get_loghashes] = lambda: self.lh
        self.addCleanup(lambda: self.td.cleanup())

        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides = {}

    def test_empty_logs(self):
        resp = self.client.get("/logs/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])

    def test_post_empty_log(self):
        with tempfile.NamedTemporaryFile() as tf:
            resp = self.client.post("/logs/", files={"logfile": tf, "sha256": (None, "invalid")})
            self.assertEqual(resp.status_code, 400)

    def test_post_invalid_log(self):
        with tempfile.NamedTemporaryFile() as tf:
            tf.write(b"Invalid Log!")
            tf.flush()
            tf.seek(0)
            resp = self.client.post("/logs/", files={"logfile": tf, "sha256": (None, sha256(tf.name))})
            self.assertEqual(resp.status_code, 400)

    def test_post_artificial_log(self):
        with artificial_logfile() as tf:
            resp = self.client.post("/logs/", files={"logfile": tf, "sha256": (None, tf.sha256)})
            self.assertEqual(resp.status_code, 200)

            resp = self.client.get("/logs/")
            print(resp.json())

            tf.seek(0)
            sha256_hash = hashlib.sha256()
            b = tf.read()
            sha256_hash.update(b)
            self.assertEqual(resp.json()[0][0]["sha256"], sha256_hash.hexdigest())

            # Posting the same log again should error out
            tf.seek(0)
            resp = self.client.post("/logs/", files={"logfile": tf, "sha256": (None, tf.sha256)})
            self.assertNotEqual(resp.status_code, 200)

            # Change the locally stored version, to simulate performing model validation for example
            with open(os.path.join(self.td.name, os.path.basename(tf.name)), "wb") as f:
                f.write(b"Modified Log!")

            self.lh.update()
            self.assertNotEqual(self.lh.files[os.path.basename(tf.name)].sha256, self.lh.files[os.path.basename(tf.name)].orig_sha256)

            # Posting the same log again should still error out
            tf.seek(0)
            resp = self.client.post("/logs/", files={"logfile": tf, "sha256": (None, tf.sha256)})
            self.assertEqual(resp.status_code, 500)


    def test_post_validated_log(self):
        test_path = os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "alphalog-41a516ae-2022-9-19-2_20.log")

        with open(test_path, "rb") as tf:
            resp = self.client.post("/logs/", files={"logfile": tf, "sha256": (None, sha256(tf.name))})
            self.assertEqual(resp.status_code, 200)

    def test_post_invalid_hash(self):
        with artificial_logfile() as tf:
            resp = self.client.post("/logs/", files={"logfile": tf, "sha256": (None, "Invalidsha256")})
            self.assertEqual(resp.status_code, 400)

    def test_read_log(self):
        with artificial_logfile() as tf:
            resp = self.client.post("/logs/", files={"logfile": tf, "sha256": (None, tf.sha256)})
            self.assertEqual(resp.status_code, 200)

            resp = self.client.get(f"/logs/{os.path.basename(tf.name)}/")
            self.assertEqual(resp.status_code, 200)

            self.assertEqual(len(resp.json()), 1)

    def test_read_video_log(self):
        with artificial_logfile(video=True) as tf:
            resp = self.client.post("/logs/", files={"logfile": tf, "sha256": (None, tf.sha256)})
            self.assertEqual(resp.status_code, 200)

            resp = self.client.get(f"/logs/{os.path.basename(tf.name)}/")
            self.assertEqual(resp.status_code, 200)

            self.assertEqual(len(resp.json()), 1)
            print(resp.json())

class LogServiceRealDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.td = os.path.join(HOST_CONFIG.RECORD_DIR, "unittest")
        self.lh = LogHashes(self.td)
        app.dependency_overrides[get_loghashes] = lambda: self.lh
        
        self.client = TestClient(app)

    def test_read_reward_frame(self):
        test_log = "alphalog-22c37d10-2022-9-16-21_21.log"

        resp = self.client.get(f"/logs/{test_log}/frame_reward/120")
        self.assertEqual(resp.status_code, 200)
        


if __name__ == '__main__':
    unittest.main()
