import unittest
import tempfile
import hashlib
import os

from fastapi.testclient import TestClient
from cereal import log
from src.logutil import sha256, LogHashes
from src.tests.utils import artificial_logfile
from src.web.logservice import app, get_loghashes


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
        resp = self.client.get("/logs")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [[]])

    def test_post_empty_log(self):
        with tempfile.NamedTemporaryFile() as tf:
            resp = self.client.post("/logs", files={"logfile": tf, "sha256": (None, "invalid")})
            self.assertEqual(resp.status_code, 400)

    def test_post_invalid_log(self):
        with tempfile.NamedTemporaryFile() as tf:
            tf.write(b"Invalid Log!")
            tf.flush()
            tf.seek(0)
            resp = self.client.post("/logs", files={"logfile": tf, "sha256": (None, sha256(tf.name))})
            self.assertEqual(resp.status_code, 400)

    def test_post_log(self):
        with artificial_logfile() as tf:
            resp = self.client.post("/logs", files={"logfile": tf, "sha256": (None, tf.sha256)})
            self.assertEqual(resp.status_code, 200)

            resp = self.client.get("/logs")
            print(resp.json())

            tf.seek(0)
            sha256_hash = hashlib.sha256()
            b = tf.read()
            sha256_hash.update(b)
            self.assertEqual(resp.json()[0][0]["sha256"], sha256_hash.hexdigest())

            # Posting the same log again should error out
            tf.seek(0)
            resp = self.client.post("/logs", files={"logfile": tf})
            self.assertNotEqual(resp.status_code, 200)

    def test_post_invalid_hash(self):
        with artificial_logfile() as tf:
            resp = self.client.post("/logs", files={"logfile": tf, "sha256": (None, "Invalidsha256")})
            self.assertEqual(resp.status_code, 400)

    def test_read_log(self):
        with artificial_logfile() as tf:
            resp = self.client.post("/logs", files={"logfile": tf, "sha256": (None, tf.sha256)})
            self.assertEqual(resp.status_code, 200)

            resp = self.client.get(f"/logs/{os.path.basename(tf.name)}")
            self.assertEqual(resp.status_code, 200)

            self.assertEqual(len(resp.json()), 1)

    def test_read_video_log(self):
        with artificial_logfile(video=True) as tf:
            resp = self.client.post("/logs", files={"logfile": tf, "sha256": (None, tf.sha256)})
            self.assertEqual(resp.status_code, 200)

            resp = self.client.get(f"/logs/{os.path.basename(tf.name)}")
            self.assertEqual(resp.status_code, 200)

            self.assertEqual(len(resp.json()), 1)
            print(resp.json())


if __name__ == '__main__':
    unittest.main()
