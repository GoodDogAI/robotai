import unittest
import tempfile
import hashlib

from unittest.mock import patch
from fastapi.testclient import TestClient
from src.logutil import sha256, LogHashes
from src.web.main import app


class LogServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(lambda: self.td.cleanup())

        with patch("src.web.main.get_loghashes") as lh:
            lh.return_value = LogHashes(self.td.name)
            self.client = TestClient(app)

    def test_empty_logs(self):
        resp = self.client.get("/logs")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])

    def test_post_log(self):
        tf = tempfile.NamedTemporaryFile("wb+")
        self.addCleanup(lambda: tf.close())

        tf.write(b"Wow, this is a log")
        tf.flush()
        tf.seek(0)

        resp = self.client.post("/logs", files={"logfile": tf})
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get("/logs")
        print(resp.json())

        sha256_hash = hashlib.sha256()
        sha256_hash.update(b"Wow, this is a log")
        self.assertEqual(resp.json()[0]["sha256"], sha256_hash.hexdigest())

if __name__ == '__main__':
    unittest.main()
