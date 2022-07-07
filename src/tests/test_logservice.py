import unittest
import tempfile
import hashlib

from fastapi.testclient import TestClient
from src.logutil import sha256, LogHashes
from src.web.logservice import app, get_loghashes


class LogServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        lh = LogHashes(self.td.name)
        app.dependency_overrides[get_loghashes] = lambda: lh
        self.addCleanup(lambda: self.td.cleanup())

        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides = {}

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

        # Posting the same log again should error out
        tf.seek(0)
        resp = self.client.post("/logs", files={"logfile": tf})
        self.assertNotEqual(resp.status_code, 200)


if __name__ == '__main__':
    unittest.main()
