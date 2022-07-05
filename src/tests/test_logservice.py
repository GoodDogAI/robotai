import unittest
import tempfile

from unittest.mock import patch
from fastapi.testclient import TestClient


class LogServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(lambda: self.td.cleanup())

        with patch.dict("os.environ", {"ROBOTAI_RECORD_DIR": self.td.name}):
            from src.web.main import app
            self.client = TestClient(app)

    def test_empty_logs(self):
        resp = self.client.get("/logs")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])


if __name__ == '__main__':
    unittest.main()
