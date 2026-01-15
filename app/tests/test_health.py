import unittest
from fastapi.testclient import TestClient

from app.main import app


class TestHealth(unittest.TestCase):
    def test_health(self):
        with TestClient(app) as client:
            r = client.get("/health")
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertEqual(data.get("status"), "ok")
            self.assertIn("ollama", data)
