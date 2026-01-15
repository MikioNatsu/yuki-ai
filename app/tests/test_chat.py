import unittest
from fastapi.testclient import TestClient

from app.main import app


class TestChatValidation(unittest.TestCase):
    def test_chat_empty_text(self):
        with TestClient(app) as client:
            r = client.post("/chat", json={"session_id": "s1", "user_text": "   "})
            self.assertEqual(r.status_code, 422)
            data = r.json()
            self.assertEqual(data.get("code"), "validation_error")
