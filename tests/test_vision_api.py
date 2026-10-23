from __future__ import annotations

import io
import unittest
from unittest.mock import Mock, patch

from src import multimodal_client as mmc
from web.app import app


class VisionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        with self.client.session_transaction() as session:
            session["username"] = "qzj"

    def test_requires_login(self) -> None:
        anonymous = app.test_client()
        response = anonymous.post(
            "/api/vision/analyze",
            data=b"\xff\xd8\xffimg",
            content_type="image/jpeg",
        )
        self.assertEqual(response.status_code, 401)

    def test_missing_and_invalid_image(self) -> None:
        missing = self.client.post(
            "/api/vision/analyze", data={}, content_type="multipart/form-data")
        invalid = self.client.post(
            "/api/vision/analyze", data=b"not-an-image", content_type="image/jpeg")
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(invalid.status_code, 400)
        self.assertIn("hint", missing.get_json())

    def test_bridge_unavailable_has_text_fallback(self) -> None:
        fake = Mock()
        fake.vision.side_effect = mmc.BridgeUnavailable("bridge down")
        with patch("web.app.MULTIMODAL", fake):
            response = self.client.post(
                "/api/vision/analyze",
                data=b"\xff\xd8\xffimage",
                content_type="image/jpeg",
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["fallback"], "text-only")

    def test_success_builds_visual_query(self) -> None:
        fake = Mock()
        fake.vision.return_value = {
            "scene": "室内有白色洗衣机",
            "objects": ["洗衣机", "冰箱"],
            "colors": ["白色"],
            "category_hints": ["家用电器", "家居装饰"],
            "search_query": "洗衣机 冰箱",
            "model": "qwen2.5vl:3b",
            "elapsed_ms": 3200,
        }
        with patch("web.app.MULTIMODAL", fake):
            response = self.client.post(
                "/api/vision/analyze",
                data={
                    "image": (io.BytesIO(b"\xff\xd8\xffimage"), "scene.jpg"),
                    "query": "200元以内",
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertNotIn("200", data["suggested_query"])
        self.assertNotIn("元", data["suggested_query"])
        self.assertIn("家用电器", data["suggested_query"])
        self.assertIn("洗衣机", data["suggested_query"])
        self.assertTrue(data["has_visual_terms"])


if __name__ == "__main__":
    unittest.main()
