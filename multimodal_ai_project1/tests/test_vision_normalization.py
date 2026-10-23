from __future__ import annotations

import unittest

from bridge_server import normalize_vision_result


class VisionNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.categories = ["家用电器", "家居装饰", "宿舍用品"]

    def test_fenced_json_and_object_list_are_normalized(self) -> None:
        raw = """```json
{"scene":"室内有风扇","objects":[{"name":"风扇","color":"白色"}],
 "colors":["白色"],"category_hints":["家用电器"],"search_query":"白色风扇"}
```"""
        result = normalize_vision_result(raw, self.categories)
        self.assertEqual(result["scene"], "室内有风扇")
        self.assertEqual(result["objects"], ["风扇 白色"])
        self.assertEqual(result["category_hints"], ["家用电器"])
        self.assertEqual(result["search_query"], "白色风扇")

    def test_invalid_category_is_removed(self) -> None:
        raw = (
            '{"scene":"房间","objects":["台灯"],"colors":["蓝色"],'
            '"category_hints":["不存在的类目"],"search_query":""}'
        )
        result = normalize_vision_result(raw, self.categories)
        self.assertEqual(result["category_hints"], [])
        self.assertEqual(result["search_query"], "台灯 蓝色")

    def test_malformed_output_keeps_only_short_description(self) -> None:
        result = normalize_vision_result("这里是一段普通描述，不是 JSON", self.categories)
        self.assertEqual(result["scene"], "这里是一段普通描述，不是 JSON")
        self.assertEqual(result["objects"], [])
        self.assertEqual(result["category_hints"], [])
        self.assertEqual(result["search_query"], "")

    def test_outputs_are_capped(self) -> None:
        raw = (
            '{"scene":"场景","objects":["1","2","3","4","5","6"],'
            '"colors":["1","2","3","4","5","6"],'
            '"category_hints":["家用电器","家居装饰","宿舍用品"],'
            '"search_query":"查询"}'
        )
        result = normalize_vision_result(raw, self.categories)
        self.assertEqual(len(result["objects"]), 5)
        self.assertEqual(len(result["colors"]), 5)
        self.assertLessEqual(len(result["category_hints"]), 3)


if __name__ == "__main__":
    unittest.main()
