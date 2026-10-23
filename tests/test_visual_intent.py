from __future__ import annotations

import unittest

from src import visual_intent as vi


class VisualIntentTests(unittest.TestCase):
    def test_compose_ignores_user_price_and_deduplicates(self) -> None:
        analysis = {
            "scene": "室内有冰箱",
            "objects": ["冰箱", "洗衣机", "冰箱"],
            "colors": ["白色", "蓝色"],
            "category_hints": ["家用电器", "家用电器"],
            "search_query": "冰箱 洗衣机",
        }
        query = vi.compose_search_query("200元以内", analysis)
        self.assertNotIn("200", query)
        self.assertNotIn("元", query)
        self.assertIn("家用电器", query)
        self.assertEqual(query.count("冰箱"), 1)
        self.assertNotIn("白色", query)

    def test_normalize_removes_prices_from_model_fields(self) -> None:
        analysis = {
            "scene": "桌面有预算100元的台灯",
            "objects": ["100元以内的台灯", "20到80元收纳盒"],
            "colors": ["白色"],
            "category_hints": ["宿舍用品"],
            "search_query": "价格100元 台灯 收纳盒",
        }
        result = vi.normalize_analysis(analysis, ["宿舍用品"])
        query = vi.compose_search_query("", result)
        self.assertNotIn("100", query)
        self.assertNotIn("80", query)
        self.assertNotIn("元", query)
        self.assertIn("台灯", query)
        self.assertIn("收纳盒", query)
        self.assertIn("宿舍用品", query)
        self.assertNotIn("的台灯", query)

    def test_normalize_filters_invalid_and_generic_terms(self) -> None:
        analysis = {
            "scene": "  一个宿舍房间  ",
            "objects": ["人物", "台灯", "台灯"],
            "colors": ["蓝色"],
            "category_hints": ["宿舍用品", "不存在的类目"],
            "search_query": "",
        }
        result = vi.normalize_analysis(
            analysis, ["宿舍用品", "家用电器", "家居装饰"])
        self.assertEqual(result["scene"], "一个宿舍房间")
        self.assertEqual(result["objects"], ["台灯"])
        self.assertEqual(result["category_hints"], ["宿舍用品"])
        self.assertNotIn("人物", result["objects"])

    def test_empty_analysis_does_not_add_facts(self) -> None:
        result = vi.normalize_analysis({}, ["家用电器"])
        self.assertFalse(vi.has_visual_terms(result))
        self.assertEqual(vi.compose_search_query("100元以内", result), "")


if __name__ == "__main__":
    unittest.main()
