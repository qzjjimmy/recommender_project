"""项目三视觉选品联调自检。

前置条件：
  1. 在项目二环境启动 ``bridge_server.py``（默认 127.0.0.1:8100）；
  2. 当前目录下已存在推荐系统模型与商品画像。

脚本使用 Flask test client 模拟已登录用户，上传无人物隐私的家电示意图，
验证“图片 → 视觉线索 → 建议查询 → 现有推荐”的完整链路。结果写入
``outputs/stage22/stage22_vision_recommendation.json``，供报告追溯。
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import stage_output_dir  # noqa: E402
from web.app import MULTIMODAL, SEARCHER, app  # noqa: E402

IMAGE_PATH = ROOT / "web" / "static" / "img" / "photos" / "家用电器.jpg"
OUTPUT_PATH = stage_output_dir(22) / "stage22_vision_recommendation.json"
USERNAME = "qzj"


def main() -> int:
    if SEARCHER is None:
        print("[FAIL] 智能推荐资源未加载")
        return 1
    if not IMAGE_PATH.exists():
        print(f"[FAIL] 缺少测试图片：{IMAGE_PATH}")
        return 1

    status = MULTIMODAL.status()
    if not (status.get("available") and status.get("vision")):
        print(f"[FAIL] 视觉桥接不可用：{status.get('reason') or status}")
        print("       请先启动：cd multimodal_ai_project1 && python bridge_server.py")
        return 1

    client = app.test_client()
    with client.session_transaction() as session:
        session["username"] = USERNAME

    started = time.time()
    response = client.post(
        "/api/vision/analyze",
        data={
            "image": (io.BytesIO(IMAGE_PATH.read_bytes()), "appliance.jpg"),
            "query": "200元以内",
        },
        content_type="multipart/form-data",
    )
    request_elapsed_ms = int((time.time() - started) * 1000)
    data = response.get_json(silent=True) or {}
    if response.status_code != 200:
        print(f"[FAIL] /api/vision/analyze HTTP {response.status_code}: {data}")
        return 1

    vision = data.get("vision") or {}
    suggested = str(data.get("suggested_query") or "").strip()
    if not vision.get("scene") or not suggested:
        print(f"[FAIL] 视觉结果缺少场景或建议查询：{data}")
        return 1
    if "家用电器" not in (vision.get("category_hints") or []):
        print(f"[FAIL] 家电示意图未命中“家用电器”：{vision}")
        return 1
    if any(marker in suggested for marker in ("200", "元", "价格", "预算")):
        print(f"[FAIL] 视觉建议查询不应包含价格条件：{suggested!r}")
        return 1

    recommendation = SEARCHER.search(
        suggested, username=USERNAME, top_n=5)
    if not recommendation.get("results"):
        print(f"[FAIL] 视觉建议查询未返回推荐：{suggested!r}")
        return 1

    evidence = {
        "tested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image": "photos/家用电器.jpg",
        "user_query": "200元以内",
        "request_elapsed_ms": request_elapsed_ms,
        "vision_model": data.get("model", ""),
        "vision_elapsed_ms": data.get("elapsed_ms", 0),
        "vision": vision,
        "suggested_query": suggested,
        "recommendation": {
            "filtered_items": recommendation.get("filtered_items", 0),
            "returned": recommendation.get("returned", 0),
            "constraints": recommendation.get("constraints", {}),
            "top_items": [
                {
                    "item_id": row.get("item_id"),
                    "name": row.get("name"),
                    "category": row.get("category"),
                    "price": row.get("price"),
                    "reason": row.get("reason"),
                }
                for row in recommendation.get("results", [])[:5]
            ],
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "[PASS] 图片 → 视觉线索 → 推荐闭环\n"
        f"        类目：{vision.get('category_hints')}\n"
        f"        建议查询：{suggested}\n"
        f"        推荐：{recommendation.get('returned')} 条\n"
        f"        证据：{OUTPUT_PATH.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
