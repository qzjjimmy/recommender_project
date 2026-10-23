#!/usr/bin/env python3
"""配置大模型 API Key，并做一次连通性自检。

用法（在项目根目录）：
    python scripts/11_set_llm_key.py sk-xxxxxxxx        # 保存并自检
    python scripts/11_set_llm_key.py --check            # 只自检，不修改
    python scripts/11_set_llm_key.py --clear            # 删除已保存的 Key

Key 保存位置：``data/.llm_key``（已加入 .gitignore，不会进版本库）。
优先级：显式传入 > 环境变量 RECOMMENDER_LLM_API_KEY > 本地文件。

自检会真的发一次请求，成功后会打印模型解析出的结构化条件；
如果失败会把 HTTP 状态码与错误信息原样打出来，便于定位是 Key 错了、
余额不足、还是网络不通。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import HYPERPARAMS, PATHS
from src import llm_parser

TEST_QUERY = "便宜又好看的家居小礼物"


def main() -> None:
    args = [a for a in sys.argv[1:]]

    if "--clear" in args:
        target = Path(PATHS["llm_key_file"])
        if target.exists():
            target.unlink()
            print(f"已删除 {target}")
        else:
            print(f"{target} 不存在，无需删除")
        return

    if "--check" not in args and args:
        key = args[0].strip()
        target = llm_parser.save_api_key(key)
        print(f"已保存 API Key → {target}")
        print(f"（长度 {len(key)}，前 6 位 {key[:6]}…；该文件已 gitignore，不会进版本库）")

    cfg = HYPERPARAMS.get("search", {}).get("llm", {})
    print(f"\n配置：enabled={cfg.get('enabled')}  model={cfg.get('model')}  base_url={cfg.get('base_url')}")

    parser = llm_parser.build_llm_parser(cfg)
    print(f"状态：{parser.describe()}")
    if not parser.api_key:
        print("\n没有可用的 API Key，请先运行：python scripts/11_set_llm_key.py sk-xxxx")
        return

    # 自检时强制启用，避免因为 enabled=False 而跳过真实请求
    parser.enabled = True
    print(f"\n正在用示例查询做连通性自检：{TEST_QUERY!r} …")

    from src import smart_search as ss

    profile = ss.load_item_profile()
    categories = list(profile["category"].dropna().unique())
    prices = profile["price"].astype(float)
    prices = prices[prices > 0]
    price_context = (
        f"本数据集商品均价约 {prices.median():.1f} 元，价格区间 {prices.min():.1f}~{prices.max():.0f} 元。"
        if len(prices) else ""
    )

    try:
        result = parser.parse(TEST_QUERY, categories, price_context)
    except Exception as exc:  # noqa: BLE001
        print(f"自检失败（异常）：{type(exc).__name__}: {exc}")
        return

    if result is None:
        print(
            "自检失败：没有拿到有效的解析结果。\n"
            f"  调用次数 {parser.calls}，失败次数 {parser.failures}\n"
            "  常见原因：Key 无效 / 余额不足 / 网络不通 / 返回不是合法 JSON。"
        )
        return

    print("自检通过 ✔")
    print(f"  解析出的条件：类别={result.categories or '无'}"
          f"  价格={result.min_price}~{result.max_price}"
          f"  关键词={result.keywords or '无'}")
    print("\n下一步：确认 config/settings.py 里 search.llm.enabled = True，然后重启 web/app.py。")


if __name__ == "__main__":
    main()
