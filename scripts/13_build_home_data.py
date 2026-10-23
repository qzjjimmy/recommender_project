#!/usr/bin/env python3
"""推荐首页的数据准备：用户购买历史 + 季节性统计。

首页的两个板块各需要一份"预聚合"的数据，都从干净交易表算一次、落盘缓存：

    data/processed/customer_history.csv  客户 × 商品（订单数 / 件数 / 首末购买时间）
    data/processed/season_stats.csv      月份 × 类目（成交额 / 月内份额 / 季节指数）

Web 服务启动时若发现缺文件也会自动生成，但提前跑一次更快，而且能在这里把
「档期日历 × 数据实测」并排打印出来，方便检查日历写得合不合理。

运行（项目根目录）：
    python scripts/13_build_home_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config.settings import PATHS, set_seed
from src import product_catalog as catalog
from src import seasonal
from src.purchase_history import build_customer_history, load_purchase_history
from src.seasonal import build_season_stats, load_season_table


def main() -> None:
    set_seed()
    clean = pd.read_csv(PATHS["clean_transactions"], dtype=str)
    products = catalog.load_products()
    print(f"干净交易表：{len(clean)} 行")

    # ---------------- 购买历史 ----------------
    print("\n[1/2] 正在聚合用户购买历史 ...")
    history = build_customer_history(clean)
    Path(PATHS["customer_history"]).parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(PATHS["customer_history"], index=False, encoding="utf-8")
    store = load_purchase_history(PATHS["customer_history"], force=True)
    stats = store.stats()
    print(f"  {stats['customers']} 位客户 / {stats['pairs']} 条（客户, 商品）记录")
    print(f"  每位客户商品数：中位数 {stats['median_items']}，最多 {stats['max_items']}")
    print(f"  → {PATHS['customer_history']}")

    # ---------------- 季节统计 ----------------
    print("\n[2/2] 正在计算季节性指数 ...")
    window = seasonal.common_window([PATHS["raw_transactions"], *PATHS["extra_transactions"]])
    if window is not None:
        print(
            "  统计窗口 : "
            f"{window[0]:%Y-%m-%d} ~ {window[1]:%Y-%m-%d}"
            "（原始数据与补充数据都覆盖的区间）"
        )
    table = build_season_stats(clean, products, window=window)
    table.to_csv(PATHS["season_stats"], index=False, encoding="utf-8")
    seasons = load_season_table(PATHS["season_stats"], force=True)
    print(f"  {len(seasons)} 条（月份, 类目）记录，覆盖 {len(seasons.months)} 个月")
    print("  口径：季节指数 = 该类目当月**日均**成交额 ÷ 该类目全年日均成交额")
    print("        （按天归一，既处理月份天数不同，也处理最后一个月只有 9 天的情况）")
    print(f"  → {PATHS['season_stats']}")

    # ---------------- 档期日历 × 数据实测 ----------------
    print("\n档期日历 vs 数据实测（每个月的两个档期 + 当月季节指数最高的类目）")
    print(f"  {'月份':<6}{'档期':<26}{'季节指数最高的类目（实测）'}")
    for month in range(1, 13):
        themes = seasonal.themes_for(month)
        names = " / ".join(t["name"] for t in themes)
        top = seasons.top_categories(month, k=4)
        evidence = "、".join(f"{r['category']}×{r['lift']:.1f}" for r in top)
        print(f"  {month:>2} 月  {names:<24}{evidence}")

    print("\n完成。接下来启动服务后打开 http://127.0.0.1:5000/home")


if __name__ == "__main__":
    main()
