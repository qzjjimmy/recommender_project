#!/usr/bin/env python3
"""应用层数据准备：生成商品画像与演示客户列表。

智能推荐需要"商品 ID → 名称 / 类目 / 均价 / 热度"的画像表，以及一批
有辨识度的演示客户（注册时可选，用于初始化个性化向量）。

这两个文件由本脚本显式生成；Web 服务启动时若发现缺失也会自动生成，
但提前跑一次可以让服务启动更快、也便于单独检查数据是否正确。

产物：
    data/processed/item_profile.csv      商品画像
    data/processed/demo_customers.json   演示客户列表
    data/processed/customer_history.csv  推荐首页：客户 × 商品购买历史
    data/processed/season_stats.csv      推荐首页：月份 × 类目季节指数

另外会初始化两个 SQLite 库（若不存在）：
    data/app_users.db       账号
    data/app_feedback.db    点赞 / 点踩

并确保两个演示账号存在（已存在则跳过）：
    demo_user / demo123456   关联客户 14897（原始数据里的家居装饰偏好客户）
    qzj       / 123456       关联客户 QZJ001（**大学生画像**：数码 / 文具 / 图书 / 宿舍用品）
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config.settings import PATHS, set_seed
from src import feedback as fb
from src import product_catalog as catalog
from src import smart_search as ss
from src import synthetic_data as syn
from src import user_store
from src.purchase_history import load_purchase_history
from src.seasonal import load_season_table

# 演示账号：注册时也可以手动选这些客户，这里直接建好省事
DEMO_ACCOUNTS = [
    {
        "username": "demo_user",
        "password": "demo123456",
        "display_name": "演示用户",
        "customer_id": "14897",
    },
    {
        "username": "qzj",
        "password": "123456",
        "display_name": "QZJ（大学生）",
        "customer_id": "QZJ001",
    },
]


def persona_entries() -> list[dict]:
    """把 ``src/synthetic_data.py`` 里的人物画像整理成"演示客户"条目。

    画像客户的购买记录是脚本 12 按剧本生成的（大学生 QZJ 的入学采购、618 换手机、
    开学季买平板……），这里统计出他的商品数与偏好类目，写进 demo_customers.json，
    这样注册页的下拉框里也能选到他。
    """
    clean = pd.read_csv(
        PATHS["clean_transactions"], dtype=str, usecols=["StockCode", "CustomerID"]
    )
    categories = catalog.build_category_map(catalog.load_products())
    entries: list[dict] = []
    for persona in syn.PERSONAS:
        customer = str(persona["customer_id"])
        sub = clean[clean["CustomerID"].astype(str) == customer]
        if sub.empty:
            continue
        counts = sub["StockCode"].map(categories).fillna(catalog.UNKNOWN_CATEGORY).value_counts()
        top_category = str(counts.index[0])
        share = float(counts.iloc[0]) / max(1, int(counts.sum()))
        entries.append(
            {
                "customer_id": customer,
                "n_items": int(sub["StockCode"].nunique()),
                "top_category": top_category,
                "category_share": round(share, 3),
                "label": f"{persona['label']}｜历史 {sub['StockCode'].nunique()} 件商品，"
                f"主类目 {top_category} 占 {share:.0%}",
            }
        )
    return entries


def ensure_accounts() -> None:
    """确保演示账号存在；已存在就跳过（不覆盖用户自己改过的密码）。"""
    for account in DEMO_ACCOUNTS:
        existing = user_store.get_user(PATHS["user_db"], account["username"])
        if existing:
            print(
                f"  {account['username']:<10} 已存在（关联客户 {existing['customer_id'] or '—'}）"
            )
            continue
        try:
            user_store.create_user(
                PATHS["user_db"],
                account["username"],
                account["password"],
                account["display_name"],
                account["customer_id"],
            )
            print(
                f"  {account['username']:<10} 已创建（密码 {account['password']}，"
                f"关联客户 {account['customer_id']}）"
            )
        except ValueError as exc:
            print(f"  {account['username']:<10} 创建失败：{exc}")


def main() -> None:
    set_seed()

    print("正在生成商品画像 ...")
    profile = ss.load_item_profile(force=True)
    print(f"  商品画像 : {len(profile)} 件商品 → {PATHS['item_profile']}")
    print(f"  类目数量 : {profile['category'].nunique()} 个")
    print(f"  价格范围 : {profile['price'].min():.2f} ~ {profile['price'].max():.2f}")
    print("  各类目商品数 Top8：")
    for cat, count in profile["category"].value_counts().head(8).items():
        print(f"    {cat:<8} {count}")

    print("\n正在挑选演示客户 ...")
    customers = ss.load_demo_customers(force=True)
    # 人物画像（QZJ 等）排在最前面：注册页第一眼就能看到"大学生"这个画像
    personas = persona_entries()
    persona_ids = {str(p["customer_id"]) for p in personas}
    customers = personas + [c for c in customers if str(c["customer_id"]) not in persona_ids]
    Path(PATHS["demo_customers"]).write_text(
        json.dumps(customers, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  演示客户 : {len(customers)} 位 → {PATHS['demo_customers']}")
    for c in customers:
        print(f"    {c['customer_id']:<8} {c['label']}")

    print("\n正在初始化应用数据库 ...")
    user_store.init_db(PATHS["user_db"])
    fb.init_db(PATHS["feedback_db"])
    print(f"  账号库   : {PATHS['user_db']}（已有 {user_store.count_users(PATHS['user_db'])} 位用户）")
    print(f"  反馈库   : {PATHS['feedback_db']}")
    ensure_accounts()

    print("\n正在生成推荐首页数据（购买历史 + 季节统计，约 30 秒）...")
    history = load_purchase_history(force=True)
    stats = history.stats()
    print(
        f"  购买历史 : {stats['customers']} 位客户 / {stats['pairs']} 条记录"
        f" → {PATHS['customer_history']}"
    )
    seasons = load_season_table(force=True)
    print(f"  季节统计 : {len(seasons)} 条（月份 × 类目） → {PATHS['season_stats']}")

    print("\n完成。接下来可以启动服务：")
    print("  python web/app.py   然后浏览器打开 http://127.0.0.1:5000/home")


if __name__ == "__main__":
    main()
