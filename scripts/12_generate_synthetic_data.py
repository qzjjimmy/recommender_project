#!/usr/bin/env python3
"""生成补充数据（模拟"外部采集 + 下单"），**不触碰 data/raw/ 下的原始数据**。

为什么需要它
------------
原始数据只有 4070 个商品、22 个类目，且**均价中位数只有 1.95 元** ——
这导致两个问题：商品太少、价格带太窄（"100 元以内"几乎不过滤）。
本脚本生成一批结构完全一致的补充数据，覆盖 22 个新类目与几十到上万元的价格带。

v2 还补齐了推荐首页需要的两件事（见 `补充数据生成与合并.md`）：

    · **真实季节性**：每个类目按月度权重抽日期，于是 8~9 月开学季数码/文具真的更旺、
      11 月双 11、12 月圣诞也真的成立 —— 首页的季节指数就是从这份数据算出来的；
    · **品牌化 3C + 补齐薄类目**：手机/笔记本/平板/耳机等用真实品牌型号命名，
      `文具用品`（原始只有 4 个商品）与 `办公用品`（3 个）各补 200 个，另新增 `宿舍用品`。

两条硬约束
----------
1. **不修改原始数据**：结果写到 ``data/generated/``，原始文件一个字节都不动；
2. **不改清洗规则**：生成的表结构与原始完全一致（transactions 8 列 / products 4 列），
   阶段一那套三级清洗漏斗原样套用。

产物：
    data/generated/transactions_syn.csv   合成交易明细
    data/generated/products_syn.csv       合成商品主数据
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config.settings import PATHS, SEED, set_seed
from src import product_catalog as catalog
from src import synthetic_data as syn

# 生成规模（想调整只改这里）
PER_CATEGORY = 200  # 每个新类目生成多少商品
FILLER_PER_CATEGORY = 80  # 每个**原始类目**派生多少"高端款"（让客户偏好落得到商品上）
INVOICE_COUNT = 78000  # 生成多少张订单（类目从 12 个涨到 22 个，订单量同步提高）
NEW_CUSTOMERS = 900  # 新增多少位"新客户"
START_INVOICE = 800001  # 发票号起始（远离原始数据的号段，不冲突）
# 时间范围刻意**与原始数据完全一致**：这样"所有数据源都覆盖的窗口"就是整段时间，
# 季节统计不会因为某个来源缺了某个月而失真（见 src/seasonal.py::common_window）。
DATE_START = "2010-12-01"
DATE_END = "2011-12-09"


def main() -> None:
    set_seed()
    out_dir = Path(PATHS["generated_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 取**原始数据**里的客户与热门国家，让新数据"接着长" ----------
    # 固定读 data/raw/：这样重复运行本脚本时，参考的客户集合不会随
    # 上一次生成的合成客户而膨胀（保证可复现）。
    source = Path(PATHS["raw_transactions"])
    print(f"参考原始数据：{source.name}")
    ref = pd.read_csv(source, dtype=str, encoding="latin-1", usecols=["CustomerID", "Country"])
    customers = sorted(ref["CustomerID"].dropna().astype(str).str.strip().unique())
    customers = [c for c in customers if c and c.lower() != "nan"]
    top_countries = ref["Country"].dropna().astype(str).value_counts().head(6).index.tolist()
    print(f"  沿用原客户 {len(customers)} 位；国家池 {top_countries}")

    # ---------- 每位原始客户最常买的 1~3 个类目 → 作为他在补充数据里的偏好 ----------
    # 这样新数据会"贴着"这些客户的真实口味长，而不是让每个人随机逛；
    # 没有这层偏好结构，用户-商品矩阵就是噪声，协同过滤会输给热门基线。
    hist = pd.read_csv(
        source, dtype=str, encoding="latin-1", usecols=["CustomerID", "StockCode"]
    )
    cat_map = catalog.build_category_map(catalog.load_products(PATHS["raw_products"]))
    hist["类别"] = hist["StockCode"].map(cat_map)
    hist = hist.dropna(subset=["类别"])
    counts = hist.groupby(["CustomerID", "类别"]).size().rename("n").reset_index()
    counts["rank"] = counts.groupby("CustomerID")["n"].rank(method="first", ascending=False)
    top = counts[counts["rank"] <= 3]
    prefs = top.groupby("CustomerID")["类别"].agg(list).to_dict()
    print(f"  客户偏好 : {len(prefs)} 位客户带偏好类目（取各自最常买的 3 个类目）")

    # ---------- 原始商品表：用来派生"同品类高端款" ----------
    # 没有这一层，原始客户的偏好类目（家居装饰 / 装饰品 / 厨房用品…）在补充数据里没有商品，
    # 偏好只能退化成随机分配（见 src/synthetic_data.py 里 RAW_FILLER_SPECS 的说明）。
    raw_products = pd.read_csv(PATHS["raw_products"], dtype=str, encoding="utf-8-sig")
    print(f"  原始商品表：{len(raw_products)} 个商品（用于派生同品类高端款）")

    # ---------- 生成 ----------
    print(f"\n正在生成：{len(syn.CATEGORY_SPECS)} 个新类目 × 每类 {PER_CATEGORY} 个商品 …")
    transactions, products = syn.generate_synthetic(
        seed=SEED,
        per_category=PER_CATEGORY,
        invoice_count=INVOICE_COUNT,
        existing_customers=customers,
        customer_preferences=prefs,
        raw_products=raw_products,
        filler_per_category=FILLER_PER_CATEGORY,
        new_customer_count=NEW_CUSTOMERS,
        start_invoice=START_INVOICE,
        country_pool=top_countries,
        date_start=DATE_START,
        date_end=DATE_END,
    )

    txn_path = out_dir / "transactions_syn.csv"
    prd_path = out_dir / "products_syn.csv"
    # 交易表只有 ASCII 内容，用 utf-8 写、用 latin-1 读的结果一致；
    # 商品表含中文，用 utf-8 写（读取端用 utf-8-sig）。
    transactions.to_csv(txn_path, index=False, encoding="utf-8")
    products.to_csv(prd_path, index=False, encoding="utf-8")

    # ---------- 摘要 ----------
    prices = pd.to_numeric(transactions["UnitPrice"], errors="coerce")
    print("\n===== 生成结果 =====")
    print(f"  商品主数据 : {len(products):,} 个商品 / {products['Product_Category'].nunique()} 个类目")
    print(f"  交易明细   : {len(transactions):,} 行 / {transactions['InvoiceNo'].nunique():,} 张订单")
    print(f"  价格区间   : {prices.min():.2f} ~ {prices.max():.2f}（中位数 {prices.median():.2f}）")
    print(f"  客户数     : {transactions['CustomerID'].replace('', pd.NA).nunique():,}")
    print("  注入的脏数据（供清洗漏斗处理）：")
    print(f"    缺失 CustomerID : {int((transactions['CustomerID'] == '').sum()):,} 行")
    print(f"    退货（C 发票）  : {int(transactions['InvoiceNo'].str.startswith('C').sum()):,} 行")
    print(f"    异常价格(≤0)    : {int((prices <= 0).sum()):,} 行")
    print("\n  各类目商品数：")
    for cat, n in products["Product_Category"].value_counts().items():
        print(f"    {cat:<8} {n}")

    # ---------- 季节性体检：9 月 / 11 月占各月平均的倍数 ----------
    months = pd.to_datetime(
        transactions["InvoiceDate"], format="%m/%d/%Y %H:%M", errors="coerce"
    ).dt.month
    by_cat = transactions.assign(月份=months, 类目=transactions["StockCode"].map(
        dict(zip(products["StockCode"], products["Product_Category"]))
    ))
    print("\n  季节性体检（该类目当月行数 ÷ 各月平均行数；>1 表示这个月更旺）：")
    print(f"    {'类目':<8}{'9 月':>8}{'11 月':>8}{'12 月':>8}")
    for cat in ["手机", "笔记本电脑", "平板电脑", "数码配件", "文具用品", "图书音像",
                "宿舍用品", "家用电器", "珠宝首饰", "运动户外"]:
        sub = by_cat[by_cat["类目"] == cat]
        if sub.empty:
            continue
        counts = sub.groupby("月份").size()
        mean = counts.sum() / 12
        lift = {m: counts.get(m, 0) / mean for m in (9, 11, 12)}
        print(f"    {cat:<8}{lift[9]:>8.2f}{lift[11]:>8.2f}{lift[12]:>8.2f}")

    # ---------- 人物画像 ----------
    for persona in syn.PERSONAS:
        rows = transactions[transactions["CustomerID"] == persona["customer_id"]]
        print(
            f"\n  人物画像   : {persona['customer_id']} {persona['label']}"
            f" → {len(rows)} 行 / {rows['StockCode'].nunique()} 件商品 / "
            f"{rows['InvoiceNo'].nunique()} 次采购"
        )

    print(f"\n已写出：\n  {txn_path}\n  {prd_path}")
    print("\n下一步：重新跑阶段一，会自动把这份数据一起清洗并合并：")
    print("  python scripts\\01_clean_data.py")


if __name__ == "__main__":
    main()
