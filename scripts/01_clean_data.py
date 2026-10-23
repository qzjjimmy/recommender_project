#!/usr/bin/env python3
"""阶段一入口：原始数据体检与三级清洗（可选合并补充数据）。

处理顺序
--------
1. **原始数据照旧**：体检 → 三级清洗 → 保留比例，这一步的产出与原来完全一致；
2. **补充数据（如果有）**：用**同一套清洗规则**再洗一遍
   （来源见 ``config/settings.py`` 的 ``extra_transactions``，由
   ``scripts/12_generate_synthetic_data.py`` 生成，**不修改 data/raw/**）；
3. **合并**：把两份干净数据拼成一张表，供后续阶段统一使用。

产物：
    data/processed/clean_transactions.csv    清洗后交易表（原始 + 补充）
    outputs/stage01/stage01_probe.txt                原始数据体检报告（仅原始数据）
    outputs/stage01/stage01_retention.csv            三级清洗保留比例（按数据源分别列出）
    outputs/stage01/stage01_top_countries.csv        国家/地区 Top10
    outputs/stage01/stage01_top_items.csv            热门商品 Top10
    outputs/stage01/stage01_spotcheck.csv            热门商品名称抽查
    outputs/stage01/stage01_extra_probe.txt          补充数据体检报告（如果有）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config.settings import PATHS, set_seed, stage_output_dir
from src import data_cleaning as dc


def main() -> None:
    set_seed()
    outputs = stage_output_dir(1)
    outputs.mkdir(parents=True, exist_ok=True)
    Path(PATHS["clean_transactions"]).parent.mkdir(parents=True, exist_ok=True)

    # ================= 步骤 1：原始数据（流程与产物保持原样） =================
    raw = dc.load_transactions(PATHS["raw_transactions"])
    products = dc.load_products(PATHS["raw_products"])
    print(f"原始交易表 : {len(raw)} 行 × {len(raw.columns)} 列")
    print(f"商品主数据 : {len(products)} 行 × {len(products.columns)} 列")
    print(f"商品表列名 : {list(products.columns)}")
    print(products.head(3).to_string(index=False))

    probe = dc.probe_raw(raw)
    dc.write_probe_report(probe, outputs / "stage01_probe.txt")
    dc.top_countries(raw).to_csv(
        outputs / "stage01_top_countries.csv", index=False, encoding="utf-8-sig"
    )
    dc.top_items(raw).to_csv(
        outputs / "stage01_top_items.csv", index=False, encoding="utf-8-sig"
    )
    print(
        "原始体检   : "
        f"行数={probe['rows']} 用户={probe['users']} 商品={probe['items']} "
        f"缺失用户={probe['missing_user_ratio']:.2%} "
        f"热门Top{probe['top_n']}占比={probe['top_n_item_share']:.2%}"
    )

    clean_main, retention_main = dc.clean_transactions(raw)
    retention_main.insert(0, "source", "原始数据")
    print("\n原始数据三级清洗漏斗：")
    print(retention_main.drop(columns=["source"]).to_string(index=False))
    print(f"原始数据清洗后: {len(clean_main)} 行（保留 {len(clean_main) / len(raw):.2%}）")

    # ================= 步骤 2：补充数据（同一套规则，分别清洗） =================
    extra_txns = dc.load_extra_sources(PATHS.get("extra_transactions", []), dc.load_transactions)
    extra_prods = dc.load_extra_sources(PATHS.get("extra_products", []), dc.load_products)

    clean_parts = [clean_main]
    retention_parts = [retention_main]
    product_parts = [products]

    if extra_txns:
        print(f"\n发现 {len(extra_txns)} 个补充数据源，用同一套规则清洗：")
        for name, extra_raw in extra_txns:
            extra_probe = dc.probe_raw(extra_raw)
            dc.write_probe_report(extra_probe, outputs / f"stage01_extra_probe_{name}.txt")
            print(
                f"  {name}: {len(extra_raw)} 行，"
                f"用户 {extra_probe['users']}，商品 {extra_probe['items']}，"
                f"缺失用户 {extra_probe['missing_user_ratio']:.2%}"
            )
            clean_extra, retention_extra = dc.clean_transactions(extra_raw)
            retention_extra.insert(0, "source", f"补充数据({name})")
            retention_parts.append(retention_extra)
            clean_parts.append(clean_extra)
            print(
                f"    清洗后 {len(clean_extra)} 行"
                f"（保留 {len(clean_extra) / len(extra_raw):.2%}）"
            )
    else:
        print("\n未发现补充数据（可运行 scripts/12_generate_synthetic_data.py 生成）。")

    if extra_prods:
        product_parts += [df for _, df in extra_prods]
        print(f"  合并商品主数据：{len(products)} → {sum(len(p) for p in product_parts)} 行")

    # ================= 步骤 3：合并 + 自检 =================
    clean = pd.concat(clean_parts, ignore_index=True)
    retention = pd.concat(retention_parts, ignore_index=True)

    rows = retention_main["rows_after"].tolist()
    if rows != sorted(rows, reverse=True):
        raise AssertionError(f"原始数据清洗后行数必须单调不增，实际为 {rows}")

    clean.to_csv(PATHS["clean_transactions"], index=False, encoding="utf-8")
    retention.to_csv(outputs / "stage01_retention.csv", index=False, encoding="utf-8-sig")

    print("\n===== 合并结果 =====")
    print(f"  清洗后总行数 : {len(clean):,}")
    print(f"  用户数       : {clean['CustomerID'].nunique():,}")
    print(f"  商品数       : {clean['StockCode'].nunique():,}")
    print(f"  发票数       : {clean['InvoiceNo'].nunique():,}")
    print(f"  数据源       : {[s for s in retention['source'].unique()]}")
    print("\n各数据源保留比例：")
    print(retention[["source", "step_name", "rows_before", "rows_removed", "rows_after"]].to_string(index=False))

    # ================= 步骤 4：商品名称抽查 =================
    merged_products = pd.concat(product_parts, ignore_index=True)
    merged_products = merged_products.drop_duplicates(subset=["StockCode"], keep="first")
    merged_products.to_csv(PATHS["products_all"], index=False, encoding="utf-8")
    print(
        f"\n合并商品主数据: {len(merged_products):,} 行 / "
        f"{merged_products['Product_Category'].nunique()} 个类目"
        f" → {PATHS['products_all']}"
    )
    spot = dc.spotcheck_product_names(clean, merged_products)
    spot.to_csv(outputs / "stage01_spotcheck.csv", index=False, encoding="utf-8-sig")
    match_rate = float(spot["matched"].mean()) if len(spot) else 0.0
    print(f"\n名称抽查   : Top{len(spot)} 商品匹配率 {match_rate:.2%}")
    print(spot[["StockCode", "interactions", "display_name", "matched"]].to_string(index=False))

    print("\n已写出：")
    for path in (
        PATHS["clean_transactions"],
        PATHS["products_all"],
        outputs / "stage01_probe.txt",
        outputs / "stage01_retention.csv",
        outputs / "stage01_top_countries.csv",
        outputs / "stage01_top_items.csv",
        outputs / "stage01_spotcheck.csv",
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
