"""阶段三：商品名称与类目拼装。

检索只给出商品 ID，展示需要可读名称。本模块把商品主数据整理成
``StockCode -> 名称/类目`` 的查询结构，并在查不到时明确返回"未知商品"，
保证下游不会因为个别编码缺失而中断。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import PATHS

UNKNOWN_NAME = "未知商品"
UNKNOWN_CATEGORY = "未知类目"


def load_products(path: Path | None = None) -> pd.DataFrame:
    """读取商品主数据，**优先用合并后的表**。

    ``data/processed/products_all.csv``（原始 + 补充数据，由阶段一生成）
    存在就用它；不存在则回退到 ``data/raw/products.csv``。
    这样无论有没有补充数据，下游脚本都不用改代码。
    """
    target = Path(path) if path else Path(PATHS["products_all"])
    if not target.exists():
        target = Path(PATHS["raw_products"])
    return pd.read_csv(target, dtype=str, encoding="utf-8-sig")


def normalize_products(products: pd.DataFrame) -> pd.DataFrame:
    """统一 ``StockCode`` 为字符串并去空格，按首次出现去重。

    原始 ``products.csv`` 存在重复 ``StockCode``（3866 行 / 3642 唯一），
    且部分重复行的中文名或类目不同。这里固定"取首次出现"的口径并写进文档，
    避免同一个商品在不同阶段显示成不同名称。
    """
    prod = products.copy()
    prod["StockCode"] = prod["StockCode"].astype(str).str.strip()
    return prod.drop_duplicates("StockCode", keep="first")


def build_name_map(products: pd.DataFrame) -> dict[str, str]:
    """``StockCode -> 可读名称``（中文优先，退回英文，再退回"未知商品"）。"""
    prod = normalize_products(products)
    name = pd.Series(pd.NA, index=prod.index, dtype="string")
    if "Chinese_Description" in prod.columns:
        name = name.fillna(prod["Chinese_Description"])
    if "Description" in prod.columns:
        name = name.fillna(prod["Description"])
    return dict(zip(prod["StockCode"], name.fillna(UNKNOWN_NAME).astype(str)))


def build_category_map(products: pd.DataFrame) -> dict[str, str]:
    """``StockCode -> 类目``；缺失时返回"未知类目"。"""
    prod = normalize_products(products)
    if "Product_Category" not in prod.columns:
        return {code: UNKNOWN_CATEGORY for code in prod["StockCode"]}
    category = prod["Product_Category"].fillna(UNKNOWN_CATEGORY).astype(str)
    return dict(zip(prod["StockCode"], category))


def build_name_lookup(products: pd.DataFrame) -> dict[str, str]:
    """``StockCode -> 展示名称``（与 :func:`build_name_map` 同义，便于命名统一）。"""
    return build_name_map(products)


def get_product_name(products: pd.DataFrame, item_id: str) -> str:
    """根据商品 ID 查名称；找不到返回"未知商品"而不是报错。"""
    return build_name_map(products).get(str(item_id).strip(), UNKNOWN_NAME)


def get_product_category(products: pd.DataFrame, item_id: str) -> str:
    """根据商品 ID 查类目；找不到返回"未知类目"。"""
    return build_category_map(products).get(str(item_id).strip(), UNKNOWN_CATEGORY)


def enrich_results(
    pairs: list[tuple[str, float]],
    products: pd.DataFrame,
    names: dict[str, str] | None = None,
    categories: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """把 ``(id, score)`` 列表补成含 ``id/name/category/score`` 的结构。

    可传入预先构建好的 ``names`` / ``categories`` 字典以避免在循环里重复建表
    （Web 服务启动时构建一次即可）。
    """
    names = names if names is not None else build_name_map(products)
    categories = categories if categories is not None else build_category_map(products)

    results: list[dict[str, Any]] = []
    for raw_id, score in pairs:
        item_id = str(raw_id)
        results.append(
            {
                "id": item_id,
                "name": names.get(item_id, UNKNOWN_NAME),
                "category": categories.get(item_id, UNKNOWN_CATEGORY),
                "score": round(float(score), 6),
            }
        )
    return results
