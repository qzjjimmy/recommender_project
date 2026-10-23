"""用户购买历史：把干净交易表压成「客户 → 买过什么」的查询表。

推荐首页的板块一（"猜你还会买"）要基于**用户购买过的商品**做相似扩展，
因此需要按客户查历史。干净交易表有 57 万行、51 MB，每次请求都扫一遍不现实，
所以这里做两件事：

    1. :func:`build_customer_history` 把交易表**按 (客户, 商品) 聚合**成一行
       —— 顺便把"买过几次、一共几件、最后一次什么时候买"一起算好；
    2. :func:`load_purchase_history` 把结果缓存到
       ``data/processed/customer_history.csv``，Web 服务只加载这份小表。

关于"次数"的口径：同一张发票里出现两次算同一个订单（``orders`` 按发票去重计数），
``qty`` 才是累计件数。首页的权重用 ``log1p(orders)`` —— 回购次数多的商品更能
代表用户口味，但不想让批发买家的大单量一边倒。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from config.settings import PATHS

HISTORY_COLUMNS = ["customer_id", "item_id", "orders", "qty", "first_date", "last_date"]


class HistoryItem(NamedTuple):
    """一条历史购买记录：商品 ID、订单数、件数、首末购买时间。"""

    item_id: str
    orders: int
    qty: float
    first_date: str
    last_date: str


def build_customer_history(clean_df: pd.DataFrame) -> pd.DataFrame:
    """按 (客户, 商品) 聚合交易明细。

    ``orders`` 用发票号去重计数：同一张发票里出现两次的商品只算一次购买。
    """
    df = clean_df.copy()
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["_dt"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df[(df["CustomerID"] != "") & (df["StockCode"] != "") & df["_dt"].notna()]

    grouped = df.groupby(["CustomerID", "StockCode"])
    history = pd.DataFrame(
        {
            "orders": grouped["InvoiceNo"].nunique(),
            "qty": grouped["Quantity"].sum(),
            "first_date": grouped["_dt"].min(),
            "last_date": grouped["_dt"].max(),
        }
    ).reset_index()
    history = history.rename(columns={"CustomerID": "customer_id", "StockCode": "item_id"})
    history["first_date"] = history["first_date"].dt.strftime("%Y-%m-%d %H:%M")
    history["last_date"] = history["last_date"].dt.strftime("%Y-%m-%d %H:%M")
    history["qty"] = history["qty"].astype(float).round(2)
    history["orders"] = history["orders"].astype(int)
    return history[HISTORY_COLUMNS].sort_values(
        ["customer_id", "last_date"], ascending=[True, False]
    ).reset_index(drop=True)


class PurchaseHistory:
    """``客户 ID → 历史购买记录`` 的只读查询视图。"""

    def __init__(self, history: pd.DataFrame) -> None:
        self.history = history
        self._items: dict[str, list[HistoryItem]] = {}
        for row in history.itertuples(index=False):
            self._items.setdefault(str(row.customer_id), []).append(
                HistoryItem(
                    item_id=str(row.item_id),
                    orders=int(row.orders),
                    qty=float(row.qty),
                    first_date=str(row.first_date),
                    last_date=str(row.last_date),
                )
            )
        # 每个客户内部按"最后一次购买时间"倒序，最近的排最前
        for items in self._items.values():
            items.sort(key=lambda i: i.last_date, reverse=True)

    # ---------------------------------------------------------------- 查询
    def __len__(self) -> int:
        """有多少位有购买记录的客户。"""
        return len(self._items)

    @property
    def customers(self) -> list[str]:
        return list(self._items.keys())

    def has(self, customer_id: str) -> bool:
        return str(customer_id or "") in self._items

    def items(self, customer_id: str, limit: int | None = None) -> list[HistoryItem]:
        """某位客户的历史购买记录（最近的在前）。"""
        items = self._items.get(str(customer_id or ""), [])
        return list(items[:limit]) if limit else list(items)

    def item_ids(self, customer_id: str) -> set[str]:
        """某位客户买过的**全部**商品 ID（用于把已购商品排除掉）。"""
        return {i.item_id for i in self._items.get(str(customer_id or ""), [])}

    def count(self, customer_id: str) -> int:
        return len(self._items.get(str(customer_id or ""), []))

    def stats(self) -> dict[str, Any]:
        sizes = [len(v) for v in self._items.values()]
        if not sizes:
            return {"customers": 0, "pairs": 0, "median_items": 0, "max_items": 0}
        series = pd.Series(sizes)
        return {
            "customers": len(sizes),
            "pairs": int(self.history.shape[0]),
            "median_items": int(series.median()),
            "max_items": int(series.max()),
        }


def load_purchase_history(path: Path | None = None, force: bool = False) -> PurchaseHistory:
    """读取缓存的购买历史；不存在（或 ``force``）时重建并落盘。"""
    path = Path(path or PATHS["customer_history"])
    if path.exists() and not force:
        history = pd.read_csv(path, dtype={"customer_id": str, "item_id": str})
    else:
        clean = pd.read_csv(PATHS["clean_transactions"], dtype=str)
        history = build_customer_history(clean)
        path.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(path, index=False, encoding="utf-8")
    return PurchaseHistory(history)
