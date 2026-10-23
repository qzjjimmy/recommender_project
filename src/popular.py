"""阶段三：热门基线（最强朴素对照）。

热门榜是最难打败的朴素基线：如果个性化方案连热门都赢不了，
就说明它没有提供额外价值。因此本项目所有对比都以热门为基准。
"""
from __future__ import annotations

import pandas as pd


class PopularRecommender:
    """按历史交互次数排序的全局热门推荐器。"""

    def __init__(self, exclude_purchased: bool = True) -> None:
        """``exclude_purchased=True`` 时推荐结果会剔除该用户已购商品。

        默认开启是为了与协同过滤做**同口径对比**（CF 也会剔除已购），
        否则热门榜会把用户买过的东西再推一遍，指标不可比。
        """
        self.exclude_purchased = exclude_purchased
        self.item_counts_: pd.Series = pd.Series(dtype="int64")
        self.ranking_: list[str] = []
        self.user_items_: dict[str, set[str]] = {}

    def fit(self, interactions: pd.DataFrame) -> "PopularRecommender":
        """统计历史交互最多的商品，并记录每个用户的已购集合。"""
        user = interactions["CustomerID"].astype(str).str.strip()
        item = interactions["StockCode"].astype(str).str.strip()

        pairs = pd.DataFrame({"user": user, "item": item}).drop_duplicates()
        self.item_counts_ = pairs["item"].value_counts()
        self.ranking_ = [str(x) for x in self.item_counts_.index]
        self.user_items_ = {
            str(u): set(g) for u, g in pairs.groupby("user")["item"].agg(set).items()
        }
        return self

    def recommend(self, user_id: str, k: int = 10) -> list[str]:
        """返回全局热门 TopK（默认剔除该用户已购商品）。"""
        purchased = self.user_items_.get(str(user_id), set()) if self.exclude_purchased else set()

        out: list[str] = []
        for item_id in self.ranking_:
            if item_id in purchased:
                continue
            out.append(item_id)
            if len(out) >= k:
                break
        return out

    def score(self, item_id: str) -> int:
        """某个商品的历史交互次数（不在榜上返回 0）。"""
        return int(self.item_counts_.get(str(item_id), 0))

    def top_items(self, k: int = 10) -> list[str]:
        """不剔除任何商品的纯全局热门榜。"""
        return self.ranking_[:k]

    def is_hot(self, item_id: str, top_n: int = 100) -> bool:
        """判断商品是否落在全局热门 TopN 内（用于统计召回集的热门占比）。"""
        return str(item_id) in set(self.ranking_[:top_n])
