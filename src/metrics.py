"""阶段三/六：推荐效果指标。

评测口径统一为：**先截断到 K，再与真实集合求交**。
Precision@K 的分母固定取 K（推荐列表不足 K 个时按 K 计，惩罚短列表），
这是推荐评测的通行做法，避免"只推 1 个且命中"就得到 1.0 的虚高。
"""
from __future__ import annotations

from typing import Iterable, Sequence, Set


def _top_k(recommended: Sequence[str] | Iterable[str], k: int) -> list[str]:
    """取推荐列表前 K 个（先截断再求交）。"""
    if k <= 0:
        raise ValueError(f"k 必须为正整数，收到 {k}")
    return list(recommended)[:k]


def precision_at_k(recommended: list[str], ground_truth: Set[str], k: int) -> float:
    """推荐列表前 K 个命中真实集合的比例（分母固定为 K）。"""
    top = _top_k(recommended, k)
    hits = sum(1 for item in top if item in ground_truth)
    return hits / k


def recall_at_k(recommended: list[str], ground_truth: Set[str], k: int) -> float:
    """前 K 个推荐覆盖了多少真实喜欢的商品（分母为真实集合大小）。"""
    if not ground_truth:
        return 0.0
    top = _top_k(recommended, k)
    hits = sum(1 for item in top if item in ground_truth)
    return hits / len(ground_truth)


def f1_at_k(recommended: list[str], ground_truth: Set[str], k: int) -> float:
    """Precision@K 与 Recall@K 的调和平均。"""
    p = precision_at_k(recommended, ground_truth, k)
    r = recall_at_k(recommended, ground_truth, k)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def overlap_count(a: Iterable[str], b: Iterable[str]) -> int:
    """两路推荐列表的重叠个数（集合语义，重复项只计一次）。"""
    return len(set(a) & set(b))


def overlap_rate(a: Iterable[str], b: Iterable[str], k: int) -> float:
    """两路推荐列表的重叠率（重叠个数 / K）。"""
    if k <= 0:
        raise ValueError(f"k 必须为正整数，收到 {k}")
    return overlap_count(list(a)[:k], list(b)[:k]) / k


def hit_rate_at_k(recommended: list[str], ground_truth: Set[str], k: int) -> float:
    """前 K 个里是否至少命中一个（0/1）。"""
    return 1.0 if any(item in ground_truth for item in _top_k(recommended, k)) else 0.0


def mean(values: Iterable[float]) -> float:
    """算术平均，空集合返回 0.0。"""
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
