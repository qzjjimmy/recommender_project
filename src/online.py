"""阶段五：在线排序链路（召回集 → 特征 → 精排 → TopN）。

模拟真实请求：**先有候选，再精排**。

    读取该用户的召回集候选  →  为「用户 × 每个候选」构造特征
    →  用指定模型打分  →  按分数排序截断为 TopN

接口接受 ``feature_builder``（一个 ``(user_id, candidates) -> DataFrame`` 的可调用对象），
通常是 ``FeatureEngine.build``，因此离线评测与在线服务共用同一套特征逻辑。
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import pandas as pd


def score_candidates(
    user_id: str,
    candidates: Iterable[str],
    model_bundle: dict[str, Any],
    feature_builder: Callable[[str, list[str]], pd.DataFrame],
) -> pd.DataFrame:
    """给候选打分并附加分数列，返回按分数降序的明细（不截断）。"""
    candidate_list = [str(x) for x in candidates]
    if not candidate_list:
        return pd.DataFrame(columns=["user_id", "item_id", "score"])

    features = feature_builder(str(user_id), candidate_list)
    if features.empty:
        return pd.DataFrame(columns=["user_id", "item_id", "score"])

    scores = _predict(model_bundle, features)
    out = features[["user_id", "item_id"]].copy()
    out["score"] = scores
    return out.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)


def _predict(model_bundle: dict[str, Any], features: pd.DataFrame) -> list[float]:
    """按模型类型选择打分函数（LR 需要 scaler，树模型不需要）。"""
    if model_bundle.get("kind") == "lgbm":
        from src import ranker_lgbm

        return ranker_lgbm.predict_scores(model_bundle, features)
    from src import ranker_lr

    return ranker_lr.predict_scores(model_bundle, features)


def rank_for_user(
    user_id: str,
    candidates: list[str],
    model_bundle: dict[str, Any],
    feature_builder: Callable[[str, list[str]], pd.DataFrame],
    topn: int = 10,
) -> list[str]:
    """对同一用户的召回候选打分排序并截断为 TopN，返回商品 ID 列表。"""
    if topn <= 0:
        raise ValueError(f"topn 必须为正整数，收到 {topn}")
    scored = score_candidates(user_id, candidates, model_bundle, feature_builder)
    if scored.empty:
        return []
    return scored["item_id"].head(topn).tolist()


def rank_with_details(
    user_id: str,
    candidates: list[str],
    model_bundle: dict[str, Any],
    feature_builder: Callable[[str, list[str]], pd.DataFrame],
    topn: int = 10,
    name_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """与 :func:`rank_for_user` 同样的排序，但返回带分数与名称的明细。

    用于在线接口返回、演示与失败案例排查。
    """
    if topn <= 0:
        raise ValueError(f"topn 必须为正整数，收到 {topn}")
    scored = score_candidates(user_id, candidates, model_bundle, feature_builder)
    if scored.empty:
        return []

    details: list[dict[str, Any]] = []
    for rank, row in enumerate(scored.head(topn).itertuples(index=False), 1):
        details.append(
            {
                "rank": rank,
                "item_id": row.item_id,
                "name": (name_map or {}).get(row.item_id, ""),
                "score": round(float(row.score), 6),
            }
        )
    return details


def compare_two_models(
    user_id: str,
    candidates: list[str],
    bundle_a: dict[str, Any],
    bundle_b: dict[str, Any],
    feature_builder: Callable[[str, list[str]], pd.DataFrame],
    topn: int = 10,
) -> dict[str, Any]:
    """同用户、同候选池，用两个模型各排一次，返回两路 TopN 与重叠情况。

    这是"换模型不换候选"的公平对比：候选池完全相同，只换排序模型。
    """
    list_a = rank_for_user(user_id, candidates, bundle_a, feature_builder, topn)
    list_b = rank_for_user(user_id, candidates, bundle_b, feature_builder, topn)
    overlap = len(set(list_a) & set(list_b))
    return {
        "user_id": str(user_id),
        "list_a": list_a,
        "list_b": list_b,
        "overlap": overlap,
        "overlap_rate": overlap / topn if topn else 0.0,
    }
