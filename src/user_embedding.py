"""阶段四：用户向量（用户兴趣的商品向量聚合）。

把"用户历史买过什么"聚合成一个向量：取历史商品向量的**平均**，
再用它与商品向量做近邻检索，得到用户侧候选。

为什么要限制 ``recent_n``：
    阶段二实测发现少数批发买家（如用户 14096）单个购物篮平均近 300 件商品，
    其历史商品数远超普通用户。如果对全部历史取平均，向量会被"平均成没有方向"，
    个性化能力退化。只取最近 N 个商品可以缓解这个问题，因此本模块把
    ``recent_n`` 作为一等参数，而不是可选项。

历史为空时的兜底：``build_user_vector`` 返回 ``None``，由调用方决定降级策略
（本项目走热门兜底并在召回集里标记 ``cold_start=True``）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _lookup(item_vectors: Any, item_id: str) -> np.ndarray | None:
    """从 dict 或 ItemVectorIndex 中取一个商品向量；取不到返回 None。"""
    item_id = str(item_id)
    if hasattr(item_vectors, "get_vector"):  # ItemVectorIndex
        if not item_vectors.contains(item_id):
            return None
        return np.asarray(item_vectors.get_vector(item_id), dtype=np.float32)
    vector = item_vectors.get(item_id)
    return None if vector is None else np.asarray(vector, dtype=np.float32)


def build_user_vector(
    history_item_ids: Sequence[str],
    item_vectors: Any,
    recent_n: int | None = None,
) -> np.ndarray | None:
    """由历史商品向量聚合用户向量。

    参数：
        history_item_ids: 用户历史商品 ID，**需按时间从早到晚排列**。
        item_vectors:     ``{商品ID: 向量}`` 字典，或 ``ItemVectorIndex``。
        recent_n:         只取最近 N 个商品；``None`` 表示用全部历史。

    返回 L2 归一化后的用户向量（单位向量）；历史为空、或历史商品全部不在
    向量词表中时返回 ``None``（由调用方走热门兜底）。

    归一化只改变长度、不改变方向，目的是让下游的余弦检索可以统一处理。
    """
    if recent_n is not None and recent_n > 0:
        history_item_ids = list(history_item_ids)[-recent_n:]

    vectors = []
    for item_id in history_item_ids:
        vector = _lookup(item_vectors, item_id)
        if vector is not None:
            vectors.append(vector)
    if not vectors:
        return None

    mean = np.mean(np.vstack(vectors), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        return None
    return (mean / norm).astype(np.float32)


def recommend_by_user_vector(
    user_vec,
    item_vectors: Any,
    purchased: set[str],
    topk: int = 20,
) -> list[str]:
    """在商品向量中找与用户向量最相近的 TopK，并去掉已购商品。

    ``item_vectors`` 可以是 dict，也可以直接传 ``ItemVectorIndex``
    （后者走向量化的矩阵检索，快得多，推荐在批量场景使用）。
    """
    if user_vec is None:
        return []
    purchased = {str(x) for x in (purchased or set())}

    if hasattr(item_vectors, "search_by_vector"):
        total = len(item_vectors)
        want = min(total, topk + len(purchased))
        pairs = item_vectors.search_by_vector(user_vec, topn=want)
        return [item_id for item_id, _ in pairs if item_id not in purchased][:topk]

    # dict 路径：逐条算余弦
    query = np.asarray(user_vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(query))
    if norm == 0.0:
        return []
    query = query / norm

    scored: list[tuple[str, float]] = []
    for item_id, vector in item_vectors.items():
        item_id = str(item_id)
        if item_id in purchased:
            continue
        vec = np.asarray(vector, dtype=np.float32).ravel()
        vec_norm = float(np.linalg.norm(vec))
        if vec_norm == 0.0:
            continue
        scored.append((item_id, float(np.dot(vec / vec_norm, query))))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [item_id for item_id, _ in scored[:topk]]


def build_user_vectors(
    histories: Mapping[str, Sequence[str]],
    item_vectors: Any,
    recent_n: int | None = None,
) -> dict[str, np.ndarray]:
    """批量构建用户向量，只返回成功建出向量的用户。"""
    out: dict[str, np.ndarray] = {}
    for user_id, history in histories.items():
        vector = build_user_vector(history, item_vectors, recent_n=recent_n)
        if vector is not None:
            out[str(user_id)] = vector
    return out


def save_user_vectors(user_vectors: Mapping[str, np.ndarray], path: Path) -> None:
    """把用户向量存成 ``.npz``（含用户 ID 与矩阵两部分）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    user_ids = np.asarray([str(u) for u in user_vectors.keys()], dtype=object)
    matrix = (
        np.vstack([np.asarray(v, dtype=np.float32) for v in user_vectors.values()])
        if user_vectors
        else np.zeros((0, 0), dtype=np.float32)
    )
    np.savez_compressed(path, user_ids=user_ids, vectors=matrix)


def load_user_vectors(path: Path) -> dict[str, np.ndarray]:
    """读取 ``.npz`` 用户向量。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到用户向量文件：{path}\n请先运行 scripts/06_build_recall_set.py。"
        )
    with np.load(path, allow_pickle=True) as data:
        user_ids = [str(x) for x in data["user_ids"]]
        matrix = np.asarray(data["vectors"], dtype=np.float32)
    return {user_id: matrix[i] for i, user_id in enumerate(user_ids)}


def cosine_similarity(vec_a, vec_b) -> float:
    """两个用户/商品向量之间的余弦相似度（便于做案例对照）。"""
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def vector_dim(user_vec) -> int:
    """用户向量维度；``None`` 返回 0。"""
    return 0 if user_vec is None else int(np.asarray(user_vec).ravel().shape[0])


def dumps(user_vec) -> str:
    """把用户向量转成 JSON 字符串（用于写入日志，便于人工核对）。"""
    if user_vec is None:
        return "null"
    return json.dumps([round(float(x), 6) for x in np.asarray(user_vec).ravel()])
