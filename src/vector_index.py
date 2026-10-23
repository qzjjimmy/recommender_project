"""阶段三：商品向量索引。

训练模型负责"学会向量"，索引负责"快速找出像谁"。本项目采用**暴力余弦**方案：
把向量预先做 L2 归一化后存成矩阵，查询时用一次矩阵乘法算出与所有商品的余弦相似度，
再取 TopN。理由：

    - 商品规模只有数千个（词表 3186），暴力检索的延迟完全可接受；
    - 不引入 faiss / chromadb 等额外依赖，环境更容易复现；
    - 归一化后内积即余弦，与 Item2Vec 的 ``most_similar`` 口径完全一致。

若将来商品规模上升到百万级，可把 ``search`` 换成 faiss / chromadb，接口不变。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

METRIC = "cosine"
VECTORS_FILE = "vectors.npy"
ITEMS_FILE = "items.json"
META_FILE = "meta.json"


class ItemVectorIndex:
    """商品向量的近邻检索器。"""

    def __init__(self) -> None:
        self.item_ids_: list[str] = []
        self.item_index_: dict[str, int] = {}
        self.vectors_: np.ndarray | None = None  # 已做 L2 归一化，float32

    # ---------------------------------------------------------------- build
    def build(self, item_vectors: dict[str, Any]) -> "ItemVectorIndex":
        """输入 ``{item_id: vector}``，构建可 search 的索引。"""
        if not item_vectors:
            raise ValueError("item_vectors 为空，无法构建索引。")

        self.item_ids_ = [str(k) for k in item_vectors.keys()]
        matrix = np.asarray([item_vectors[k] for k in self.item_ids_], dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"向量矩阵维度不正确：{matrix.shape}")

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors_ = matrix / norms
        self.item_index_ = {item_id: i for i, item_id in enumerate(self.item_ids_)}
        return self

    # ----------------------------------------------------------- save/load
    def save(self, dir_path: Path) -> None:
        """把索引持久化到 ``models/vector_index/``，下次启动无需重算。"""
        if self.vectors_ is None:
            raise RuntimeError("索引尚未构建，无法保存。")

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(dir_path / VECTORS_FILE, self.vectors_)
        (dir_path / ITEMS_FILE).write_text(
            json.dumps(self.item_ids_, ensure_ascii=False), encoding="utf-8"
        )
        meta = {
            "count": len(self.item_ids_),
            "dim": int(self.vectors_.shape[1]),
            "metric": METRIC,
            "normalized": True,
        }
        (dir_path / META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, dir_path: Path) -> "ItemVectorIndex":
        """从磁盘恢复到可查询状态。"""
        dir_path = Path(dir_path)
        vectors_path = dir_path / VECTORS_FILE
        items_path = dir_path / ITEMS_FILE
        if not vectors_path.exists() or not items_path.exists():
            raise FileNotFoundError(
                f"未找到向量索引：{dir_path}\n"
                f"请先运行 scripts/04_build_vector_index.py 构建索引。"
            )

        self.vectors_ = np.load(vectors_path).astype(np.float32)
        self.item_ids_ = [str(x) for x in json.loads(items_path.read_text(encoding="utf-8"))]
        self.item_index_ = {item_id: i for i, item_id in enumerate(self.item_ids_)}
        return self

    # -------------------------------------------------------------- search
    def search(self, item_id: str, topn: int = 5) -> list[tuple[str, float]]:
        """返回相似商品 ``[(item_id, score), ...]``，降序，**不含自身**。

        商品不在索引中时抛出带说明的 ``KeyError``，由调用方决定如何提示，
        而不是静默返回空结果。
        """
        item_id = str(item_id)
        if item_id not in self.item_index_:
            raise KeyError(
                f"商品 {item_id!r} 不在向量索引中（索引共 {len(self.item_ids_)} 个商品）。"
            )
        if topn <= 0:
            raise ValueError(f"topn 必须为正整数，收到 {topn}")

        return self.search_by_vector(
            self.vectors_[self.item_index_[item_id]], topn, exclude=item_id
        )

    def search_by_vector(
        self,
        vector: Iterable[float],
        topn: int = 5,
        exclude: str | None = None,
    ) -> list[tuple[str, float]]:
        """用任意向量检索最相似的商品（阶段四的"用户向量 → 候选商品"直接复用）。"""
        query = np.asarray(vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        query = query / norm

        scores = self.vectors_ @ query
        if exclude is not None and exclude in self.item_index_:
            scores[self.item_index_[exclude]] = -np.inf

        k = min(topn, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []

        candidates = np.argpartition(-scores, k - 1)[:k]
        order = candidates[np.argsort(-scores[candidates], kind="stable")]
        return [(self.item_ids_[i], float(scores[i])) for i in order]

    # ------------------------------------------------------------ 辅助方法
    def __len__(self) -> int:
        return len(self.item_ids_)

    def contains(self, item_id: str) -> bool:
        return str(item_id) in self.item_index_

    @property
    def dim(self) -> int:
        return 0 if self.vectors_ is None else int(self.vectors_.shape[1])

    def get_vector(self, item_id: str) -> np.ndarray:
        """取某个商品的单位向量。"""
        item_id = str(item_id)
        if item_id not in self.item_index_:
            raise KeyError(f"商品 {item_id!r} 不在向量索引中。")
        return self.vectors_[self.item_index_[item_id]]
