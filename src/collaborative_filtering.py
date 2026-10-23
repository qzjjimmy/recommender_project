"""阶段三：协同过滤（本项目实现 ItemCF）。

ItemCF 的基本思想：**如果两个商品经常被同一批用户购买，则它们相似**。
流程为「共现 → 相似度 → 推荐」：

    1. 用二值化的用户-商品交互矩阵 R（买过=1，不关心买了几次）；
    2. 对 R 的列做 L2 归一化后计算 ``S = R_norm.T @ R_norm``，
       得到商品两两之间的**余弦相似度**；
    3. 对用户已购的每个商品，把它们的相似商品按相似度加权汇总，取 TopK。

相似度口径写死为余弦（写进 README 与本阶段文档），保证组间可比。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


class CollaborativeFiltering:
    """基于物品的协同过滤（ItemCF）。"""

    def __init__(
        self,
        top_similar: int = 0,
        exclude_purchased: bool = True,
        min_score: float = 0.0,
    ) -> None:
        """参数：

        top_similar: 每个商品只保留最相似的 N 个（0 表示不截断）。
                     截断可以降噪并加快打分，是 ItemCF 的常规做法。
        exclude_purchased: 推荐结果是否剔除用户已购商品（默认剔除）。
        min_score: 相似度加权得分低于该阈值的结果不返回（默认 0，
                   即"与已购商品没有任何共现"的商品不推荐）。
        """
        self.top_similar = top_similar
        self.exclude_purchased = exclude_purchased
        self.min_score = min_score

        self.item_ids_: list[str] = []
        self.item_index_: dict[str, int] = {}
        self.user_items_: dict[str, list[int]] = {}
        self.similarity_: sparse.csr_matrix | None = None
        self.n_users_ = 0
        self.n_items_ = 0

    # ------------------------------------------------------------------ fit
    def fit(self, interactions: pd.DataFrame) -> "CollaborativeFiltering":
        """构建用户-商品交互矩阵，并计算商品间的余弦相似度。"""
        user = interactions["CustomerID"].astype(str).str.strip()
        item = interactions["StockCode"].astype(str).str.strip()
        pairs = pd.DataFrame({"user": user, "item": item}).drop_duplicates()

        self.item_ids_ = [str(x) for x in sorted(pairs["item"].unique())]
        self.item_index_ = {item_id: i for i, item_id in enumerate(self.item_ids_)}
        users = pd.Index(sorted(pairs["user"].unique()))

        rows = users.get_indexer(pairs["user"])
        cols = pairs["item"].map(self.item_index_).to_numpy()
        self.n_users_, self.n_items_ = len(users), len(self.item_ids_)

        matrix = sparse.csr_matrix(
            (np.ones(len(pairs), dtype=np.float32), (rows, cols)),
            shape=(self.n_users_, self.n_items_),
        )
        matrix.sum_duplicates()
        matrix.data[:] = 1.0  # 二值化：买过即为 1，买几次不影响相似度

        # 列（商品）L2 归一化后做内积 = 余弦相似度
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel()).astype(np.float32)
        norms[norms == 0] = 1.0
        # 一维数组按列广播：第 j 列整体乘以 1/norms[j]
        normalized = matrix.multiply(1.0 / norms).tocsr()

        similarity = (normalized.T @ normalized).tocsr().astype(np.float32)
        similarity.setdiag(0.0)  # 去掉自身，保证推荐结果不含自己
        similarity.eliminate_zeros()
        if self.top_similar > 0:
            similarity = self._truncate_rows(similarity, self.top_similar)
        self.similarity_ = similarity

        # 每个用户已购商品的列下标
        item_positions = pd.Series(cols, index=pairs["user"].to_numpy())
        self.user_items_ = {
            str(u): [int(x) for x in group.to_numpy()]
            for u, group in item_positions.groupby(level=0)
        }
        return self

    @staticmethod
    def _truncate_rows(matrix: sparse.csr_matrix, top_k: int) -> sparse.csr_matrix:
        """每个商品只保留相似度最高的 top_k 个邻居。"""
        matrix = matrix.tocsr()
        if matrix.nnz == 0:
            return matrix

        indptr = [0]
        indices: list[np.ndarray] = []
        data: list[np.ndarray] = []
        for i in range(matrix.shape[0]):
            start, end = matrix.indptr[i], matrix.indptr[i + 1]
            row_index = matrix.indices[start:end]
            row_data = matrix.data[start:end]
            if len(row_data) > top_k:
                keep = np.argpartition(-row_data, top_k)[:top_k]
                row_index, row_data = row_index[keep], row_data[keep]
            indices.append(row_index)
            data.append(row_data)
            indptr.append(indptr[-1] + len(row_index))

        return sparse.csr_matrix(
            (np.concatenate(data), np.concatenate(indices), np.asarray(indptr)),
            shape=matrix.shape,
        )

    # ------------------------------------------------------------ recommend
    def recommend(self, user_id: str, k: int = 10) -> list[str]:
        """返回该用户 TopK 商品（默认排除已购）。

        打分方式：候选商品 i 的得分 = 该用户已购商品与 i 的相似度之和。
        得分 <= 0 表示与已购商品没有任何共现证据，不予推荐。
        """
        positions = self.user_items_.get(str(user_id))
        if not positions:
            return []

        scores = np.asarray(self.similarity_[positions].sum(axis=0)).ravel().astype(np.float64)
        if self.exclude_purchased:
            scores[positions] = -np.inf

        out: list[str] = []
        for i in np.argsort(-scores, kind="stable"):
            score = scores[i]
            if not np.isfinite(score) or score <= self.min_score:
                break
            out.append(self.item_ids_[i])
            if len(out) >= k:
                break
        return out

    def similar_items(self, item_id: str, topn: int = 10) -> list[tuple[str, float]]:
        """ItemCF 口径下的相似商品，用于与 Item2Vec 的相似结果对照。"""
        item_id = str(item_id)
        if item_id not in self.item_index_:
            raise KeyError(f"商品 {item_id!r} 不在 ItemCF 的交互矩阵中。")

        row = self.similarity_[self.item_index_[item_id]]
        indices = row.indices
        values = row.data
        if len(values) == 0:
            return []

        order = np.argsort(-values, kind="stable")[:topn]
        return [(self.item_ids_[indices[i]], float(values[i])) for i in order]

    def known_user(self, user_id: str) -> bool:
        """该用户是否出现在训练交互中（用于统计冷启动用户比例）。"""
        return str(user_id) in self.user_items_
