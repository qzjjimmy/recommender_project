"""阶段五：防泄漏特征与样本构造。

时间口径（本项目全流程统一）：

    ├─────────────── 训练期 (80%) ───────────────┼── 验证期 (20%) ──┤
    │  特征期 (70%)          │   标签期 (30%)     │                   │
    │  ← 只用来算特征 →       │  ← 只用来取标签 →  │  ← 只用来评测 →   │

关键设计 —— **固定长度的回看窗口**：
    特征只能用"预测时刻之前、lookback_days 天以内"的数据统计。
    训练时预测时刻 = 标签期起点，评测时预测时刻 = 训练/验证切分点。
    两个窗口等长（默认 180 天），因此特征的分布口径可比，
    既避免了"用标签期信息算特征"的泄漏，也避免了特征漂移。

如果改用"训练期全量统计 + 随机切分"，就会出现任务书明令禁止的泄漏：
先用全量数据统计用户/商品均值，再切分训练测试。
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src import user_embedding as ue

UNKNOWN_CATEGORY = "未知类目"

USER_FEATURES = ["u_events", "u_items", "u_invoices", "u_avg_price", "u_recency_days"]
ITEM_FEATURES = ["i_users", "i_events", "i_price", "i_hot"]
MATCH_FEATURES = ["m_vec_sim", "m_cat_share", "m_cat_match", "m_price_ratio"]
FEATURE_COLS = USER_FEATURES + ITEM_FEATURES + MATCH_FEATURES


# ------------------------------------------------------------ 时间切分（阶段三起共用）
def choose_split_time(df: pd.DataFrame, time_col: str, valid_ratio: float = 0.2):
    """按比例选出切分时间点：让验证期大约占 ``valid_ratio`` 的样本。"""
    time = pd.to_datetime(df[time_col], errors="coerce")
    if time.notna().sum() == 0:
        raise ValueError(f"列 {time_col!r} 中没有可解析的时间。")
    ratio = min(max(float(valid_ratio), 0.0), 1.0)
    return time.quantile(1.0 - ratio)


def time_split(
    df: pd.DataFrame,
    time_col: str,
    split_time=None,
    valid_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间切分训练期/验证期，避免用未来信息。

    训练期 = ``时间 < split_time``；验证期 = ``时间 >= split_time``。
    时间无法解析的行归入训练期（不能作为"未来真实行为"参与评测）。
    """
    time = pd.to_datetime(df[time_col], errors="coerce")
    if split_time is None:
        split_time = choose_split_time(df, time_col, valid_ratio)
    split_time = pd.Timestamp(split_time)

    train_mask = time.isna() | (time < split_time)
    valid_mask = (~time.isna()) & (time >= split_time)
    return df[train_mask].copy(), df[valid_mask].copy()


def prepare_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """统一交互表的类型：时间解析、ID 转字符串去空格、单价格式化。"""
    out = df.copy()
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")
    out["CustomerID"] = out["CustomerID"].astype(str).str.strip()
    out["StockCode"] = out["StockCode"].astype(str).str.strip()
    out["UnitPrice"] = pd.to_numeric(out["UnitPrice"], errors="coerce")
    return out


# ------------------------------------------------------------------ 特征引擎
class FeatureEngine:
    """持有"截止某一时刻已知的"统计量，为 (用户, 候选商品) 生成特征。

    训练与评测共用同一个类，只是 ``reference_time`` 不同 ——
    这保证两边算出来的特征含义完全一致，不会出现"训练用一套、上线用另一套"。

    三类特征：
        用户特征 u_*  : 活跃度、消费水平、最近活跃间隔
        商品特征 i_*  : 热度、价格、是否全局热门
        匹配特征 m_*  : 用户向量与商品向量的余弦、类目偏好、价格比
    """

    def __init__(
        self,
        *,
        reference_time: pd.Timestamp,
        lookback_days: int,
        user_stats: pd.DataFrame,
        item_stats: pd.DataFrame,
        user_vectors: Mapping[str, np.ndarray],
        user_category_share: Mapping[str, Mapping[str, float]],
        item_category: Mapping[str, str],
        item_index,
        hot_items: set[str],
        histories: Mapping[str, Sequence[str]] | None = None,
        feature_cols: Sequence[str] = FEATURE_COLS,
    ) -> None:
        self.reference_time = pd.Timestamp(reference_time)
        self.lookback_days = int(lookback_days)
        self.user_stats = user_stats
        self.item_stats = item_stats
        # 转成普通字典，避免在十万级样本上反复用 DataFrame.loc 查表
        self.user_stats_map: dict[str, dict[str, float]] = user_stats.to_dict("index")
        self.item_stats_map: dict[str, dict[str, float]] = item_stats.to_dict("index")
        self.item_vector_map: dict[str, np.ndarray] = (
            {item: item_index.get_vector(item) for item in item_index.item_ids_}
            if item_index is not None
            else {}
        )
        self.user_vectors = dict(user_vectors)
        self.user_category_share = {k: dict(v) for k, v in user_category_share.items()}
        self.item_category = dict(item_category)
        self.item_index = item_index
        self.hot_items = set(hot_items)
        self.histories: dict[str, list[str]] = {
            str(k): [str(x) for x in v] for k, v in (histories or {}).items()
        }
        self.feature_cols = list(feature_cols)

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        interactions: pd.DataFrame,
        *,
        reference_time,
        lookback_days: int = 180,
        item_index,
        item_category: Mapping[str, str],
        recent_n: int | None = 50,
        hot_top_n: int = 100,
    ) -> "FeatureEngine":
        """用 ``reference_time`` 之前 lookback_days 天内的数据统计特征。"""
        reference_time = pd.Timestamp(reference_time)
        df = prepare_interactions(interactions)
        start = reference_time - timedelta(days=int(lookback_days))
        # 上界取开区间：reference_time 当刻及之后的数据属于"预测目标期"，绝不能进特征
        window = df[(df["InvoiceDate"] >= start) & (df["InvoiceDate"] < reference_time)]
        if window.empty:
            raise ValueError(
                f"回看窗口 {start} ~ {reference_time} 内没有交互数据，无法统计特征。"
            )

        # ---- 用户特征 ----
        grouped = window.groupby("CustomerID")
        user_stats = pd.DataFrame(
            {
                "u_events": grouped.size(),
                "u_items": grouped["StockCode"].nunique(),
                "u_invoices": grouped["InvoiceNo"].nunique(),
                "u_avg_price": grouped["UnitPrice"].mean(),
                "_last": grouped["InvoiceDate"].max(),
            }
        )
        user_stats["u_recency_days"] = (
            reference_time - user_stats["_last"]
        ).dt.total_seconds() / 86400.0
        user_stats = user_stats.drop(columns=["_last"])

        # ---- 商品特征 ----
        grouped_item = window.groupby("StockCode")
        item_stats = pd.DataFrame(
            {
                "i_events": grouped_item.size(),
                "i_users": grouped_item["CustomerID"].nunique(),
                "i_price": grouped_item["UnitPrice"].mean(),
            }
        )
        hot_items = set(
            item_stats["i_users"].sort_values(ascending=False).head(hot_top_n).index.astype(str)
        )
        item_stats["i_hot"] = item_stats.index.astype(str).isin(hot_items).astype(int)

        # ---- 用户类目偏好 ----
        window = window.assign(
            _cat=window["StockCode"].map(item_category).fillna(UNKNOWN_CATEGORY)
        )
        cat_counts = window.groupby(["CustomerID", "_cat"]).size().rename("n").reset_index()
        cat_counts["share"] = cat_counts["n"] / cat_counts.groupby("CustomerID")["n"].transform("sum")
        user_category_share = {
            str(user): dict(zip(sub["_cat"], sub["share"].astype(float)))
            for user, sub in cat_counts.groupby("CustomerID")
        }

        # ---- 用户历史 → 用户向量（同一窗口、同一 recent_n，训练与评测口径一致）----
        ordered = window.sort_values("InvoiceDate", kind="stable")
        histories = {
            str(u): [str(x) for x in items]
            for u, items in ordered.groupby("CustomerID")["StockCode"].agg(list).items()
        }
        user_vectors = ue.build_user_vectors(histories, item_index, recent_n=recent_n)

        return cls(
            reference_time=reference_time,
            lookback_days=lookback_days,
            user_stats=user_stats,
            item_stats=item_stats,
            user_vectors=user_vectors,
            user_category_share=user_category_share,
            item_category=item_category,
            item_index=item_index,
            hot_items=hot_items,
            histories=histories,
        )

    # ---------------------------------------------------------------- build
    def build(self, user_id: str, candidates: Iterable[str]) -> pd.DataFrame:
        """为某个用户的一批候选商品生成特征（每个候选一行）。"""
        user_id = str(user_id)
        user_row = self.user_stats_map.get(user_id)
        user_vec = self.user_vectors.get(user_id)
        user_cats = self.user_category_share.get(user_id, {})

        def as_float(value, default: float = 0.0) -> float:
            if value is None:
                return default
            try:
                number = float(value)
            except (TypeError, ValueError):
                return default
            return default if np.isnan(number) else number

        u_events = as_float(user_row["u_events"]) if user_row is not None else 0.0
        u_items = as_float(user_row["u_items"]) if user_row is not None else 0.0
        u_invoices = as_float(user_row["u_invoices"]) if user_row is not None else 0.0
        u_avg_price = as_float(user_row["u_avg_price"]) if user_row is not None else 0.0
        u_recency = (
            as_float(user_row["u_recency_days"], float(self.lookback_days))
            if user_row is not None
            else float(self.lookback_days)
        )

        rows: list[dict[str, Any]] = []
        for raw_item in candidates:
            item_id = str(raw_item)
            item_row = self.item_stats_map.get(item_id)
            category = self.item_category.get(item_id, UNKNOWN_CATEGORY)

            item_vec = self.item_vector_map.get(item_id)
            if user_vec is not None and item_vec is not None:
                vec_sim = float(np.dot(user_vec, item_vec))
            else:
                vec_sim = 0.0

            i_price = as_float(item_row["i_price"]) if item_row is not None else 0.0
            price_ratio = (i_price / u_avg_price) if (u_avg_price > 0 and i_price > 0) else 1.0

            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "u_events": u_events,
                    "u_items": u_items,
                    "u_invoices": u_invoices,
                    "u_avg_price": u_avg_price,
                    "u_recency_days": u_recency,
                    "i_users": as_float(item_row["i_users"]) if item_row is not None else 0.0,
                    "i_events": as_float(item_row["i_events"]) if item_row is not None else 0.0,
                    "i_price": i_price,
                    "i_hot": as_float(item_row["i_hot"]) if item_row is not None else 0.0,
                    "m_vec_sim": vec_sim,
                    "m_cat_share": float(user_cats.get(category, 0.0)),
                    "m_cat_match": 1.0 if category in user_cats else 0.0,
                    "m_price_ratio": float(min(price_ratio, 10.0)),
                }
            )
        return pd.DataFrame(rows, columns=["user_id", "item_id"] + list(self.feature_cols))

    # ---------------------------------------------------------------- 便捷方法
    def build_many(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """按 ``user_id`` 分组的候选表批量生成特征（比逐次调用 build 快）。"""
        frames = [
            self.build(user_id, group["item_id"].tolist())
            for user_id, group in pairs.groupby("user_id", sort=False)
        ]
        if not frames:
            return pd.DataFrame(columns=["user_id", "item_id"] + list(self.feature_cols))
        return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ 样本构造
def build_samples_and_features(
    candidates: pd.DataFrame,
    positives: set[tuple[str, str]],
    engine: FeatureEngine,
    label_col: str = "label",
) -> pd.DataFrame:
    """在候选 (用户, 商品) 上构造特征并打标签。

    参数：
        candidates: 含 ``user_id`` / ``item_id`` 两列的候选表（正负样本混在一起）
        positives:  正样本集合，元素为 ``(user_id, item_id)``
        engine:     已 fit 的 :class:`FeatureEngine`
        label_col:  标签列名
    """
    out = engine.build_many(candidates)
    out[label_col] = [
        1 if (str(u), str(i)) in positives else 0
        for u, i in zip(out["user_id"], out["item_id"])
    ]
    return out


class NegativeSampler:
    """按热度加权从候选池里采样负样本（排除用户已交互过的商品）。

    为什么按热度加权，而不是均匀采样：
        均匀采样会抽出大量冷门商品，模型只要学会"热门 = 正样本"就能刷出很高 AUC，
        指标虚高却不代表真实能力。按热度加权后负样本也集中在头部商品，
        分布更接近真实召回候选，模型必须靠**用户-商品匹配度**才能区分。

    候选池与权重在构造时预处理一次，避免在批量采样时反复重建。
    """

    def __init__(self, item_pool: Sequence[str], weights: np.ndarray, rng: np.random.Generator) -> None:
        self.pool = np.asarray([str(x) for x in item_pool])
        self.weights = np.clip(np.asarray(weights, dtype=np.float64), 1e-9, None)
        self.index = {item: idx for idx, item in enumerate(self.pool)}
        self.rng = rng

    def sample(self, user_id: str, n: int, exclude: Iterable[str]) -> list[tuple[str, str]]:
        """给一个用户采 n 个负样本。"""
        if n <= 0 or len(self.pool) == 0:
            return []

        keep = np.ones(len(self.pool), dtype=bool)
        positions = [
            self.index[str(item)] for item in exclude if str(item) in self.index
        ]
        if positions:
            keep[np.asarray(positions, dtype=np.int64)] = False
        if keep.sum() == 0:
            return []

        candidates = self.pool[keep]
        probs = self.weights[keep]
        probs = probs / probs.sum()

        size = min(n, len(candidates))
        picked = self.rng.choice(candidates, size=size, replace=False, p=probs)
        return [(str(user_id), str(item)) for item in picked]

    def sample_many(
        self, user_counts: Mapping[str, int], user_exclude: Mapping[str, Iterable[str]]
    ) -> list[tuple[str, str]]:
        """按 ``{用户: 采样条数}`` 批量采样。"""
        out: list[tuple[str, str]] = []
        for user_id, n in user_counts.items():
            out.extend(self.sample(user_id, n, user_exclude.get(str(user_id), set())))
        return out
