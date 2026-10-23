"""智能推荐：把一句中文查询变成一份带理由的个性化推荐列表。

流程（与前面阶段的链路一致，只是把"召回"换成"条件过滤 + 热度召回"）：

    1. **查询理解**：解析出品类 / 价格区间 / 关键词（``query_parser``）
    2. **过滤**：在商品全集上按条件筛选候选
    3. **打分**：个性化相似度 + 商品热度 + 反馈偏好，三项加权
    4. **反馈**：剔除点踩过的商品，点赞过的商品提升同类候选
    5. **组织结果**：附上"为什么推荐这个"的理由

商品画像（名称 / 类目 / 均价 / 热度）从阶段一的干净交易表 + 商品主数据生成，
首次运行会缓存到 ``data/processed/item_profile.csv``。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import HYPERPARAMS, PATHS
from src import feedback as fb
from src import llm_parser
from src import product_catalog as catalog
from src import query_parser as qp
from src.vector_index import ItemVectorIndex

# 表格里用短标签（列窄），完整说明放在 reason_detail 里作为悬浮提示
REASON_LABELS = {
    "liked": "你点赞过",
    "feedback": "像你点赞的",
    "similarity": "符合你的偏好",
    "popularity": "热门",
    "balanced": "综合推荐",
    "no_vector": "暂无向量",
}
REASON_DETAIL = {
    "liked": "你点过「喜欢」的商品",
    "feedback": "与你点赞过的商品相似",
    "similarity": "与你历史购买的商品相似",
    "popularity": "该条件下的热门商品",
    "balanced": "综合热度与你的偏好推荐",
    "no_vector": "该商品暂无向量，按热度推荐",
}

# 规则解析不了、但大模型能处理的"主观/模糊"表达。
# 出现这类词时，即使用户也说了品类，也值得让大模型再理解一次 ——
# 例如"便宜"需要被换算成价格上限，规则做不到。
VAGUE_HINTS = (
    "便宜", "实惠", "划算", "性价比", "不贵", "低价", "平价",
    "高端", "高档", "高级", "精致", "精美", "好看", "漂亮", "有质感",
    "适合送", "送礼", "礼物", "纪念", "入门",
    "小", "大", "轻便", "可爱", "复古", "简约",
)


# ------------------------------------------------------------------ 商品画像
def build_item_profile(clean_df: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """从干净交易表 + 商品主数据生成商品画像。"""
    clean = clean_df.copy()
    clean["StockCode"] = clean["StockCode"].astype(str).str.strip()
    clean["CustomerID"] = clean["CustomerID"].astype(str).str.strip()
    clean["UnitPrice"] = pd.to_numeric(clean["UnitPrice"], errors="coerce")

    grouped = clean.groupby("StockCode")
    profile = pd.DataFrame(
        {
            "price": grouped["UnitPrice"].mean(),
            "users": grouped["CustomerID"].nunique(),
            "events": grouped.size(),
        }
    )

    prod = catalog.normalize_products(products)
    names = catalog.build_name_map(products)
    categories = catalog.build_category_map(products)

    profile["item_id"] = profile.index.astype(str)
    profile["name"] = profile["item_id"].map(names).fillna(catalog.UNKNOWN_NAME)
    profile["category"] = profile["item_id"].map(categories).fillna(catalog.UNKNOWN_CATEGORY)
    profile["price"] = profile["price"].fillna(0.0).round(2)
    profile["users"] = profile["users"].fillna(0).astype(int)
    profile["events"] = profile["events"].fillna(0).astype(int)

    profile = profile[profile["name"] != catalog.UNKNOWN_NAME]
    return profile[["item_id", "name", "category", "price", "users", "events"]].reset_index(drop=True)


def load_item_profile(path: Path | None = None, force: bool = False) -> pd.DataFrame:
    """读取缓存的商品画像；不存在（或 ``force``）时重建并落盘。"""
    path = Path(path or PATHS["item_profile"])
    if path.exists() and not force:
        return pd.read_csv(path, dtype={"item_id": str})

    clean = pd.read_csv(PATHS["clean_transactions"], dtype=str)
    products = catalog.load_products()
    profile = build_item_profile(clean, products)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(path, index=False, encoding="utf-8")
    return profile


# ------------------------------------------------------------------ 演示账号
def build_demo_customers(
    clean_df: pd.DataFrame,
    products: pd.DataFrame,
    top_categories: int = 6,
    min_items: int = 30,
) -> list[dict[str, Any]]:
    """挑出"偏好集中、有辨识度"的演示客户，供注册时选择。

    规则：对每个类目，取该类目占比最高（且商品数 >= min_items）的客户。
    这样注册页上每个账号都能用一句话说明"这个人喜欢什么"。
    """
    clean = clean_df.copy()
    clean["StockCode"] = clean["StockCode"].astype(str).str.strip()
    clean["CustomerID"] = clean["CustomerID"].astype(str).str.strip()
    categories = catalog.build_category_map(products)
    clean["_cat"] = clean["StockCode"].map(categories).fillna(catalog.UNKNOWN_CATEGORY)

    sizes = clean.groupby("CustomerID")["StockCode"].nunique()
    counts = clean.groupby(["CustomerID", "_cat"]).size().rename("n").reset_index()
    counts["share"] = counts["n"] / counts.groupby("CustomerID")["n"].transform("sum")

    # 只保留商品数够多的客户，且排除"其他/未知"这类无信息量的类目
    valid_users = set(sizes[sizes >= min_items].index)
    counts = counts[counts["CustomerID"].isin(valid_users)]
    counts = counts[~counts["_cat"].isin([catalog.UNKNOWN_CATEGORY, "其他"])]

    # 每个客户取占比最高的类目
    best = (
        counts.sort_values(["n", "share"], ascending=False)
        .drop_duplicates("CustomerID")
        .set_index("CustomerID")
    )
    best["n_items"] = sizes.reindex(best.index).fillna(0).astype(int)

    # 每个类目只留一个最有代表性的客户
    picked = (
        best.sort_values("share", ascending=False)
        .drop_duplicates("_cat")
        .sort_values("share", ascending=False)
        .head(top_categories)
    )

    out: list[dict[str, Any]] = []
    for customer_id, row in picked.iterrows():
        out.append(
            {
                "customer_id": str(customer_id),
                "n_items": int(row["n_items"]),
                "top_category": str(row["_cat"]),
                "category_share": round(float(row["share"]), 3),
                "label": f"{row['_cat']}偏好（历史 {int(row['n_items'])} 件商品，该类目占 {float(row['share']):.0%}）",
            }
        )
    return out


def load_demo_customers(force: bool = False) -> list[dict[str, Any]]:
    """读取缓存的演示客户列表；不存在时重建。"""
    path = Path(PATHS["demo_customers"])
    if path.exists() and not force:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    clean = pd.read_csv(PATHS["clean_transactions"], dtype=str)
    products = catalog.load_products()
    customers = build_demo_customers(clean, products)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(customers, ensure_ascii=False, indent=2), encoding="utf-8")
    return customers


# ------------------------------------------------------------------ 智能推荐
class SmartSearch:
    """带个性化与反馈闭环的查询式推荐器。"""

    def __init__(
        self,
        *,
        profile: pd.DataFrame,
        index: ItemVectorIndex | None,
        user_vectors: Mapping[str, np.ndarray],
        names: Mapping[str, str],
        weights: Mapping[str, float],
        feedback_weight: float,
        dislike_penalty: float,
        top_n: int,
        llm=None,
    ) -> None:
        self.profile = profile
        self.index = index
        self.user_vectors = dict(user_vectors)
        self.names = dict(names)
        self.weights = dict(weights)
        self.feedback_weight = float(feedback_weight)
        self.dislike_penalty = float(dislike_penalty)
        self.top_n = int(top_n)
        # 可选的大模型解析器：未配置时 parse_query 会自动只走规则
        self.llm = llm
        self.categories = sorted(set(profile["category"].astype(str)))
        # 预先归一化热度，便于与相似度加权
        max_users = float(profile["users"].max() or 1)
        self._pop_scale = np.log1p(max_users)
        # 给大模型的价格上下文：让它能把"便宜/高端"换算成本数据集里合理的数字
        prices = profile["price"].astype(float)
        prices = prices[prices > 0]
        self.price_context = (
            f"本数据集商品均价约 {prices.median():.1f} 元，"
            f"价格区间 {prices.min():.1f}~{prices.max():.0f} 元，"
            f"四分之一到四分之三的商品在 {prices.quantile(0.25):.1f}~{prices.quantile(0.75):.1f} 元之间。"
            "（判断「便宜」时上限可参考中位数附近）"
            if len(prices)
            else ""
        )
        # 关键词命中检查用的检索串（名称 + 商品ID，小写）
        self._haystack = (
            profile["name"].fillna("").astype(str) + " " + profile["item_id"].astype(str)
        ).str.lower()
        # 全库词表（上面检索串的拼接），交给规则解析丢掉"库里根本不存在的词"。
        # 只在构造时拼一次，避免每次查询都去扫 8000 行。
        self._vocabulary = " ".join(self._haystack.tolist())

    def keyword_hits(self, keyword: str) -> int:
        """某个关键词在商品库里能命中多少件（用于过滤前的合理性检查）。"""
        if not keyword:
            return 0
        return int(self._haystack.str.contains(str(keyword).lower(), regex=False).sum())

    # -------------------------------------------------------------- 加载
    @classmethod
    def load(cls, *, force_profile: bool = False) -> "SmartSearch":
        """从磁盘加载全部依赖（商品画像、向量索引、用户向量）。"""
        search_cfg = HYPERPARAMS.get("search", {})
        profile = load_item_profile(force=force_profile)

        index: ItemVectorIndex | None = None
        try:
            index = ItemVectorIndex().load(PATHS["vector_index_dir"])
        except FileNotFoundError:
            index = None  # 没有索引时退化为纯热度排序

        try:
            from src.user_embedding import load_user_vectors

            user_vectors = load_user_vectors(Path(PATHS["models_dir"]) / "user_vectors.npz")
        except FileNotFoundError:
            user_vectors = {}

        names = dict(zip(profile["item_id"], profile["name"]))
        return cls(
            profile=profile,
            index=index,
            user_vectors=user_vectors,
            names=names,
            weights=search_cfg.get("weights", {"similarity": 0.55, "popularity": 0.30, "feedback": 0.15}),
            feedback_weight=float(search_cfg.get("feedback_weight", 0.8)),
            dislike_penalty=float(search_cfg.get("dislike_penalty", 0.35)),
            top_n=int(search_cfg.get("top_n", 24)),
            llm=llm_parser.build_llm_parser(search_cfg.get("llm")),
        )

    # -------------------------------------------------------------- 过滤
    def _should_retry_with_llm(
        self, constraints: qp.QueryConstraints, filtered: pd.DataFrame, query: str
    ) -> bool:
        """规则结果是否"不理想"到值得请大模型再理解一次。

        两种情况值得重试：
            1. 候选太少（< 3 件）—— 多半是关键词匹配不上商品名，或条件写得太窄；
            2. 规则只解析出关键词、没有品类也没有价格 —— 说明没抓住用户的主要意图。
        """
        if not (query or "").strip():
            return False
        if len(filtered) < 3:
            return True
        # 出现了规则无法量化的主观词（便宜 / 高档 / 适合送人…）
        if constraints.max_price is None and constraints.min_price is None:
            if any(hint in query for hint in VAGUE_HINTS):
                return True
        return (
            not constraints.categories
            and constraints.min_price is None
            and constraints.max_price is None
        )

    def _constraints_effective(self, constraints: qp.QueryConstraints) -> bool:
        """规则解析出的条件是否**真的起到了过滤作用**。

        如果既没有类别、没有价格，关键词又全都在商品库里匹配不到，
        那结果就是"返回全部商品"—— 等于没理解用户意图，必须让大模型接管。
        """
        if constraints.categories or constraints.min_price is not None or constraints.max_price is not None:
            return True
        return any(self.keyword_hits(word) > 0 for word in constraints.keywords)

    @staticmethod
    def _llm_adds_constraint(rules: qp.QueryConstraints, llm: qp.QueryConstraints) -> bool:
        """大模型是否补充了规则没有识别出的类别或价格约束。"""
        if llm.categories and not rules.categories:
            return True
        if llm.max_price is not None and rules.max_price is None:
            return True
        if llm.min_price is not None and rules.min_price is None:
            return True
        return False

    @staticmethod
    def _llm_variants(
        rules: qp.QueryConstraints, llm: qp.QueryConstraints
    ) -> tuple[qp.QueryConstraints, qp.QueryConstraints]:
        """把"规则"与"大模型"两种解读组合成两套候选条件。

        · 方案一（合并）：价格**整对**以规则为准（正则读数字最稳），品类与关键词取并集；
        · 方案二（纯大模型）：品类与关键词完全采用大模型的解读，价格同样"整对"以规则为准
          （规则读不出价格时才用大模型的，避免"8000以内"被读成"≥8000 且 ≤8000"）。

        两套都算一遍过滤、谁的结果更合适就用谁 —— 手动模式取"第一套有结果的"，
        自动模式取"候选更多的那套"。
        """
        merged = qp.merge_constraints(rules, llm)
        # 价格同样"整对"以规则为准：规则读到任何价格线索时，方案二也用规则的价格。
        # 否则大模型把"8000以内"读成"=8000"时，方案二会直接变成空集。
        if rules.min_price is not None or rules.max_price is not None:
            price_min, price_max = rules.min_price, rules.max_price
        else:
            price_min, price_max = llm.min_price, llm.max_price
        llm_only = qp.QueryConstraints(
            raw=llm.raw,
            categories=list(llm.categories),
            min_price=price_min,
            max_price=price_max,
            keywords=list(llm.keywords),
            source="llm",
        )
        return merged, llm_only

    def _apply_constraints(
        self, constraints: qp.QueryConstraints
    ) -> tuple[pd.DataFrame, list[str]]:
        df = self.profile
        notes: list[str] = []

        if constraints.categories:
            before = len(df)
            df = df[df["category"].isin(constraints.categories)]
            notes.append(f"按品类筛选：{before} → {len(df)} 件")

        if constraints.max_price is not None:
            before = len(df)
            df = df[df["price"] <= float(constraints.max_price)]
            notes.append(f"按价格上限 {constraints.max_price:g} 筛选：{before} → {len(df)} 件")

        if constraints.min_price is not None:
            before = len(df)
            df = df[df["price"] >= float(constraints.min_price)]
            notes.append(f"按价格下限 {constraints.min_price:g} 筛选：{before} → {len(df)} 件")

        if constraints.keywords:
            # 先剔除"在**当前候选集**里一个都匹配不到"的关键词。
            # 两类词会被剔除：
            #   · 主观评价词（"礼物""好看"），全库就没有商品叫这个名字；
            #   · 只在别的品类里出现的词 —— 必须在**已经过品类筛选的子集**上判断，
            #     否则会出现"全库有命中、子集里 0 命中"，把结果直接清空。
            usable, dropped = [], []
            subset_text = (
                df["name"].fillna("").astype(str) + " " + df["item_id"].astype(str)
            ).str.lower()
            for word in constraints.keywords:
                hit_in_subset = bool(subset_text.str.contains(str(word).lower(), regex=False).any())
                (usable if hit_in_subset else dropped).append(word)
            if dropped:
                notes.append(
                    "关键词 " + "、".join(dropped) + " 在当前候选里匹配不到，已忽略（不影响其余条件）"
                )
            if usable:
                before = len(df)
                mask = [
                    qp.match_keywords(str(name), str(item), usable)
                    for name, item in zip(df["name"], df["item_id"])
                ]
                df = df[pd.Series(mask, index=df.index)]
                notes.append(f"按关键词 {('、'.join(usable))} 筛选：{before} → {len(df)} 件")

        return df, notes

    # -------------------------------------------------------------- 打分
    def _score(
        self,
        candidates: pd.DataFrame,
        preference_vector: np.ndarray | None,
        liked: Sequence[str],
        disliked: Sequence[str],
        has_history: bool,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        df = candidates.copy()

        # --- 热度分（log 缩放，避免头部商品一边倒）---
        df["pop_score"] = np.log1p(df["users"].to_numpy(dtype=float)) / (self._pop_scale or 1.0)

        # --- 个性化相似度 ---
        sims: list[float] = []
        has_vec: list[bool] = []
        for item_id in df["item_id"]:
            in_index = self.index is not None and self.index.contains(str(item_id))
            has_vec.append(in_index)
            if in_index and preference_vector is not None:
                vec = self.index.get_vector(str(item_id))
                sims.append(float(max(0.0, np.dot(preference_vector, vec))))
            else:
                # 商品不在向量索引中、或用户还没有偏好向量时，相似度记 0
                sims.append(0.0)
        df["sim_score"] = sims
        df["has_vector"] = has_vec

        # --- 反馈分：与点赞商品的相似度（反馈对排序的直接影响）---
        liked_vecs = [
            self.index.get_vector(str(i))
            for i in liked
            if self.index is not None and self.index.contains(str(i))
        ]
        if liked_vecs and self.index is not None:
            matrix = np.vstack(liked_vecs)
            fb_scores: list[float] = []
            for item_id in df["item_id"]:
                if self.index.contains(str(item_id)):
                    vec = self.index.get_vector(str(item_id))
                    sims_to_liked = matrix @ vec
                    # 取最相似的 3 个点赞商品求平均，再截断到 [0,1]
                    top = np.sort(sims_to_liked)[-3:]
                    fb_scores.append(float(np.clip(top.mean(), 0.0, 1.0)))
                else:
                    fb_scores.append(0.0)
            df["fb_score"] = fb_scores
        else:
            df["fb_score"] = 0.0

        w = self.weights
        df["score"] = (
            float(w.get("similarity", 0.55)) * df["sim_score"]
            + float(w.get("popularity", 0.30)) * df["pop_score"]
            + float(w.get("feedback", 0.15)) * df["fb_score"]
        )

        info = {
            "has_preference": preference_vector is not None,
            "has_history": has_history,
            "liked_used": len(liked_vecs),
            "disliked_excluded": len(disliked),
        }
        return df, info

    @staticmethod
    def _reason_key(
        has_vector: bool,
        w_sim: float,
        w_pop: float,
        w_fb: float,
        has_preference: bool,
    ) -> str:
        """按三项加权贡献里最大的那一项给出推荐理由。"""
        if not has_vector:
            return "no_vector"
        if not has_preference:
            return "popularity"
        dominant = int(np.argmax([w_sim, w_pop, w_fb]))
        if dominant == 2 and w_fb > 0:
            return "feedback"
        if dominant == 0 and w_sim > 0:
            return "similarity"
        if dominant == 1:
            return "popularity"
        return "balanced"

    # -------------------------------------------------------------- 主入口
    def search(
        self,
        query: str,
        *,
        username: str | None = None,
        top_n: int | None = None,
        force_llm: bool = False,
    ) -> dict[str, Any]:
        """执行一次智能推荐查询。

        ``force_llm=True`` 表示用户在界面上手动点了「大模型识别」：不做"规则够不够
        用"的自动判断，直接请大模型重新理解一次，并优先采用它的结果。
        """
        top_n = int(top_n or self.top_n)
        # ---- 第一步：规则解析 + 过滤（总是先跑规则，零延迟零成本）----
        constraints = qp.parse_query_rules(
            query, self.categories, vocabulary=self._vocabulary
        )
        filtered, filter_notes = self._apply_constraints(constraints)

        # ---- 第二步：请大模型再理解一次 ----
        # 两种触发方式：
        #   · 自动 —— 规则结果"不理想"（候选太少，或压根没解析出条件）；
        #   · 手动 —— 用户点了「大模型识别」按钮，此时无条件调用。
        # 自动模式的采用原则是"只有变得更好才用"（候选更多才采纳）；手动模式则以
        # 大模型的理解为准，因为用户已经明确表示"规则解析得不理想"。
        llm_ready = self.llm is not None and self.llm.available
        if force_llm and not llm_ready:
            constraints.llm_notes.append(
                "你点了「大模型识别」，但大模型当前不可用，仍按规则解析结果推荐（"
                + (self.llm.describe() if self.llm is not None else "未接入大模型")
                + "）"
            )
        elif llm_ready and (
            force_llm or self._should_retry_with_llm(constraints, filtered, query)
        ):
            try:
                llm_constraints = self.llm.parse(
                    query, self.categories, getattr(self, "price_context", "")
                )
            except Exception:  # noqa: BLE001
                llm_constraints = None
            if llm_constraints is not None:
                # 把大模型的解读当作"另一种方案"而不是"补丁"：
                # 方案一 = 规则 + 大模型合并；方案二 = 完全采用大模型的解读
                # （价格仍以规则为准，因为正则对数字最准）。谁的结果多就用谁。
                merged_variant, llm_only = self._llm_variants(constraints, llm_constraints)

                if force_llm:
                    # 手动模式：以大模型的理解为准，依次试两套条件，取第一套有结果的
                    adopted: tuple[qp.QueryConstraints, pd.DataFrame, list[str]] | None = None
                    for variant in (merged_variant, llm_only):
                        df_variant, notes_variant = self._apply_constraints(variant)
                        if len(df_variant) > 0:
                            adopted = (variant, df_variant, notes_variant)
                            break
                    if adopted is not None:
                        constraints, filtered, filter_notes = adopted
                        filter_notes = [
                            "已按你的要求改用大模型识别结果"
                            f"（{getattr(self.llm, 'model', '') or '大模型'}）："
                        ] + filter_notes
                        constraints.llm_notes.append(
                            "大模型给出的条件：" + qp.describe(llm_constraints)
                        )
                    else:
                        constraints.llm_notes.append(
                            "大模型给出的条件（"
                            + qp.describe(llm_constraints)
                            + "）在当前商品库里没匹配到商品，已保留规则解析结果"
                        )
                else:
                    variants = [merged_variant, llm_only]
                    best: tuple[qp.QueryConstraints, pd.DataFrame, list[str]] | None = None
                    for variant in variants:
                        df_variant, notes_variant = self._apply_constraints(variant)
                        if best is None or len(df_variant) > len(best[1]):
                            best = (variant, df_variant, notes_variant)

                    # 采用大模型结果的三种情形（按优先级）：
                    #   1. 规则压根没起到过滤作用（等于"返回全部"）——必须让大模型接管；
                    #   2. 大模型补充了规则没识别出的类别或价格（例如把"便宜"变成价格上限）；
                    #   3. 大模型重解析后候选更多（放宽了过窄的条件）。
                    # 注意不能只用"候选更多"做判据：规则返回全部商品时，
                    # 大模型更准确的窄条件反而会被误判为"更差"。
                    adopt, reason = False, ""
                    if best is not None and len(best[1]) > 0:
                        if not self._constraints_effective(constraints):
                            adopt, reason = True, "规则没能识别出有效条件，改用大模型的理解"
                        elif self._llm_adds_constraint(constraints, llm_constraints):
                            adopt, reason = True, "大模型补充了规则没识别出的条件"
                        elif len(best[1]) > len(filtered):
                            adopt, reason = True, "大模型重解析后候选更多"

                    if adopt:
                        constraints, filtered, filter_notes = best
                        filter_notes = [reason + "："] + filter_notes
                    else:
                        constraints.llm_notes.append("大模型重解析后未带来更好结果，沿用规则结果")
            else:
                hint = (
                    self.llm.failure_hint()
                    if hasattr(self.llm, "failure_hint")
                    else "常见原因：网络不通、Key 无效或超时"
                )
                constraints.llm_notes.append(
                    ("你点了「大模型识别」，但大模型" if force_llm else "大模型解析")
                    + "未返回结果，已回退到规则结果"
                    f"（调用 {getattr(self.llm, 'calls', 0)} 次 / 失败 {getattr(self.llm, 'failures', 0)} 次；"
                    + hint
                    + "）"
                )

        # --- 反馈 ---
        feedback_map: dict[str, str] = {}
        if username:
            feedback_map = fb.get_feedback(PATHS["feedback_db"], username)
        liked, disliked = fb.split_feedback(feedback_map)
        disliked_set = set(disliked)

        # --- 基础用户向量（来自阶段四的关联客户） ---
        base_vector = None
        customer_id = ""
        if username:
            from src import user_store

            user = user_store.get_user(PATHS["user_db"], username)
            if user:
                customer_id = str(user.get("customer_id") or "")
        if customer_id and customer_id in self.user_vectors:
            base_vector = self.user_vectors[customer_id]

        preference_vector, pref_info = fb.build_preference_vector(
            liked, disliked, self.index, base_vector,
            feedback_weight=self.feedback_weight, dislike_penalty=self.dislike_penalty,
        )

        # --- 剔除点踩 → 打分 ---
        if disliked_set:
            before = len(filtered)
            filtered = filtered[~filtered["item_id"].isin(disliked_set)]
            filter_notes.append(f"剔除你点过「不喜欢」的商品：{before} → {len(filtered)} 件")

        scored, score_info = self._score(
            filtered, preference_vector, liked, disliked,
            has_history=bool(customer_id),
        )
        scored = scored.sort_values(["score", "users"], ascending=False).head(top_n)

        results: list[dict[str, Any]] = []
        for row in scored.itertuples(index=False):
            w_sim = float(self.weights.get("similarity", 0.55)) * float(row.sim_score)
            w_pop = float(self.weights.get("popularity", 0.30)) * float(row.pop_score)
            w_fb = float(self.weights.get("feedback", 0.15)) * float(row.fb_score)
            if feedback_map.get(row.item_id) == "like":
                reason_key = "liked"
            else:
                reason_key = self._reason_key(
                    bool(row.has_vector), w_sim, w_pop, w_fb, preference_vector is not None
                )
            results.append(
                {
                    "item_id": row.item_id,
                    "name": row.name,
                    "category": row.category,
                    "price": round(float(row.price), 2),
                    "users": int(row.users),
                    "score": round(float(row.score), 4),
                    "reason": REASON_LABELS[reason_key],
                    "reason_detail": REASON_DETAIL[reason_key],
                    "sim_score": round(float(row.sim_score), 4),
                    "pop_score": round(float(row.pop_score), 4),
                    "fb_score": round(float(row.fb_score), 4),
                    "feedback": feedback_map.get(row.item_id, ""),
                }
            )

        notes = list(constraints.notes) + filter_notes
        notes += constraints.llm_notes
        if not results:
            notes.append("没有符合条件的商品，试试放宽价格或换个品类")
            if self.llm is not None and not self.llm.available:
                # 模糊描述在纯规则下大概率查不到，提示用户还有一条可选路径
                notes.append(
                    "提示：这句话偏模糊，规则解析拿不准；"
                    f"开启大模型解析后可以更好理解（当前状态：{self.llm.describe()}）"
                )

        return {
            "query": constraints.raw,
            "constraints": {
                "categories": constraints.categories,
                "min_price": constraints.min_price,
                "max_price": constraints.max_price,
                "keywords": constraints.keywords,
                "text": qp.describe(constraints),
                "source": constraints.source,
            },
            "parser": {
                "source": constraints.source,
                "llm": self.llm.describe() if self.llm is not None else "未接入",
                "llm_available": bool(getattr(self.llm, "available", False)),
                "llm_forced": bool(force_llm),
                "llm_model": getattr(self.llm, "model", "") if self.llm is not None else "",
                "llm_calls": int(getattr(self.llm, "calls", 0)),
                "llm_failures": int(getattr(self.llm, "failures", 0)),
            },
            "llm_forced": bool(force_llm),
            "total_items": int(len(self.profile)),
            "filtered_items": int(len(filtered)),
            "returned": len(results),
            "notes": notes,
            "personalized": preference_vector is not None,
            "preference_info": {**pref_info, **score_info},
            "liked_count": len(liked),
            "disliked_count": len(disliked),
            "results": results,
        }
