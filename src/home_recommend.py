"""推荐首页：把"购买相似"与"季节档期"两块拼成一份首页数据。

首页不是又一个搜索框：它的输入是**时间**和**这个用户自己**，所以两个板块分别
回答两个不同的问题：

    板块一「猜你还会买」（**基于购买过的商品**）
        取用户最近买过的 N 个商品，各自在 Item2Vec 向量里找最相似的邻居，
        按"买得越近权重越大 + 相似度越高越靠前"累加打分，剔除他已经买过的商品。
        口径与阶段四的"相似扩展路"一致，只是这里把结果直接摆到首页。
        每件推荐都带着来源："因为你买过《白色悬挂心形茶烛台》"。

    板块二「季节档期」（**基于当前月份**）
        档期日历（:mod:`src.seasonal`）给出这个月该推的类目池与关键词，
        排序 = 季节指数 × 热度 × 个性化匹配。
        9 月就是开学季（数码 / 文具 / 图书）+ 秋季换季（家居 / 保暖 / 配饰），
        11 月是双 11 大促，12 月是圣诞礼赠。

    顶部「滚动推荐」（**两个板块的头部商品轮流上镜**）
        首页最显眼的位置只有一条横幅，所以这里不是"再算一遍推荐"，而是
        **把上面两个板块的头部商品重新编队**：板块→轮流取、同类目最多 1 条
        （轮播图是按类目取照片的，同类目连着两条等于图片不换），不够再拿全站热销补。
        每条都带着"它为什么在这儿"（来源板块 + 推荐理由），点进去看它的相似商品。

两块共用同一套商品画像（名称/类目/均价/热度）、同一个向量索引、同一份点赞点踩：
点踩的商品在两个板块里都会被剔除，点赞过的商品会被标记并在排序里加权。
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import HYPERPARAMS, PATHS
from src import feedback as fb
from src import seasonal
from src import user_store
from src.purchase_history import HistoryItem, PurchaseHistory, load_purchase_history
from src.seasonal import SeasonTable, load_season_table
from src.smart_search import load_item_profile
from src.user_embedding import load_user_vectors
from src.vector_index import ItemVectorIndex

# 首页的推荐理由：短标签放标签里，长解释放悬浮提示
REASON_LABELS = {
    "bought_similar": "买过相似",
    "season_hot": "季节热销",
    "season_keyword": "档期关键词",
    "popular": "热销",
}
REASON_DETAIL = {
    "bought_similar": "与你购买过的商品相似",
    "season_hot": "该品类在当月的成交占比高于全年平均",
    "season_keyword": "命中当月档期的关键词",
    "popular": "全站热销商品",
}


class HomeRecommender:
    """推荐首页的数据源：购买相似 + 季节档期。"""

    def __init__(
        self,
        *,
        profile: pd.DataFrame,
        index: ItemVectorIndex | None,
        history: PurchaseHistory,
        seasons: SeasonTable,
        user_vectors: Mapping[str, np.ndarray],
        cfg: Mapping[str, Any] | None = None,
    ) -> None:
        self.profile = profile
        self.index = index
        self.history = history
        self.seasons = seasons
        self.user_vectors = dict(user_vectors)
        self.cfg = dict(cfg if cfg is not None else HYPERPARAMS.get("home", {}))

        max_users = float(profile["users"].max() or 1)
        self._pop_scale = math.log1p(max_users)
        self._names = dict(zip(profile["item_id"].astype(str), profile["name"].astype(str)))
        self._categories = dict(
            zip(profile["item_id"].astype(str), profile["category"].astype(str))
        )

    # ---------------------------------------------------------------- 加载
    @classmethod
    def load(cls, *, force_profile: bool = False) -> "HomeRecommender":
        """加载首页推荐需要的全部依赖（缺向量索引时退化为纯热度排序）。"""
        profile = load_item_profile(force=force_profile)
        try:
            index = ItemVectorIndex().load(PATHS["vector_index_dir"])
        except FileNotFoundError:
            index = None
        try:
            user_vectors = load_user_vectors(PATHS["models_dir"] / "user_vectors.npz")
        except (FileNotFoundError, OSError):
            user_vectors = {}
        return cls(
            profile=profile,
            index=index,
            history=load_purchase_history(),
            seasons=load_season_table(),
            user_vectors=user_vectors,
            cfg=HYPERPARAMS.get("home", {}),
        )

    # -------------------------------------------------------------- 小工具
    @staticmethod
    def month_label(month: int) -> str:
        return f"{int(month)} 月"

    @staticmethod
    def _row_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
        """按模板需要的字段组织一行商品。"""
        return {
            "item_id": str(record["item_id"]),
            "name": str(record["name"]),
            "category": str(record["category"]),
            "price": round(float(record["price"]), 2),
            "users": int(record["users"]),
            "score": 0.0,
            "reason": "",
            "reason_detail": "",
            "feedback": "",
            "owned": False,
        }

    def history_items(self, customer_id: str, limit: int | None = None) -> list[HistoryItem]:
        """用户最近买过的商品（默认最近 N 个，N 见 ``HYPERPARAMS['home']``）。"""
        limit = int(limit if limit is not None else self.cfg.get("similar_history_n", 20))
        return self.history.items(customer_id, limit=limit)

    def _diversify(
        self,
        rows: list[dict[str, Any]],
        top_n: int,
        *,
        group_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """多样化：同一个来源商品 / 同一个类目最多占 ``category_cap`` 个位置。

        向量近邻有一个通病：最相似的往往是一堆近重复商品（"红色餐厅挂钟/黑色餐厅
        挂钟/象牙色餐厅挂钟"）。首页只有一排卡片，全放近重复商品体验很差，所以这里
        做一次截流：

            1. 如果给了 ``group_key``（板块一是来源商品），按来源**轮流**取，
               让每个历史商品都能带出一两个推荐，而不是让某一件商品刷屏；
            2. 任何情况下同一个类目最多 ``category_cap`` 个；
            2b. 同名商品只留一个（数据集里存在"名字一样、编码不同"的商品，
               两张一模一样的卡片摆在首页上很难看）；
            3. 名额没用满时，再按原分数顺序把剩下的补回来（结果不会变少）。

        ``rows`` 需已按分数降序。
        """
        category_cap = int(self.cfg.get("category_cap", 2))
        picked: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        category_used: dict[str, int] = {}
        name_used: set[str] = set()

        def take(row: dict[str, Any]) -> bool:
            category = str(row.get("category", ""))
            if category_cap > 0 and category_used.get(category, 0) >= category_cap:
                return False
            name = str(row.get("name", ""))
            if name and name in name_used:
                return False
            category_used[category] = category_used.get(category, 0) + 1
            if name:
                name_used.add(name)
            picked.append(row)
            return True

        if group_key:
            buckets: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                buckets.setdefault(str(row.get(group_key, "")), []).append(row)
            cursors = {key: 0 for key in buckets}
            while len(picked) < top_n and cursors:
                progressed = False
                for key in list(cursors):
                    bucket = buckets[key]
                    index = cursors[key]
                    if index >= len(bucket):
                        del cursors[key]
                        continue
                    cursors[key] = index + 1
                    take(bucket[index])
                    progressed = True
                    if len(picked) >= top_n:
                        break
                if not progressed:
                    break

        taken = {id(row) for row in picked}
        for row in rows:
            if len(picked) >= top_n:
                break
            if id(row) in taken:
                continue
            if take(row):
                continue
            deferred.append(row)
        for row in deferred:  # 类目上限卡住的，按分数依次补回来
            if len(picked) >= top_n:
                break
            picked.append(row)
        return picked[:top_n]

    def history_rows(self, customer_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """把"买过的商品"整理成可展示的行（首页要把推荐依据摆出来）。"""
        rows: list[dict[str, Any]] = []
        for item in self.history.items(customer_id, limit=limit):
            rows.append(
                {
                    "item_id": item.item_id,
                    "name": self._names.get(item.item_id, "未知商品"),
                    "category": self._categories.get(item.item_id, "未知类目"),
                    "price": None,
                    "orders": int(item.orders),
                    "qty": float(item.qty),
                    "last_date": item.last_date,
                }
            )
        prices = dict(
            zip(self.profile["item_id"].astype(str), self.profile["price"].astype(float))
        )
        for row in rows:
            price = prices.get(row["item_id"])
            row["price"] = round(float(price), 2) if price is not None else None
        return rows

    # ------------------------------------------------- 板块一：购买相似推荐
    def similar_rows(
        self,
        history_items: Sequence[HistoryItem],
        *,
        purchased: Iterable[str] = (),
        disliked: Iterable[str] = (),
        liked: Iterable[str] = (),
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """基于购买过的商品做相似扩展（与阶段四"相似扩展路"同一口径）。"""
        top_n = int(top_n if top_n is not None else self.cfg.get("similar_top_n", 6))
        if self.index is None or not history_items:
            return []

        purchased_set = {str(i) for i in purchased}
        disliked_set = {str(i) for i in disliked}
        liked_set = {str(i) for i in liked}
        # 数据集里存在"同一个商品名、不同编码"的商品（例如 22191 / 23315 都叫
        # "象牙色餐厅挂钟"）。只按编码剔除已购，会把这些"影子商品"顶到第一位，
        # 所以这里连名字一起剔除：买过的商品不再出现在"猜你还会买"里。
        purchased_names = {self._names[i] for i in purchased_set if i in self._names}
        per_item = int(self.cfg.get("similar_per_item", 12))
        decay = float(self.cfg.get("recency_decay", 0.92))
        weights = self.cfg.get("weights_similar", {"similarity": 0.75, "popularity": 0.25})

        acc: dict[str, float] = {}
        origin: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(history_items):
            if not self.index.contains(item.item_id):
                continue
            # 越近买过的商品权重越大；回购次数多的也更重要（log 压缩，避免大单一边倒）
            weight = (decay**rank) * math.log1p(max(int(item.orders), 1))
            for candidate, similarity in self.index.search(item.item_id, topn=per_item):
                candidate = str(candidate)
                if candidate in purchased_set or candidate in disliked_set:
                    continue
                if self._names.get(candidate) in purchased_names:
                    continue
                contribution = weight * max(float(similarity), 0.0)
                if contribution <= 0:
                    continue
                acc[candidate] = acc.get(candidate, 0.0) + contribution
                previous = origin.get(candidate)
                if previous is None or contribution > previous["contribution"]:
                    origin[candidate] = {
                        "contribution": contribution,
                        "item_id": item.item_id,
                        "similarity": float(similarity),
                    }

        if not acc:
            return []
        max_acc = max(acc.values()) or 1.0
        subset = self.profile[self.profile["item_id"].astype(str).isin(acc.keys())]

        rows: list[dict[str, Any]] = []
        for record in subset.to_dict("records"):
            item_id = str(record["item_id"])
            sim_norm = acc[item_id] / max_acc
            pop_norm = math.log1p(int(record["users"])) / (self._pop_scale or 1.0)
            score = (
                float(weights.get("similarity", 0.75)) * sim_norm
                + float(weights.get("popularity", 0.25)) * pop_norm
            )
            if item_id in liked_set:  # 明确点过喜欢的往前放
                score += 0.05

            source = origin[item_id]
            source_name = self._names.get(source["item_id"], "你买过的商品")
            row = self._row_from_record(record)
            row.update(
                {
                    "score": round(score, 4),
                    "similarity": round(float(source["similarity"]), 4),
                    "origin_item_id": source["item_id"],
                    "origin_name": source_name,
                    "reason": REASON_LABELS["bought_similar"],
                    "reason_detail": (
                        f"因为你买过《{source_name}》，这件商品与它的向量余弦相似度是 "
                        f"{float(source['similarity']):.3f}"
                    ),
                    "feedback": "like" if item_id in liked_set else "",
                }
            )
            rows.append(row)

        rows.sort(key=lambda r: (-r["score"], -r["users"]))
        # 同一个来源商品、同一个类目都限流，避免首页被近重复商品占满
        return self._diversify(rows, top_n, group_key="origin_item_id")

    # ------------------------------------------------- 板块二：季节档期推荐
    def _keyword_mask(self, theme: Mapping[str, Any]) -> pd.Series:
        """商品名里的档期关键词命中（"笔""书""暖"这类词）。"""
        keywords = [str(k) for k in theme.get("keywords", []) if str(k).strip()]
        mask = pd.Series(False, index=self.profile.index)
        if not keywords:
            return mask
        text = self.profile["name"].fillna("").astype(str)
        for word in keywords:
            mask |= text.str.contains(word, case=False, regex=False)
        return mask

    def season_rows(
        self,
        theme: Mapping[str, Any],
        month: int,
        *,
        preference_vector: np.ndarray | None = None,
        disliked: Iterable[str] = (),
        purchased: Iterable[str] = (),
        liked: Iterable[str] = (),
        exclude: Iterable[str] = (),
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """按档期主题推荐：类目池 + 档期关键词，用季节指数与热度排序。"""
        top_n = int(top_n if top_n is not None else self.cfg.get("season_top_n", 6))
        clip_max = float(self.cfg.get("season_lift_clip", 3.0)) or 1.0
        keyword_score = float(self.cfg.get("season_keyword_score", 0.6))
        weights = self.cfg.get(
            "weights_season", {"season": 0.45, "popularity": 0.30, "personal": 0.25}
        )

        categories = {str(c) for c in theme.get("categories", [])}
        in_pool = self.profile["category"].astype(str).isin(categories)
        keyword_mask = self._keyword_mask(theme)
        candidates = self.profile[in_pool | keyword_mask]
        if candidates.empty:
            return []
        keyword_ids = set(self.profile.loc[keyword_mask, "item_id"].astype(str))

        disliked_set = {str(i) for i in disliked}
        purchased_set = {str(i) for i in purchased}
        liked_set = {str(i) for i in liked}
        excluded = {str(i) for i in exclude}

        rows: list[dict[str, Any]] = []
        for record in candidates.to_dict("records"):
            item_id = str(record["item_id"])
            if item_id in disliked_set or item_id in excluded:
                continue

            category = str(record["category"])
            lift = self.seasons.lift(month, category)
            season_score = min(max(lift, 0.0), clip_max) / clip_max
            keyword_hit = item_id in keyword_ids
            if keyword_hit:  # 类目不在档期池、但名字命中档期关键词
                season_score = max(season_score, keyword_score)

            pop_score = math.log1p(int(record["users"])) / (self._pop_scale or 1.0)
            personal = 0.0
            if (
                preference_vector is not None
                and self.index is not None
                and self.index.contains(item_id)
            ):
                personal = float(max(0.0, np.dot(preference_vector, self.index.get_vector(item_id))))

            score = (
                float(weights.get("season", 0.45)) * season_score
                + float(weights.get("popularity", 0.30)) * pop_score
                + float(weights.get("personal", 0.25)) * personal
            )
            if item_id in liked_set:
                score += 0.05

            if keyword_hit:
                reason = "season_keyword"
                detail = (
                    f"商品名命中 {self.month_label(month)} 档期「{theme.get('name', '')}」的关键词"
                )
            elif lift > 1.05:
                reason = "season_hot"
                detail = (
                    f"{category}在 {month} 月的日均成交额是它全年日均的 {lift:.1f} 倍"
                )
            else:
                reason = "popular"
                detail = f"{category}在 {month} 月是常青品类，按热销排序"

            row = self._row_from_record(record)
            row.update(
                {
                    "score": round(score, 4),
                    "season_lift": round(float(lift), 3),
                    "season_score": round(float(season_score), 4),
                    "personal_score": round(float(personal), 4),
                    "reason": REASON_LABELS[reason],
                    "reason_detail": detail,
                    "feedback": "like" if item_id in liked_set else "",
                    "owned": item_id in purchased_set,
                }
            )
            rows.append(row)

        rows.sort(key=lambda r: (-r["score"], -r["users"]))
        # 按**类目**轮流取：一个档期要覆盖"数码 + 文具 + 图书 + 宿舍"这些品类，
        # 而不是让季节指数最高的那一个类目包场（开学季 = 数码与文教一起推）。
        return self._diversify(rows, top_n, group_key="category")

    # ------------------------------------------------- 全站热销（轮播兜底）
    def popular_rows(
        self,
        *,
        exclude: Iterable[str] = (),
        disliked: Iterable[str] = (),
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """全站热销 TopN（剔除已购与点过不喜欢的）。

        轮播需要固定条数的商品，而上面两个板块在极端情况下会不足
        （没登录、没历史、当月档期商品都被点踩过），所以留一条永远有货的兜底路。
        """
        skip = {str(i) for i in exclude} | {str(i) for i in disliked}
        frame = self.profile[~self.profile["item_id"].astype(str).isin(skip)]
        frame = frame.sort_values(["users", "price"], ascending=[False, True]).head(int(top_n))
        rows: list[dict[str, Any]] = []
        for record in frame.to_dict("records"):
            row = self._row_from_record(record)
            row.update(
                {
                    "reason": REASON_LABELS["popular"],
                    "reason_detail": REASON_DETAIL["popular"],
                }
            )
            rows.append(row)
        return rows

    # ------------------------------------------------- 顶部滚动推荐（轮播）
    def carousel_rows(
        self,
        sections: Sequence[Mapping[str, Any]],
        *,
        purchased: Iterable[str] = (),
        disliked: Iterable[str] = (),
        slides: int | None = None,
        include_popular: bool = True,
    ) -> list[dict[str, Any]]:
        """把各板块的头部商品编成一条轮播（首页最显眼的位置）。

        编队规则与"板块内部"的多样化不是一回事，所以这里重做一次：

            1. **板块轮流取**：板块一 → 档期一 → 档期二 → 板块一 → …，
               一条横幅里既能看到"猜你还会买"也能看到当月档期；
            2. **同类目最多 ``carousel_max_per_category`` 条**（默认 1）：
               轮播图按类目取照片，同类目连着两条等于图片根本没换；
            3. **同名只留一条**（数据集里有同名不同编码的商品）；
            4. 条数不够时：先放宽类目限制，再拿全站热销补满，**不会比要的少**。
        """
        slides = int(slides if slides is not None else self.cfg.get("carousel_slides", 5))
        cap = int(self.cfg.get("carousel_max_per_category", 1))
        purchased_set = {str(i) for i in purchased}
        disliked_set = {str(i) for i in disliked}

        buckets: list[tuple[list[dict[str, Any]], str, str]] = []
        for section in sections:
            rows = [
                row
                for row in section.get("rows", [])
                if str(row.get("item_id")) not in purchased_set
                and str(row.get("item_id")) not in disliked_set
            ]
            if not rows:
                continue
            kind = "personal" if section.get("kind") == "similar" else "season"
            badge = "猜你还会买" if kind == "personal" else str(section.get("title", "当月档期"))
            buckets.append((rows, badge, kind))

        picked: list[dict[str, Any]] = []
        seen_items: set[str] = set()
        seen_names: set[str] = set()
        category_used: dict[str, int] = {}
        cap_on = cap > 0

        def take(row: Mapping[str, Any], badge: str, kind: str) -> None:
            item_id = str(row.get("item_id", ""))
            category = str(row.get("category", ""))
            name = str(row.get("name", ""))
            seen_items.add(item_id)
            if name:
                seen_names.add(name)
            category_used[category] = category_used.get(category, 0) + 1
            picked.append(
                {
                    "item_id": item_id,
                    "name": name,
                    "category": category,
                    "price": row.get("price"),
                    "users": int(row.get("users") or 0),
                    "badge": badge,
                    "badge_kind": kind,
                    "reason": row.get("reason", ""),
                    "reason_detail": row.get("reason_detail", ""),
                    "origin_name": row.get("origin_name", ""),
                    "season_lift": row.get("season_lift"),
                    "owned": bool(row.get("owned", False)),
                    "score": row.get("score", 0.0),
                }
            )

        def is_usable(row: Mapping[str, Any]) -> bool:
            item_id = str(row.get("item_id", ""))
            name = str(row.get("name", ""))
            if not item_id or item_id in seen_items:
                return False
            if name and name in seen_names:
                return False
            if cap_on and category_used.get(str(row.get("category", "")), 0) >= cap:
                return False
            return True

        cursors = [0] * len(buckets)
        while len(picked) < slides:
            progressed = False
            for index, (rows, badge, kind) in enumerate(buckets):
                while cursors[index] < len(rows):
                    row = rows[cursors[index]]
                    cursors[index] += 1
                    if not is_usable(row):
                        continue
                    take(row, badge, kind)
                    progressed = True
                    break
                if len(picked) >= slides:
                    break
            if not progressed:
                break

        if len(picked) < slides and cap_on:  # 先放宽"同类目最多 1 条"
            cap_on = False
            for index, (rows, badge, kind) in enumerate(buckets):
                for row in rows[cursors[index]:]:
                    if len(picked) >= slides:
                        break
                    if is_usable(row):
                        take(row, badge, kind)
                if len(picked) >= slides:
                    break

        if len(picked) < slides and include_popular:  # 再拿全站热销补满
            fallback = self.popular_rows(
                exclude=seen_items,
                disliked=disliked_set,
                top_n=slides - len(picked) + 3,
            )
            for row in fallback:
                if len(picked) >= slides:
                    break
                if not str(row.get("item_id", "")):
                    continue
                take(row, "全站热销", "popular")

        return picked[:slides]

    # ---------------------------------------------------------------- 组装
    def build(
        self,
        *,
        username: str | None = None,
        customer_id: str = "",
        month: int | None = None,
    ) -> dict[str, Any]:
        """组装首页数据：档期背景 + 各板块 + 推荐依据。"""
        month = int(month or date.today().month)
        if month < 1 or month > 12:
            raise ValueError(f"month 必须在 1~12 之间，收到 {month}")

        # ---- 用户侧信号：关联客户、购买历史、点赞点踩 ----
        if username and not customer_id:
            user = user_store.get_user(PATHS["user_db"], username)
            customer_id = str(user.get("customer_id") or "") if user else ""
        feedback_map = fb.get_feedback(PATHS["feedback_db"], username) if username else {}
        liked, disliked = fb.split_feedback(feedback_map)

        recent = self.history_items(customer_id) if customer_id else []
        purchased = self.history.item_ids(customer_id) if customer_id else set()
        base_vector = self.user_vectors.get(str(customer_id)) if customer_id else None
        preference_vector, pref_info = fb.build_preference_vector(
            liked,
            disliked,
            self.index,
            base_vector,
            feedback_weight=float(HYPERPARAMS["search"].get("feedback_weight", 0.8)),
            dislike_penalty=float(HYPERPARAMS["search"].get("dislike_penalty", 0.35)),
        )

        # ---- 板块一：基于购买过的商品 ----
        similar = self.similar_rows(
            recent, purchased=purchased, disliked=disliked, liked=liked
        )
        shown = {str(r["item_id"]) for r in similar}
        sections: list[dict[str, Any]] = [
            {
                "key": "for_you",
                "kind": "similar",
                "title": "猜你还会买",
                "tagline": "基于你购买过的商品，在 Item2Vec 向量里找最接近的邻居",
                "rows": similar,
                "empty": (
                    "还没有可用的购买记录：注册时关联一位演示客户即可，"
                    "或先在智能推荐里点几个「喜欢」。"
                ),
            }
        ]

        # ---- 板块二：当月档期（默认最多两个档期）----
        themes = seasonal.themes_for(month, self.cfg.get("max_themes", 2))
        top_cats = self.seasons.top_categories(month, k=5)
        min_events = int(self.cfg.get("season_min_events", 30))
        for theme in themes:
            rows = self.season_rows(
                theme,
                month,
                preference_vector=preference_vector,
                disliked=disliked,
                purchased=purchased,
                liked=liked,
                exclude=shown,
            )
            shown |= {str(r["item_id"]) for r in rows}
            # 档期类目在当月的实测指数：页面上要能看到"为什么这个档期推这些品类"
            evidence_rows: list[dict[str, Any]] = []
            for category in theme.get("categories", []):
                info = self.seasons.info(month, str(category))
                if info is None:
                    continue
                enough = int(info["events"]) >= min_events
                evidence_rows.append(
                    {
                        "category": str(category),
                        "lift": round(float(info["lift"]), 2) if enough else None,
                        "events": int(info["events"]),
                    }
                )
            evidence_rows.sort(key=lambda r: -(r["lift"] if r["lift"] is not None else 0.0))
            lifts = [r["lift"] for r in evidence_rows if r["lift"] is not None]
            note = ""
            if lifts and all(0.9 <= float(v) <= 1.1 for v in lifts):
                note = (
                    "该档期类目在本数据集里的季节指数都接近 1"
                    "（补充数据是按均匀日期生成的，本身没有季节性），"
                    "所以这一节主要靠热度与你的偏好排序，档期日历负责给出候选品类。"
                )
            sections.append(
                {
                    "key": theme["key"],
                    "kind": "season",
                    "title": theme["name"],
                    "tagline": theme.get("tagline", ""),
                    "categories": list(theme.get("categories", [])),
                    "keywords": list(theme.get("keywords", [])),
                    "evidence_rows": evidence_rows,
                    "note": note,
                    "rows": rows,
                    "empty": "这个档期暂时没有匹配到商品，换个品类词在智能推荐里搜搜看。",
                }
            )

        history_rows = self.history_rows(customer_id) if customer_id else []
        history_categories: list[dict[str, Any]] = []
        if customer_id and self.history.count(customer_id):
            counts: dict[str, int] = {}
            for item in self.history.items(customer_id):
                category = self._categories.get(item.item_id)
                if category:
                    counts[category] = counts.get(category, 0) + 1
            ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
            history_categories = [{"category": c, "count": n} for c, n in ranked]

        # ---- 顶部滚动推荐：把上面几个板块的头部商品编成一条横幅 ----
        carousel = self.carousel_rows(
            sections, purchased=purchased, disliked=disliked
        )

        return {
            "month": month,
            "month_label": self.month_label(month),
            "today": date.today().isoformat(),
            "themes": [{"key": t["key"], "name": t["name"]} for t in themes],
            "season_top_categories": top_cats,
            "coverage": self.seasons.coverage(month),
            "logged_in": bool(username),
            "customer_id": customer_id,
            "history_count": self.history.count(customer_id) if customer_id else 0,
            "history_used": len(recent),
            "history_rows": history_rows,
            "history_top_categories": history_categories,
            "personalized": preference_vector is not None,
            "preference_info": pref_info,
            "like_count": len(liked),
            "dislike_count": len(disliked),
            "carousel": carousel,
            "carousel_interval_ms": int(self.cfg.get("carousel_interval_ms", 6000)),
            "sections": sections,
        }
