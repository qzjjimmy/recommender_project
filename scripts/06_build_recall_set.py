#!/usr/bin/env python3
"""阶段四入口：用户向量 → 多路召回 → 融合为召回集 → 导出与评估。

构造口径（与阶段三、五共用同一套时间切分，防止泄漏）：
    1. 按时间切成训练期（前 80%）与验证期（后 20%）；
    2. 用户向量、热门榜、相似扩展**全部只用训练期信息**；
    3. 用户集 = 验证期有新购买的用户中固定种子抽 200 个（含冷启动用户）；
    4. 真实答案 = 验证期购买且训练期未买过的商品。

产物：
    data/processed/recall_set.csv        召回集（供阶段五精排）
    models/user_vectors.npz              用户向量（user_ids + vectors）
    outputs/stage04/stage04_recall_stats.txt     覆盖率/热门占比/配额对比/冷启动对比
    outputs/stage04/stage04_quota_comparison.csv 两版配额的逐用户指标
    outputs/stage04/stage04_recall_samples.txt   3 个用户的召回明细（含中文名）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import HYPERPARAMS, PATHS, SEED, set_seed, stage_output_dir
from src import features
from src import product_catalog as catalog
from src import recall as R
from src import user_embedding as ue
from src.popular import PopularRecommender
from src.vector_index import ItemVectorIndex

SAMPLE_USER_NUM = 3
HISTORY_HEAD = 8


def main() -> None:
    set_seed()
    outputs = stage_output_dir(4)
    outputs.mkdir(parents=True, exist_ok=True)
    Path(PATHS["recall_set"]).parent.mkdir(parents=True, exist_ok=True)

    recall_cfg = HYPERPARAMS.get("recall", {})
    final_k = int(recall_cfg.get("final_k", 50))
    quotas = dict(recall_cfg.get("quotas", {}))
    quotas_variant = dict(recall_cfg.get("quotas_variant", {}))
    recent_n = recall_cfg.get("recent_n", 50)
    similar_history_n = int(recall_cfg.get("similar_history_n", 20))
    similar_per_item = int(recall_cfg.get("similar_per_item", 5))
    hot_top_n = int(recall_cfg.get("hot_top_n", 100))
    eval_cfg = HYPERPARAMS.get("eval", {})
    valid_ratio = float(eval_cfg.get("valid_ratio", 0.2))
    sample_users = int(eval_cfg.get("sample_users", 200))

    # ---------- 1) 时间切分（与阶段三/五一致） ----------
    clean = pd.read_csv(PATHS["clean_transactions"], dtype=str)
    split_time = features.choose_split_time(clean, "InvoiceDate", valid_ratio)
    train, valid = features.time_split(clean, "InvoiceDate", split_time=split_time)
    split_info = {
        "split_time": split_time,
        "train_rows": len(train),
        "valid_rows": len(valid),
    }
    print(f"时间切分   : {split_time}  训练期 {len(train)} / 验证期 {len(valid)} 行")

    # ---------- 2) 只用训练期构建物料 ----------
    index = ItemVectorIndex().load(PATHS["vector_index_dir"])
    popular = PopularRecommender(exclude_purchased=True).fit(train)
    hot_items = set(popular.top_items(hot_top_n))
    print(f"向量索引   : {len(index)} 个商品 | 热门榜 Top{hot_top_n} 已建立")

    # 训练期用户历史：按时间从早到晚（保留重复购买，重复即代表更强的兴趣）
    train_sorted = train.sort_values("InvoiceDate", kind="stable")
    histories: dict[str, list[str]] = {
        str(u): [str(x) for x in g]
        for u, g in train_sorted.groupby("CustomerID")["StockCode"].agg(list).items()
    }
    purchased_map = {u: set(h) for u, h in histories.items()}

    # ---------- 3) 用户向量（只用训练期历史） ----------
    user_vectors = ue.build_user_vectors(histories, index, recent_n=recent_n)
    ue.save_user_vectors(user_vectors, PATHS["models_dir"] / "user_vectors.npz")
    print(
        f"用户向量   : 有训练历史 {len(histories)} 人 → 成功建出 {len(user_vectors)} 人"
        f"（维度 {ue.vector_dim(next(iter(user_vectors.values()))) if user_vectors else 0}）"
    )

    # ---------- 4) 真实答案与用户集 ----------
    train_pairs = set(zip(train["CustomerID"], train["StockCode"]))
    ground_truth: dict[str, set[str]] = {}
    for user, item in zip(valid["CustomerID"], valid["StockCode"]):
        if (user, item) not in train_pairs:
            ground_truth.setdefault(user, set()).add(item)

    candidates = sorted(ground_truth)
    rng = np.random.default_rng(SEED)
    if len(candidates) > sample_users:
        picked = rng.choice(np.array(candidates), size=sample_users, replace=False)
        users = sorted(str(u) for u in picked)
    else:
        users = candidates
    cold_users = sum(1 for u in users if u not in histories)
    print(
        f"用户集     : {len(users)} 人（验证期有新购买，SEED={SEED} 抽样）"
        f"，其中冷启动（无训练历史）{cold_users} 人"
    )

    # ---------- 5) 多路召回 + 融合 ----------
    recall_map: dict[str, list[str]] = {}
    road_map: dict[str, dict[str, list[str]]] = {}
    source_map: dict[str, list[tuple[str, str]]] = {}
    variant_map: dict[str, list[str]] = {}
    records: list[dict] = []

    for user in users:
        history = histories.get(user, [])
        purchased = purchased_map.get(user, set())
        user_vec = user_vectors.get(user)

        # 每路都取到 final_k，配额在融合阶段才生效 —— 这样两版配额可以共用同一批候选
        roads = {
            "popular": R.popular_road(popular, user, final_k),
            "similar": R.similar_road(
                index,
                history,
                final_k,
                history_n=similar_history_n,
                per_item=similar_per_item,
                purchased=purchased,
            ),
            "user_vec": R.user_vector_road(index, user_vec, purchased, k=final_k),
        }
        with_sources = R.fuse_recall_with_sources(roads, quotas, final_k)
        fused = [item for item, _ in with_sources]
        contribution = R.road_contribution(with_sources)

        recall_map[user] = fused
        road_map[user] = roads
        source_map[user] = with_sources
        variant_map[user] = R.fuse_recall(roads, quotas_variant, final_k)

        records.append(
            {
                "user_id": user,
                "recall_items": json.dumps(fused, ensure_ascii=False),
                "list_len": len(fused),
                "cold_start": user not in histories,
                "has_vector": user in user_vectors,
                "road_popular": contribution.get("popular", 0),
                "road_similar": contribution.get("similar", 0),
                "road_user_vec": contribution.get("user_vec", 0),
            }
        )

    recall_df = pd.DataFrame.from_records(records)
    recall_df.to_csv(PATHS["recall_set"], index=False, encoding="utf-8")

    # ---------- 6) 评估 ----------
    metrics_default = R.evaluate_recall(recall_map, ground_truth, hot_items)
    metrics_variant = R.evaluate_recall(variant_map, ground_truth, hot_items)

    warm_users = [u for u in users if u in histories]
    cold_user_list = [u for u in users if u not in histories]
    warm_cold = {}
    if warm_users:
        warm_cold["有历史用户"] = R.evaluate_recall(
            {u: recall_map[u] for u in warm_users}, ground_truth, hot_items
        )
    if cold_user_list:
        warm_cold["冷启动用户"] = R.evaluate_recall(
            {u: recall_map[u] for u in cold_user_list}, ground_truth, hot_items
        )

    contributions = {
        road: float(recall_df[f"road_{road}"].mean()) for road in R.ROAD_ORDER
    }

    R.write_recall_report(
        outputs / "stage04_recall_stats.txt",
        split_info=split_info,
        metrics_default=metrics_default,
        metrics_variant=metrics_variant,
        warm_cold=warm_cold,
        contributions=contributions,
        quotas_default=quotas,
        quotas_variant=quotas_variant,
        final_k=final_k,
        sample_users=len(users),
        cold_users=cold_users,
    )

    # 配额对比的逐用户明细
    per_user = pd.DataFrame(
        {
            "user_id": users,
            "cold_start": [u not in histories for u in users],
            "gt_size": [len(ground_truth[u]) for u in users],
            "A_len": [len(recall_map[u]) for u in users],
            "B_len": [len(variant_map[u]) for u in users],
            "A_hits": [len(set(recall_map[u]) & ground_truth[u]) for u in users],
            "B_hits": [len(set(variant_map[u]) & ground_truth[u]) for u in users],
            "A_hot": [sum(1 for i in recall_map[u] if i in hot_items) for u in users],
            "B_hot": [sum(1 for i in variant_map[u] if i in hot_items) for u in users],
        }
    )
    per_user.to_csv(
        outputs / "stage04_quota_comparison.csv", index=False, encoding="utf-8-sig"
    )

    # ---------- 7) 抽样明细（含中文名），便于人工检查 ----------
    products = catalog.load_products()
    names = catalog.build_name_map(products)
    categories = catalog.build_category_map(products)

    sample_ids = [u for u in users if u in histories][:SAMPLE_USER_NUM]
    if len(sample_ids) < SAMPLE_USER_NUM:
        sample_ids = users[:SAMPLE_USER_NUM]

    lines = [
        "召回集抽样明细（阶段四）",
        f"配额：{quotas}，final_k={final_k}",
        "",
    ]
    for user in sample_ids:
        history = histories.get(user, [])
        lines += [
            f"【用户 {user}】",
            f"  训练期历史 {len(history)} 个商品（最近 {HISTORY_HEAD} 个）：",
            "    " + ", ".join(
                f"{i} {names.get(i, '未知商品')}" for i in history[-HISTORY_HEAD:]
            ),
            f"  验证期真实新购买 {len(ground_truth[user])} 个商品",
            f"  召回集 {len(recall_map[user])} 个候选：",
        ]
        for rank, (item, road) in enumerate(source_map[user], 1):
            hit = "★命中" if item in ground_truth[user] else ""
            lines.append(
                f"    {rank:>2}. [{R.ROAD_LABELS.get(road, road)}] {item:<10}"
                f" {names.get(item, '未知商品')}（{categories.get(item, '未知类目')}）{hit}"
            )
        lines.append("")

    lines += [
        "说明：★命中 = 该候选出现在用户验证期的真实新购买中（仅用于人工检查，不影响召回生成）。",
        "",
    ]
    (outputs / "stage04_recall_samples.txt").write_text("\n".join(lines), encoding="utf-8")

    # ---------- 8) 控制台摘要 ----------
    print("\n===== 默认配额召回结果 =====")
    print(f"  配额            : {quotas}")
    print(f"  平均候选数      : {metrics_default['mean_len']:.2f} / {final_k}")
    print(f"  平均覆盖率      : {metrics_default['mean_recall']:.2%}")
    print(f"  命中用户比例    : {metrics_default['hit_user_ratio']:.2%}")
    print(f"  商品级覆盖率    : {metrics_default['item_coverage']:.2%}")
    print(f"  热门占比        : {metrics_default['hot_ratio']:.2%}")
    print("  各路实际贡献    : " + ", ".join(
        f"{R.ROAD_LABELS[r]} {contributions[r]:.2f}" for r in R.ROAD_ORDER
    ))

    print("\n===== 配额对比 =====")
    print(f"  方案A {quotas}  → 覆盖率 {metrics_default['mean_recall']:.2%}"
          f"，热门占比 {metrics_default['hot_ratio']:.2%}")
    print(f"  方案B {quotas_variant}  → 覆盖率 {metrics_variant['mean_recall']:.2%}"
          f"，热门占比 {metrics_variant['hot_ratio']:.2%}")

    print("\n===== 冷启动对比 =====")
    for label, stat in warm_cold.items():
        print(f"  {label}: {stat['users']} 人，覆盖率 {stat['mean_recall']:.2%}"
              f"，热门占比 {stat['hot_ratio']:.2%}")

    print("\n已写出：")
    for path in (
        PATHS["recall_set"],
        PATHS["models_dir"] / "user_vectors.npz",
        outputs / "stage04_recall_stats.txt",
        outputs / "stage04_quota_comparison.csv",
        outputs / "stage04_recall_samples.txt",
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
