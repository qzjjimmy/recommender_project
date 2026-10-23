#!/usr/bin/env python3
"""阶段五入口：防泄漏特征 → 样本构造 → LR / LightGBM 精排 → 在线排序。

时间口径（本项目统一，防泄漏的关键）：

    ├──────────────── 训练期 (80%) ────────────────┼── 验证期 (20%) ──┤
    │   特征期         │      标签期                │                   │
    │ ← 只用来算特征 → │  ← 只用来取标签 →          │  ← 只用来评测 →   │

特征一律取"预测时刻之前 180 天以内"的固定窗口，训练与评测窗口等长、口径可比。

产物：
    models/ranker_lr.pkl / ranker_lgbm.pkl / feature_engine.pkl
    outputs/stage05/stage05_train_log.txt        切分、样本规模、特征清单、超参
    outputs/stage05/stage05_lr_report.txt        LR 指标与系数解读
    outputs/stage05/stage05_lgbm_report.txt      LightGBM 指标与特征重要性
    outputs/stage05/stage05_comparison.txt       三行对比表 + 双模型 TopN 重叠率
    outputs/stage05/stage05_online_demo.txt      在线排序演示（召回→特征→打分→TopN）
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config.settings import HYPERPARAMS, PATHS, SEED, set_seed, stage_output_dir
from src import features as F
from src import metrics as M
from src import online as O
from src import product_catalog as catalog
from src import recall as R
from src import ranker_lgbm, ranker_lr
from src.popular import PopularRecommender
from src.vector_index import ItemVectorIndex

DEMO_USER_NUM = 3
DEMO_TOP_N = 10


def main() -> None:
    set_seed()
    outputs = stage_output_dir(5)
    outputs.mkdir(parents=True, exist_ok=True)
    models_dir = Path(PATHS["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    rank_cfg = HYPERPARAMS.get("rank", {})
    eval_cfg = HYPERPARAMS.get("eval", {})
    recall_cfg = HYPERPARAMS.get("recall", {})

    topn = int(rank_cfg.get("topn", 10))
    lookback_days = int(rank_cfg.get("lookback_days", 180))
    inner_ratio = float(rank_cfg.get("inner_label_ratio", 0.3))
    neg_ratio = int(rank_cfg.get("negative_ratio", 4))
    max_train_users = int(rank_cfg.get("max_train_users", 2000))
    valid_ratio = float(eval_cfg.get("valid_ratio", 0.2))
    recent_n = recall_cfg.get("recent_n", 50)
    hot_top_n = int(recall_cfg.get("hot_top_n", 100))

    # ---------- 1) 时间切分：训练期 / 验证期，再切特征期 / 标签期 ----------
    clean = pd.read_csv(PATHS["clean_transactions"], dtype=str)
    split_time = F.choose_split_time(clean, "InvoiceDate", valid_ratio)
    train, valid = F.time_split(clean, "InvoiceDate", split_time=split_time)
    inner_split = F.choose_split_time(train, "InvoiceDate", inner_ratio)
    feature_df, label_df = F.time_split(train, "InvoiceDate", split_time=inner_split)
    print(f"验证期切分点 : {split_time}   训练期 {len(train)} / 验证期 {len(valid)} 行")
    print(f"标签期切分点 : {inner_split}   特征期 {len(feature_df)} / 标签期 {len(label_df)} 行")

    # ---------- 2) 两个特征引擎（同一套构造逻辑，只有预测时刻不同） ----------
    index = ItemVectorIndex().load(PATHS["vector_index_dir"])
    products = catalog.load_products()
    item_category = catalog.build_category_map(products)
    names = catalog.build_name_map(products)

    engine_train = F.FeatureEngine.fit(
        train,
        reference_time=inner_split,
        lookback_days=lookback_days,
        item_index=index,
        item_category=item_category,
        recent_n=recent_n,
        hot_top_n=hot_top_n,
    )
    engine_eval = F.FeatureEngine.fit(
        train,
        reference_time=split_time,
        lookback_days=lookback_days,
        item_index=index,
        item_category=item_category,
        recent_n=recent_n,
        hot_top_n=hot_top_n,
    )
    print(
        f"特征引擎     : 训练窗口 {engine_train.user_stats.shape[0]} 用户 / "
        f"{engine_train.item_stats.shape[0]} 商品；"
        f"评测窗口 {engine_eval.user_stats.shape[0]} 用户 / {engine_eval.item_stats.shape[0]} 商品"
    )

    # ---------- 3) 训练样本：在"与推理同构的召回候选池"内构造正负样本 ----------
    # 这里刻意不使用"全库随机采样"或"热度加权采样"来造负样本。
    # 我们第一版用的是全库热度加权负采样，结果训练分布与线上候选池不一致：
    # 模型学出 i_users 系数 = -1.08（把"热门"当成负样本），验证集 AUC 只有 0.366。
    # 正确做法是让负样本来自与线上同一套多路召回的候选池。
    train_typed = F.prepare_interactions(train)
    feature_window = train_typed[
        (train_typed["InvoiceDate"] >= inner_split - pd.Timedelta(days=lookback_days))
        & (train_typed["InvoiceDate"] < inner_split)
    ]
    popular_feat = PopularRecommender(exclude_purchased=True).fit(feature_window)

    label_by_user: dict[str, set[str]] = {}
    for user, item in zip(label_df["CustomerID"], label_df["StockCode"]):
        label_by_user.setdefault(str(user), set()).add(str(item))

    quotas = dict(recall_cfg.get("quotas", {}))
    final_k = int(recall_cfg.get("final_k", 50))
    similar_history_n = int(recall_cfg.get("similar_history_n", 20))
    similar_per_item = int(recall_cfg.get("similar_per_item", 5))

    eligible_users = sorted(u for u in engine_train.histories if u in label_by_user)
    rng = np.random.default_rng(SEED)
    if len(eligible_users) > max_train_users:
        picked = rng.choice(np.array(eligible_users), size=max_train_users, replace=False)
        train_users = sorted(str(x) for x in picked)
    else:
        train_users = eligible_users
    print(f"训练用户     : {len(train_users)} 人（特征期有历史且标签期有新购买）")

    sample_rows: list[dict] = []
    for user in train_users:
        history = engine_train.histories.get(user, [])
        purchased = set(history)
        roads = {
            "popular": popular_feat.recommend(user, k=final_k),
            "similar": R.similar_road(
                index, history, final_k,
                history_n=similar_history_n, per_item=similar_per_item, purchased=purchased,
            ),
            "user_vec": R.user_vector_road(
                index, engine_train.user_vectors.get(user), purchased, k=final_k
            ),
        }
        bought = label_by_user.get(user, set())
        for item in R.fuse_recall(roads, quotas, final_k):
            sample_rows.append(
                {"user_id": user, "item_id": item, "label": 1 if item in bought else 0}
            )

    candidates = pd.DataFrame(sample_rows, columns=["user_id", "item_id", "label"])
    if neg_ratio > 0:
        # 可选：限制每个用户的负样本数量，控制样本规模与正负比
        kept: list[pd.DataFrame] = []
        for _, group in candidates.groupby("user_id", sort=False):
            pos = group[group["label"] == 1]
            neg = group[group["label"] == 0]
            limit = max(1, len(pos) * neg_ratio)
            kept.append(pd.concat([pos, neg.head(limit)], ignore_index=True))
        candidates = pd.concat(kept, ignore_index=True)

    positives = set(
        zip(candidates.loc[candidates["label"] == 1, "user_id"],
            candidates.loc[candidates["label"] == 1, "item_id"])
    )
    negatives = set(
        zip(candidates.loc[candidates["label"] == 0, "user_id"],
            candidates.loc[candidates["label"] == 0, "item_id"])
    )
    print(
        f"候选池       : {len(candidates)} 个 (用户, 候选) 对，"
        f"正例 {len(positives)} / 负例 {len(negatives)}"
        f"（正例 = 标签期真实新购买且落在候选池内）"
    )

    # ---------- 4) 生成训练样本（特征只用特征期信息） ----------
    train_samples = F.build_samples_and_features(
        candidates[["user_id", "item_id"]], positives, engine_train
    )
    # 标签由 build_samples_and_features 依据 (user, item) ∈ positives 逐行判定，与行顺序无关；
    # 绝不能按位置赋标签（build_many 会按用户重排行的顺序）。
    assert set(train_samples["label"].unique()) == {0, 1}, "训练样本必须同时包含正负样本"
    assert int(train_samples["label"].sum()) == len(positives), "正样本数量与标签不一致"
    print(f"训练样本     : {len(train_samples)} 行 × {len(F.FEATURE_COLS)} 个特征")

    # ---------- 5) 训练两个模型（完全相同的数据与特征列） ----------
    lr_cfg = rank_cfg.get("lr", {})
    lgbm_cfg = rank_cfg.get("lgbm", {})
    bundle_lr = ranker_lr.train_lr(
        train_samples, F.FEATURE_COLS, "label",
        C=float(lr_cfg.get("C", 1.0)), max_iter=int(lr_cfg.get("max_iter", 1000)), seed=SEED,
    )
    bundle_lgbm = ranker_lgbm.train_lgbm(
        train_samples, F.FEATURE_COLS, "label",
        n_estimators=int(lgbm_cfg.get("n_estimators", 400)),
        learning_rate=float(lgbm_cfg.get("learning_rate", 0.05)),
        num_leaves=int(lgbm_cfg.get("num_leaves", 31)),
        min_child_samples=int(lgbm_cfg.get("min_child_samples", 50)),
        seed=SEED,
    )
    ranker_lr.save_lr(bundle_lr, models_dir / "ranker_lr.pkl")
    ranker_lgbm.save_lgbm(bundle_lgbm, models_dir / "ranker_lgbm.pkl")
    with (models_dir / "feature_engine.pkl").open("wb") as handle:
        pickle.dump(engine_eval, handle)
    print(f"模型已保存   : ranker_lr.pkl / ranker_lgbm.pkl / feature_engine.pkl")

    train_auc_lr = roc_auc_score(
        train_samples["label"], ranker_lr.predict_scores(bundle_lr, train_samples)
    )
    train_auc_lgbm = roc_auc_score(
        train_samples["label"], ranker_lgbm.predict_scores(bundle_lgbm, train_samples)
    )

    # ---------- 6) 在召回集上评测（真实任务：候选内排序） ----------
    recall_df = pd.read_csv(PATHS["recall_set"], dtype={"user_id": str})
    recall_map = {
        str(r.user_id): [str(x) for x in json.loads(r.recall_items)]
        for r in recall_df.itertuples(index=False)
    }
    eval_users = list(recall_map)

    train_pairs = set(zip(train["CustomerID"], train["StockCode"]))
    ground_truth: dict[str, set[str]] = {}
    for user, item in zip(valid["CustomerID"], valid["StockCode"]):
        if (user, item) not in train_pairs:
            ground_truth.setdefault(user, set()).add(item)

    pairs = pd.DataFrame(
        [(u, i) for u, items in recall_map.items() for i in items],
        columns=["user_id", "item_id"],
    )
    feat = engine_eval.build_many(pairs)
    # 注意：ground_truth[user] 是"商品集合"，判断的是 i 是否属于它
    labels = np.array(
        [1 if i in ground_truth.get(u, set()) else 0
         for u, i in zip(feat["user_id"], feat["item_id"])]
    )
    feat["lr_score"] = ranker_lr.predict_scores(bundle_lr, feat)
    feat["lgbm_score"] = ranker_lgbm.predict_scores(bundle_lgbm, feat)
    print(f"\n评测集       : {len(feat)} 个 (用户, 候选) 对，正例 {int(labels.sum())} 个")

    auc_lr = roc_auc_score(labels, feat["lr_score"])
    auc_lgbm = roc_auc_score(labels, feat["lgbm_score"])

    popular = PopularRecommender(exclude_purchased=True).fit(train)
    per_user: list[dict] = []
    for user in eval_users:
        sub = feat[feat["user_id"] == user]
        gt = ground_truth.get(user, set())
        top_lr = sub.sort_values("lr_score", ascending=False)["item_id"].head(topn).tolist()
        top_lgbm = sub.sort_values("lgbm_score", ascending=False)["item_id"].head(topn).tolist()
        top_pop = popular.recommend(user, k=topn)
        cand = recall_map[user]
        per_user.append(
            {
                "user_id": user,
                "gt_size": len(gt),
                "cand_size": len(cand),
                "oracle_hits": min(topn, len(set(cand) & gt)),
                "pop_hits": len(set(top_pop) & gt),
                "lr_hits": len(set(top_lr) & gt),
                "lgbm_hits": len(set(top_lgbm) & gt),
                "overlap_lr_lgbm": len(set(top_lr) & set(top_lgbm)),
            }
        )
    u = pd.DataFrame(per_user)

    def p_at_k(hits: pd.Series) -> float:
        return float(hits.mean() / topn)

    def r_at_k(hits: pd.Series) -> float:
        return float(np.mean([h / g if g else 0.0 for h, g in zip(hits, u["gt_size"])]))

    summary = {
        "pop_p": p_at_k(u["pop_hits"]), "pop_r": r_at_k(u["pop_hits"]),
        "lr_p": p_at_k(u["lr_hits"]), "lr_r": r_at_k(u["lr_hits"]),
        "lgbm_p": p_at_k(u["lgbm_hits"]), "lgbm_r": r_at_k(u["lgbm_hits"]),
        "oracle_p": p_at_k(u["oracle_hits"]),
        "mean_overlap": float(u["overlap_lr_lgbm"].mean()),
        "overlap_rate": float(u["overlap_lr_lgbm"].mean() / topn),
    }

    # ---------- 7) 在线排序演示 ----------
    # 演示挑"有命中"与"零命中"两类用户，既展示成功也暴露失败
    ranked = u.sort_values("lr_hits", ascending=False)
    hit_users = ranked[ranked["lr_hits"] > 0]["user_id"].tolist()
    miss_users = ranked[ranked["lr_hits"] == 0]["user_id"].tolist()
    demo_users = hit_users[:2] + miss_users[:1]
    demo_note = {user: "" for user in demo_users}
    if hit_users:
        demo_note[hit_users[0]] = "（LR 命中最多）"
    if miss_users:
        demo_note[miss_users[0]] = "（LR 零命中，失败案例）"
    demo_lines = [
        "在线排序演示（阶段五）：召回集 → 特征 → 打分 → TopN",
        f"模型：LR / LightGBM（同特征、同候选池），topn={topn}",
        "选取规则：2 个有命中的用户 + 1 个零命中的用户，成功与失败都展示",
        "",
    ]
    for user in demo_users:
        gt = ground_truth.get(user, set())
        details = O.rank_with_details(
            user, recall_map[user], bundle_lr, engine_eval.build, topn=topn, name_map=names
        )
        demo_lines += [
            f"【用户 {user}】{demo_note.get(user, '')}"
            f"召回候选 {len(recall_map[user])} 个，验证期真实新购买 {len(gt)} 个",
            f"  LR 精排 Top{topn}：",
        ]
        for d in details:
            mark = "★命中" if d["item_id"] in gt else ""
            demo_lines.append(
                f"    {d['rank']:>2}. {d['item_id']:<10} {d['name']:<22} 分数 {d['score']:.4f} {mark}"
            )
        demo_lines.append("")
    demo_lines.append("说明：★命中 = 该商品出现在用户验证期的真实新购买中（仅用于检查，不参与打分）。")
    (outputs / "stage05_online_demo.txt").write_text("\n".join(demo_lines), encoding="utf-8")

    # ---------- 8) 写出各份报告 ----------
    train_log = [
        "精排训练日志（阶段五）",
        "=" * 58,
        "",
        "[时间口径]",
        f"  验证期切分点        : {split_time}",
        f"  标签期切分点        : {inner_split}",
        f"  训练期 / 验证期行数 : {len(train)} / {len(valid)}",
        f"  特征期 / 标签期行数 : {len(feature_df)} / {len(label_df)}",
        f"  特征回看窗口        : 预测时刻前 {lookback_days} 天（训练与评测等长，口径可比）",
        "  防泄漏说明          : 训练特征只取标签期起点之前的数据；",
        "                        评测特征只取验证期切分点之前的数据；",
        "                        两边都不使用各自预测目标期的任何信息。",
        "",
        "[样本]",
        "  负样本来源          : 与推理同构的**召回候选池**（不用全库采样，避免分布偏移）",
        f"  训练用户数          : {len(train_users)}（上限 {max_train_users}）",
        f"  正样本              : {len(positives)}（标签期新购买且落在候选池内）",
        f"  负样本              : {len(negatives)}（同一候选池内未被购买的商品）",
        f"  训练样本合计        : {len(train_samples)}",
        f"  可选负样本上限      : negative_ratio={neg_ratio}（0 = 保留池内全部候选）",
        "",
        "[特征清单]",
    ]
    for group, cols in (("用户特征", F.USER_FEATURES), ("商品特征", F.ITEM_FEATURES), ("匹配特征", F.MATCH_FEATURES)):
        train_log.append(f"  {group}: {', '.join(cols)}")
    train_log += [
        "",
        "[超参]",
        f"  LR      : {dict(lr_cfg)}",
        f"  LightGBM: {dict(lgbm_cfg)}",
        f"  lightgbm 版本: {__import__('lightgbm').__version__}",
        "",
        "[训练集 AUC（仅供参考，不是评测指标）]",
        f"  LR       : {train_auc_lr:.4f}",
        f"  LightGBM : {train_auc_lgbm:.4f}",
        "",
    ]
    (outputs / "stage05_train_log.txt").write_text("\n".join(train_log), encoding="utf-8")

    lr_report = [
        "逻辑回归精排报告（阶段五）",
        "=" * 58,
        "",
        "[评测指标]（验证期召回候选上的真实排序任务）",
        f"  AUC        : {auc_lr:.4f}",
        f"  Precision@{topn} : {summary['lr_p']:.4f}",
        f"  Recall@{topn}    : {summary['lr_r']:.4f}",
        "",
        "[特征系数解读]（标准化尺度，绝对值越大影响越强）",
    ]
    lr_report += ranker_lr.describe_coefficients(bundle_lr, top_n=8)
    coef = ranker_lr.coefficients(bundle_lr)
    lr_report += [
        "",
        "[完整系数表]",
        f"  {'特征':<18}{'系数':>12}{'odds比':>12}",
    ]
    for row in coef.itertuples(index=False):
        lr_report.append(f"  {row.feature:<18}{row.coefficient:>12.4f}{row.odds_ratio:>12.4f}")
    lr_report.append("")
    (outputs / "stage05_lr_report.txt").write_text("\n".join(lr_report), encoding="utf-8")

    lgbm_report = [
        "LightGBM 精排报告（阶段五）",
        "=" * 58,
        "",
        "[评测指标]（与 LR **完全相同的训练集、验证候选与特征列**）",
        f"  AUC        : {auc_lgbm:.4f}",
        f"  Precision@{topn} : {summary['lgbm_p']:.4f}",
        f"  Recall@{topn}    : {summary['lgbm_r']:.4f}",
        "",
        "[特征重要性 Top5]",
    ]
    lgbm_report += ranker_lgbm.describe_importance(bundle_lgbm, top_n=5)
    imp = ranker_lgbm.feature_importance(bundle_lgbm)
    lgbm_report += [
        "",
        "[完整重要性表]",
    ]
    for row in imp.itertuples(index=False):
        lgbm_report.append(f"  {row.feature:<18}{row.importance:>10.4f}")
    lgbm_report.append("")
    (outputs / "stage05_lgbm_report.txt").write_text("\n".join(lgbm_report), encoding="utf-8")

    comparison = [
        "精排对比表（阶段五）",
        "=" * 58,
        "",
        f"统一口径：同一批用户（{len(eval_users)} 人）、同一个 K（{topn}）",
        "评测集：验证期召回候选（只由训练期信息生成，不含验证期信息）",
        "",
        f"  {'方案':<24}{'Precision@K':>14}{'Recall@K':>12}{'AUC':>10}",
        f"  {'单路热门':<24}{summary['pop_p']:>14.4f}{summary['pop_r']:>12.4f}{'—':>10}",
        f"  {'融合召回 + LR':<22}{summary['lr_p']:>14.4f}{summary['lr_r']:>12.4f}{auc_lr:>10.4f}",
        f"  {'融合召回 + LightGBM':<19}{summary['lgbm_p']:>14.4f}{summary['lgbm_r']:>12.4f}{auc_lgbm:>10.4f}",
        f"  {'（参照）召回集上界':<21}{summary['oracle_p']:>14.4f}{'—':>12}{'—':>10}",
        "",
        "[双模型 TopN 重叠率]（同用户、同候选池，只换排序模型）",
        f"  平均重叠个数 : {summary['mean_overlap']:.2f} / {topn}",
        f"  平均重叠率   : {summary['overlap_rate']:.2%}",
        "",
        "[结论]",
        f"  1) 精排相对单路热门：Precision@K {summary['pop_p']:.4f} → {summary['lgbm_p']:.4f}"
        f"（LR {summary['lr_p']:.4f}）",
        "  2) 「召回集上界」表示候选池里最多能命中多少 —— 精排不可能超过它，",
        "     与上界的差距反映排序模型的能力，与热门基线的差距反映整条链路的价值。",
        "  3) 双模型重叠率越低，说明两者的排序偏好差异越大（树模型更偏非线性交叉），",
        "     但重叠低不等于更好，仍要看 Precision / Recall。",
        "",
    ]
    (outputs / "stage05_comparison.txt").write_text("\n".join(comparison), encoding="utf-8")

    # 逐用户明细
    u.to_csv(outputs / "stage05_per_user.csv", index=False, encoding="utf-8-sig")

    # ---------- 9) 控制台摘要 ----------
    print("\n===== 评测结果（验证期召回候选上的排序）=====")
    print(f"  单路热门        P@{topn} {summary['pop_p']:.4f}   R@{topn} {summary['pop_r']:.4f}")
    print(f"  融合召回 + LR   P@{topn} {summary['lr_p']:.4f}   R@{topn} {summary['lr_r']:.4f}   AUC {auc_lr:.4f}")
    print(f"  融合召回 + LGBM P@{topn} {summary['lgbm_p']:.4f}   R@{topn} {summary['lgbm_r']:.4f}   AUC {auc_lgbm:.4f}")
    print(f"  召回集上界      P@{topn} {summary['oracle_p']:.4f}")
    print(f"  双模型重叠率    {summary['overlap_rate']:.2%}（{summary['mean_overlap']:.2f}/{topn}）")
    print(f"\n  训练集 AUC（参考）：LR {train_auc_lr:.4f} / LGBM {train_auc_lgbm:.4f}")

    print("\n已写出：")
    for path in (
        models_dir / "ranker_lr.pkl",
        models_dir / "ranker_lgbm.pkl",
        models_dir / "feature_engine.pkl",
        outputs / "stage05_train_log.txt",
        outputs / "stage05_lr_report.txt",
        outputs / "stage05_lgbm_report.txt",
        outputs / "stage05_comparison.txt",
        outputs / "stage05_online_demo.txt",
        outputs / "stage05_per_user.csv",
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
