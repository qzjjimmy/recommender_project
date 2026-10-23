#!/usr/bin/env python3
"""阶段三入口（二）：热门基线 vs 协同过滤（ItemCF）对比实验。

评测口径（与阶段五共用同一套时间切分，避免自嗨）：
    1. 按时间把交易切成训练期（前 80%）与验证期（后 20%）；
    2. **热门与 CF 都只用训练期数据训练**，杜绝用未来信息；
    3. 真实答案 = 用户在验证期购买、且训练期没买过的商品；
    4. 评测用户 = 训练期有行为、且验证期有真实新购买的用户，固定种子抽 200 个；
    5. 同一批用户、同一个 K=10，比较 Precision@K / Recall@K 与两路重叠率。

产物：
    outputs/stage03/stage03_cf_vs_popular.txt          对比结论与 2 个用户明细案例
    outputs/stage03/stage03_cf_vs_popular_peruser.csv  每个评测用户的明细指标
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import HYPERPARAMS, PATHS, SEED, set_seed, stage_output_dir
from src import features
from src import metrics as M
from src import product_catalog as catalog
from src.collaborative_filtering import CollaborativeFiltering
from src.popular import PopularRecommender


def main() -> None:
    set_seed()
    outputs = stage_output_dir(3)
    outputs.mkdir(parents=True, exist_ok=True)

    eval_cfg = HYPERPARAMS.get("eval", {})
    valid_ratio = float(eval_cfg.get("valid_ratio", 0.2))
    k = int(eval_cfg.get("k", 10))
    sample_users = int(eval_cfg.get("sample_users", 200))
    top_similar = int(HYPERPARAMS.get("cf", {}).get("top_similar", 0))

    # ---------- 1) 时间切分 ----------
    clean = pd.read_csv(PATHS["clean_transactions"], dtype=str)
    split_time = features.choose_split_time(clean, "InvoiceDate", valid_ratio)
    train, valid = features.time_split(clean, "InvoiceDate", split_time=split_time)
    print(f"总行数 {len(clean)} → 训练期 {len(train)} / 验证期 {len(valid)}")
    print(f"切分时间点 : {split_time}")

    # ---------- 2) 只用训练期训练两个基线 ----------
    pop = PopularRecommender(exclude_purchased=True).fit(train)
    cf = CollaborativeFiltering(top_similar=top_similar, exclude_purchased=True).fit(train)
    print(f"热门基线   : {len(pop.ranking_)} 个商品有交互记录")
    print(
        f"ItemCF     : {cf.n_items_} 商品 × {cf.n_users_} 用户，"
        f"相似度矩阵非零元 {cf.similarity_.nnz}"
    )

    # ---------- 3) 真实答案：验证期新购买 ----------
    train_pairs = set(zip(train["CustomerID"], train["StockCode"]))
    ground_truth: dict[str, set[str]] = {}
    for user, item in zip(valid["CustomerID"], valid["StockCode"]):
        if (user, item) not in train_pairs:
            ground_truth.setdefault(user, set()).add(item)

    train_users = set(train["CustomerID"])
    eligible = sorted(u for u in ground_truth if u in train_users)
    print(f"验证期有新购买的用户 : {len(ground_truth)}，其中有训练历史的 : {len(eligible)}")

    rng = np.random.default_rng(SEED)
    if len(eligible) > sample_users:
        picked = rng.choice(np.array(eligible), size=sample_users, replace=False)
        users = sorted(str(u) for u in picked)
    else:
        users = eligible
    print(f"评测用户数 : {len(users)}（固定种子 SEED={SEED}，规则：可评测用户中随机抽样）")

    # ---------- 4) 同一批用户、同一个 K 逐用户评测 ----------
    rows: list[dict] = []
    detail: dict[str, dict] = {}
    for user in users:
        gt = ground_truth[user]
        pop_list = pop.recommend(user, k=k)
        cf_list = cf.recommend(user, k=k)
        row = {
            "user_id": user,
            "gt_size": len(gt),
            "pop_hits": M.overlap_count(pop_list, gt),
            "cf_hits": M.overlap_count(cf_list, gt),
            "pop_precision": M.precision_at_k(pop_list, gt, k),
            "cf_precision": M.precision_at_k(cf_list, gt, k),
            "pop_recall": M.recall_at_k(pop_list, gt, k),
            "cf_recall": M.recall_at_k(cf_list, gt, k),
            "overlap": M.overlap_count(pop_list, cf_list),
            "cf_len": len(cf_list),
            "pop_len": len(pop_list),
        }
        rows.append(row)
        detail[user] = {"gt": gt, "pop": pop_list, "cf": cf_list}

    per_user = pd.DataFrame(rows)
    per_user.to_csv(
        outputs / "stage03_cf_vs_popular_peruser.csv", index=False, encoding="utf-8-sig"
    )

    # ---------- 5) 汇总 ----------
    summary = {
        "pop_precision": float(per_user["pop_precision"].mean()),
        "cf_precision": float(per_user["cf_precision"].mean()),
        "pop_recall": float(per_user["pop_recall"].mean()),
        "cf_recall": float(per_user["cf_recall"].mean()),
        "mean_overlap": float(per_user["overlap"].mean()),
        "mean_overlap_rate": float(per_user["overlap"].mean() / k),
        "cf_empty_users": int((per_user["cf_len"] == 0).sum()),
        "cf_win_users": int((per_user["cf_hits"] > per_user["pop_hits"]).sum()),
        "pop_win_users": int((per_user["pop_hits"] > per_user["cf_hits"]).sum()),
        "tie_users": int((per_user["cf_hits"] == per_user["pop_hits"]).sum()),
    }

    print("\n===== 对比结果（同用户集、同 K）=====")
    print(f"  K                     : {k}")
    print(f"  评测用户数            : {len(users)}")
    print(f"  热门 Precision@{k}      : {summary['pop_precision']:.4f}")
    print(f"  ItemCF Precision@{k}    : {summary['cf_precision']:.4f}")
    print(f"  热门 Recall@{k}         : {summary['pop_recall']:.4f}")
    print(f"  ItemCF Recall@{k}       : {summary['cf_recall']:.4f}")
    print(f"  两路平均重叠个数      : {summary['mean_overlap']:.4f} / {k}"
          f"（重叠率 {summary['mean_overlap_rate']:.2%}）")
    print(f"  CF 返回空列表的用户   : {summary['cf_empty_users']}")
    print(f"  CF 胜 / 热门胜 / 打平 : {summary['cf_win_users']} / "
          f"{summary['pop_win_users']} / {summary['tie_users']}")

    # ---------- 6) 选 2 个用户做明细案例 ----------
    names = catalog.build_name_map(
        catalog.load_products()
    )
    win_user = per_user.sort_values("cf_hits", ascending=False).iloc[0]["user_id"]
    most_similar_user = per_user.sort_values("overlap", ascending=False).iloc[0]["user_id"]

    lines = [
        "热门基线 vs 协同过滤（ItemCF）对比实验（阶段三）",
        "=" * 58,
        "",
        "[实验口径]",
        f"  时间切分点          : {split_time}",
        f"  训练期 / 验证期行数 : {len(train)} / {len(valid)}（验证期占 {valid_ratio:.0%}）",
        "  训练数据            : **热门与 CF 都只用训练期数据训练**，不使用任何验证期信息",
        "  真实答案            : 用户在验证期购买、且训练期未买过的商品",
        f"  评测用户            : {len(users)} 个（训练期有行为且验证期有新购买，SEED={SEED} 随机抽样）",
        f"  截断长度 K          : {k}",
        "  相似度口径          : 余弦（用户-商品二值矩阵，列归一化后内积）",
        f"  ItemCF 邻居截断     : {top_similar if top_similar > 0 else '不截断'}",
        "  两路均剔除用户已购商品，保证同口径",
        "",
        "[总体指标]",
        f"  {'方案':<18}{'Precision@K':>14}{'Recall@K':>12}",
        f"  {'热门基线':<18}{summary['pop_precision']:>14.4f}{summary['pop_recall']:>12.4f}",
        f"  {'ItemCF':<18}{summary['cf_precision']:>14.4f}{summary['cf_recall']:>12.4f}",
        "",
        "[两路重合度]",
        f"  平均重叠个数        : {summary['mean_overlap']:.4f} / {k}",
        f"  平均重叠率          : {summary['mean_overlap_rate']:.2%}",
        f"  CF 返回空列表的用户 : {summary['cf_empty_users']} / {len(users)}",
        "",
        "[逐用户胜负]",
        f"  ItemCF 命中更多     : {summary['cf_win_users']} 人",
        f"  热门命中更多        : {summary['pop_win_users']} 人",
        f"  两者打平            : {summary['tie_users']} 人",
        "",
        "[是否像换皮热门？]",
    ]

    overlap_rate = summary["mean_overlap_rate"]
    if overlap_rate >= 0.8:
        verdict = (
            f"  重叠率 {overlap_rate:.2%} 偏高，ItemCF 的推荐列表与热门榜高度重合，"
            "需要警惕它只是在「换皮热门」。"
        )
    elif overlap_rate >= 0.5:
        verdict = (
            f"  重叠率 {overlap_rate:.2%} 属于中等水平：ItemCF 与热门有相当一部分重合，"
            "但它确实给出了不同的候选，不是简单复制热门榜。"
        )
    else:
        verdict = (
            f"  重叠率 {overlap_rate:.2%} 偏低，说明 ItemCF 给出的候选与热门榜差异明显，"
            "个性化程度较高。"
        )
    lines += [
        verdict,
        f"  同时看 Precision@K：热门 {summary['pop_precision']:.4f} vs "
        f"ItemCF {summary['cf_precision']:.4f}，",
        "  结合两者才能判断 CF 是否真的提供了额外信息。",
        "",
        "[为什么评估必须先有热门基线？]",
        "  热门是最难打败的朴素方案：数据长尾很重时，热门商品的命中率天然就高。",
        "  如果不先测热门，就无法判断个性化方案的高指标究竟来自建模能力，",
        "  还是仅仅来自「把最热的东西推给所有人」这一无脑策略。",
        "  有了热门基线，才知道自己的方案到底提升了多少、值不值得上线。",
        "",
        "=" * 58,
        "[案例一] ItemCF 表现最好的用户",
        "",
    ]

    def render_case(user: str, note: str) -> list[str]:
        d = detail[user]
        gt = d["gt"]
        out = [
            f"  用户 {user}（{note}）",
            f"    验证期真实新购买 {len(gt)} 个商品（前 8 个）：",
            f"      {', '.join(sorted(gt)[:8])}",
            f"    热门 Top{k}（命中 {M.overlap_count(d['pop'], gt)} 个）:",
        ]
        out += [f"      {i}. {x} {names.get(x, '未知商品')}" for i, x in enumerate(d["pop"], 1)]
        out += [
            f"    ItemCF Top{k}（命中 {M.overlap_count(d['cf'], gt)} 个）:",
        ]
        out += [f"      {i}. {x} {names.get(x, '未知商品')}" for i, x in enumerate(d["cf"], 1)]
        out += [
            f"    两路重叠 {M.overlap_count(d['pop'], d['cf'])} 个",
            "    ----",
            "",
        ]
        return out

    lines += render_case(str(win_user), "ItemCF 命中数最高")
    lines += ["[案例二] 与热门重合度最高的用户（最像「换皮热门」的情形）", ""]
    lines += render_case(str(most_similar_user), "两路重叠最多")

    (outputs / "stage03_cf_vs_popular.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n案例一（CF 表现最好）用户 {win_user}")
    print(f"案例二（与热门最重合）用户 {most_similar_user}")
    print(f"\n已写出：\n  {outputs / 'stage03_cf_vs_popular.txt'}"
          f"\n  {outputs / 'stage03_cf_vs_popular_peruser.csv'}")


if __name__ == "__main__":
    main()
