"""阶段四：多路召回与融合为召回集。

三条召回路（每路职责不同，互补覆盖）：

    popular   热门路   —— 兜底、保互动；冷启动用户的主要来源
    similar   相似扩展路 —— 对用户历史商品做相似扩展，捕捉"搭配"
    user_vec  用户向量路 —— 用户向量近邻，捕捉"兴趣"

融合规则（顺序固定，写进 README 与文档）：
    按路依次取商品，跳过已经出现过的 ID（去重），每路最多取 ``quota`` 个，
    总长度不超过 ``final_k``。

注意：**召回集是精排的输入，不是最终推荐结论**。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# 固定的路顺序：兜底 → 搭配 → 兴趣
ROAD_ORDER = ("popular", "similar", "user_vec")
ROAD_LABELS = {"popular": "热门路", "similar": "相似扩展路", "user_vec": "用户向量路"}


# ------------------------------------------------------------------ 三条路
def popular_road(popular, user_id: str, k: int) -> list[str]:
    """热门路：返回全局热门 TopK（PopularRecommender 内部已剔除该用户已购）。"""
    return popular.recommend(user_id, k=k)


def similar_road(
    index,
    history_items: Sequence[str],
    k: int,
    history_n: int = 20,
    per_item: int = 5,
    purchased: Iterable[str] | None = None,
) -> list[str]:
    """相似扩展路：对用户最近 history_n 个历史商品各取 per_item 个相似商品，再合并。

    相似商品来自阶段三的向量索引（同一套余弦口径）。
    按"最近的历史商品优先"的顺序合并，保证融合时先拿到近期兴趣的扩展结果。
    """
    purchased_set = {str(x) for x in (purchased or set())}
    recent = list(history_items)[-history_n:] if history_n > 0 else list(history_items)

    out: list[str] = []
    seen: set[str] = set()
    for item_id in reversed(recent):  # 最近的在最后，反转后优先
        if not index.contains(item_id):
            continue
        for sim_id, _ in index.search(item_id, topn=per_item):
            if sim_id in seen or sim_id in purchased_set:
                continue
            seen.add(sim_id)
            out.append(sim_id)
            if len(out) >= k:
                return out
    return out


def user_vector_road(
    index,
    user_vec,
    purchased: Iterable[str] | None = None,
    k: int = 20,
) -> list[str]:
    """用户向量路：用用户向量在商品向量中做近邻检索，剔除已购。"""
    if user_vec is None:
        return []
    purchased_set = {str(x) for x in (purchased or set())}
    total = len(index)
    pairs = index.search_by_vector(user_vec, topn=min(total, k + len(purchased_set)))
    return [item_id for item_id, _ in pairs if item_id not in purchased_set][:k]


# -------------------------------------------------------------------- 融合
def fuse_recall(
    road_results: Mapping[str, list[str]],
    quotas: Mapping[str, int],
    final_k: int,
) -> list[str]:
    """按路取配额、跨路去重，总长度不超过 final_k。

    路顺序取 ``ROAD_ORDER`` 中出现的顺序，未列出的路按传入顺序追加。
    """
    if final_k <= 0:
        raise ValueError(f"final_k 必须为正整数，收到 {final_k}")

    ordered_roads = [r for r in ROAD_ORDER if r in road_results]
    ordered_roads += [r for r in road_results if r not in ordered_roads]

    out: list[str] = []
    seen: set[str] = set()
    for road in ordered_roads:
        quota = int(quotas.get(road, 0))
        if quota <= 0:
            continue
        taken = 0
        for item_id in road_results[road]:
            item_id = str(item_id)
            if item_id in seen:
                continue
            seen.add(item_id)
            out.append(item_id)
            taken += 1
            if taken >= quota or len(out) >= final_k:
                break
        if len(out) >= final_k:
            break
    return out


def fuse_recall_with_sources(
    road_results: Mapping[str, list[str]],
    quotas: Mapping[str, int],
    final_k: int,
) -> list[tuple[str, str]]:
    """与 :func:`fuse_recall` 同样的规则，但额外保留每个商品的**来源路**。

    多路候选去重后保留来源，是阶段六做"推荐理由"与来源分析的基础。
    """
    if final_k <= 0:
        raise ValueError(f"final_k 必须为正整数，收到 {final_k}")

    ordered_roads = [r for r in ROAD_ORDER if r in road_results]
    ordered_roads += [r for r in road_results if r not in ordered_roads]

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for road in ordered_roads:
        quota = int(quotas.get(road, 0))
        if quota <= 0:
            continue
        taken = 0
        for item_id in road_results[road]:
            item_id = str(item_id)
            if item_id in seen:
                continue
            seen.add(item_id)
            out.append((item_id, road))
            taken += 1
            if taken >= quota or len(out) >= final_k:
                break
        if len(out) >= final_k:
            break
    return out


def road_contribution(items_with_sources: Sequence[tuple[str, str]]) -> dict[str, int]:
    """统计融合后每一路实际贡献了多少个商品（去重之后）。"""
    counts: dict[str, int] = {}
    for _, road in items_with_sources:
        counts[road] = counts.get(road, 0) + 1
    return counts


# ------------------------------------------------------------------ 评估
def evaluate_recall(
    recall_map: Mapping[str, Sequence[str]],
    ground_truth: Mapping[str, set[str]],
    hot_items: set[str],
    k: int | None = None,
) -> dict[str, Any]:
    """召回质量评估：覆盖率（Recall@K）、命中用户比例、热门占比。

    参数：
        recall_map:   ``{用户ID: 召回候选列表}``
        ground_truth: ``{用户ID: 验证期真实新购买集合}``
        hot_items:    全局热门商品集合（用于计算热门占比）
        k:            只评估前 K 个候选；``None`` 表示用全部候选

    返回指标字典，其中：
        ``mean_recall``      用户级平均覆盖率 = 平均(|真实 ∩ 候选| / |真实|)
        ``hit_user_ratio``   至少命中一个真实商品的用户比例
        ``hot_ratio``        候选里属于全局热门的比例
        ``item_coverage``    商品级覆盖率 = |并集真实 ∩ 并集候选| / |并集真实|
    """
    users = [u for u in recall_map if u in ground_truth]
    if not users:
        return {"users": 0}

    recalls: list[float] = []
    hits = 0
    lengths: list[int] = []
    hot_hits = 0
    total_items = 0
    gt_union: set[str] = set()
    rec_union: set[str] = set()

    for user in users:
        candidates = list(recall_map[user])
        if k is not None:
            candidates = candidates[:k]
        gt = ground_truth[user]
        hit = len(set(candidates) & gt)

        recalls.append(hit / len(gt) if gt else 0.0)
        hits += 1 if hit > 0 else 0
        lengths.append(len(candidates))
        hot_hits += sum(1 for item in candidates if item in hot_items)
        total_items += len(candidates)
        gt_union |= gt
        rec_union |= set(candidates)

    return {
        "users": len(users),
        "mean_recall": float(np.mean(recalls)),
        "hit_user_ratio": hits / len(users),
        "mean_len": float(np.mean(lengths)),
        "hot_ratio": hot_hits / total_items if total_items else 0.0,
        "item_coverage": len(gt_union & rec_union) / len(gt_union) if gt_union else 0.0,
        "gt_union_size": len(gt_union),
        "rec_union_size": len(rec_union),
    }


def write_recall_report(
    path: Path,
    *,
    split_info: Mapping[str, Any],
    metrics_default: Mapping[str, Any],
    metrics_variant: Mapping[str, Any] | None,
    warm_cold: Mapping[str, Mapping[str, Any]],
    contributions: Mapping[str, float],
    quotas_default: Mapping[str, int],
    quotas_variant: Mapping[str, int] | None,
    final_k: int,
    sample_users: int,
    cold_users: int,
) -> None:
    """把召回评估结果写成可读报告。"""

    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    lines = [
        "召回集评估报告（阶段四）",
        "=" * 58,
        "",
        "[构造口径]",
        f"  时间切分点          : {split_info['split_time']}",
        f"  训练期 / 验证期行数 : {split_info['train_rows']} / {split_info['valid_rows']}",
        "  用户向量            : 只用**训练期**历史商品向量求平均（不用验证期数据）",
        "  召回各路            : 也只用训练期统计（热门榜、相似索引均基于训练期口径）",
        "  真实答案            : 用户在验证期购买、且训练期未买过的商品",
        f"  用户集              : 验证期有新购买的用户中固定种子抽 {sample_users} 个"
        f"（含冷启动 {cold_users} 个）",
        f"  final_k             : {final_k}",
        "",
        "[融合规则]",
        "  按「热门路 → 相似扩展路 → 用户向量路」依次取商品；",
        "  跨路去重（跳过已出现的 ID）；每路最多取配额个；总长不超过 final_k。",
        "",
        "[默认配额结果]",
        f"  配额                : {dict(quotas_default)}",
        f"  评测用户数          : {metrics_default['users']}",
        f"  平均候选数          : {metrics_default['mean_len']:.2f}",
        f"  平均覆盖率 Recall   : {pct(metrics_default['mean_recall'])}",
        f"  至少命中一个的用户  : {pct(metrics_default['hit_user_ratio'])}",
        f"  商品级覆盖率        : {pct(metrics_default['item_coverage'])}"
        f"（真实商品 {metrics_default['gt_union_size']} 个）",
        f"  热门占比            : {pct(metrics_default['hot_ratio'])}",
        "",
        "[每一路的实际贡献（去重后平均每用户）]",
    ]
    for road, value in contributions.items():
        lines.append(f"  {ROAD_LABELS.get(road, road):<10}: {value:.2f} 个/用户")

    if metrics_variant is not None and quotas_variant is not None:
        lines += [
            "",
            "[配额对比]",
            f"  方案 A（默认）      : {dict(quotas_default)}",
            f"  方案 B（偏个性化）  : {dict(quotas_variant)}",
            "",
            f"  {'指标':<22}{'方案 A':>12}{'方案 B':>12}",
            f"  {'平均候选数':<22}{metrics_default['mean_len']:>12.2f}"
            f"{metrics_variant['mean_len']:>12.2f}",
            f"  {'平均覆盖率':<22}{pct(metrics_default['mean_recall']):>12}"
            f"{pct(metrics_variant['mean_recall']):>12}",
            f"  {'至少命中一个用户比例':<20}{pct(metrics_default['hit_user_ratio']):>12}"
            f"{pct(metrics_variant['hit_user_ratio']):>12}",
            f"  {'热门占比':<22}{pct(metrics_default['hot_ratio']):>12}"
            f"{pct(metrics_variant['hot_ratio']):>12}",
            "",
            "  解读：提高用户向量路配额、压缩热门路配额后，热门占比下降、覆盖率随之变化，",
            "  说明配额是「保互动」与「保个性化」之间的旋钮，不是越大越好。",
        ]

    lines += [
        "",
        "[有历史用户 vs 冷启动用户]",
        f"  {'群体':<14}{'人数':>6}{'平均候选数':>12}{'平均覆盖率':>12}{'热门占比':>12}",
    ]
    for label, stat in warm_cold.items():
        lines.append(
            f"  {label:<14}{stat['users']:>6}{stat['mean_len']:>12.2f}"
            f"{pct(stat['mean_recall']):>12}{pct(stat['hot_ratio']):>12}"
        )
    lines += [
        "",
        "[结论与问题]",
        "  1) 召回集大小与覆盖率正相关，但候选过多会稀释阶段五精排的训练信号；",
        "  2) 热门占比反映召回集的同质化程度：占比过高说明候选缺乏个性化；",
        "  3) 冷启动用户没有历史商品，用户向量路与相似扩展路都会返回空，",
        "     只能靠热门路兜底，因此其热门占比必然明显高于有历史用户。",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
