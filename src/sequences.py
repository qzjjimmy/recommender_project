"""阶段二：购买序列生成。

把清洗后的交易流水切成 Item2Vec 能吃的"购物句子"。

切分规则（课堂推荐，见 README）：
    同一发票号 ``InvoiceNo`` 内的商品构成一条序列；
    序列内按时间从早到晚排序；商品 ID 一律字符串。
一条序列代表**一次结算行为**，即一个购物篮。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

SPLIT_BY_INVOICE = "invoice"
SPLIT_BY_USER = "user"


def parse_items_cell(cell: Any) -> list[str]:
    """把 ``user_sequences.csv`` 中的 items 单元格还原成商品 ID 列表。

    兼容三种写法：已经是 list、JSON/Python 列表字面量、空格分隔字符串。
    """
    if isinstance(cell, (list, tuple)):
        return [str(x) for x in cell]

    text = str(cell).strip()
    if not text or text.lower() == "nan":
        return []
    if text.startswith("["):
        try:
            return [str(x) for x in json.loads(text)]
        except (json.JSONDecodeError, TypeError):
            return [str(x) for x in ast.literal_eval(text)]
    return [x for x in text.split() if x]


def _dedupe_stable(items: Iterable[str]) -> list[str]:
    """按首次出现顺序去重，保持序列内部顺序不变。"""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_sequences(
    clean_df: pd.DataFrame,
    split_by: str = SPLIT_BY_INVOICE,
    dedupe: bool = True,
) -> pd.DataFrame:
    """生成购买序列。

    参数：
        clean_df: 阶段一产出的干净交易表。
        split_by: ``"invoice"`` 同一发票成一条序列（课堂推荐）；
                  ``"user"`` 同一用户成一条序列（按时间排序）。
        dedupe:   同一序列内重复出现的商品是否只保留首次出现。

    返回列：
        sequence_id, user_id, items(JSON 列表), seq_len(去重后长度),
        raw_len(原始行数), start_time, end_time
    """
    if split_by not in (SPLIT_BY_INVOICE, SPLIT_BY_USER):
        raise ValueError(f"split_by 只支持 'invoice' 或 'user'，收到 {split_by!r}")

    df = clean_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    for col in ("InvoiceNo", "StockCode", "CustomerID"):
        df[col] = df[col].astype("string").str.strip()

    # 商品 ID 必须有效且为字符串（85123A 这类编码不能被读坏）
    valid_item = df["StockCode"].notna() & ~df["StockCode"].isin(["", "<NA>", "nan"])
    df = df[valid_item]

    group_key = "InvoiceNo" if split_by == SPLIT_BY_INVOICE else "CustomerID"
    valid_key = df[group_key].notna() & ~df[group_key].isin(["", "<NA>", "nan"])
    df = df[valid_key]

    # 序列内按时间从早到晚；kind="stable" 保证同一时间戳内维持原始行序
    df = df.sort_values("InvoiceDate", kind="stable")

    records: list[dict[str, Any]] = []
    for key, group in df.groupby(group_key, sort=False):
        raw_items = [str(x) for x in group["StockCode"].tolist()]
        items = _dedupe_stable(raw_items) if dedupe else raw_items
        if not items:
            continue
        records.append(
            {
                "sequence_id": str(key),
                "user_id": str(group["CustomerID"].iloc[0]),
                "items": json.dumps(items, ensure_ascii=False),
                "seq_len": len(items),
                "raw_len": len(raw_items),
                "start_time": group["InvoiceDate"].min(),
                "end_time": group["InvoiceDate"].max(),
            }
        )

    return pd.DataFrame.from_records(records)


def sequence_length_stats(seq_df: pd.DataFrame, length_col: str = "seq_len") -> dict[str, Any]:
    """序列长度体检：均值、分位数、长度=1 比例、长度分布直方图。

    长度=1 的序列在 Skip-gram 的窗口内没有任何上下文，无法提供共现信号，
    其占比直接决定向量质量的上限。因此除了序列占比，还统计
    **落在长度=1 序列中的商品数占比**。
    """
    lengths = pd.to_numeric(seq_df[length_col], errors="coerce").dropna()
    total = int(len(lengths))
    if total == 0:
        return {"num_sequences": 0}

    def ratio(mask: pd.Series) -> float:
        return round(float(mask.sum()) / total, 6)

    total_items = int(lengths.sum())
    singleton_items = int(lengths[lengths == 1].sum())

    stats: dict[str, Any] = {
        "num_sequences": total,
        "total_items": total_items,
        "mean": round(float(lengths.mean()), 4),
        "std": round(float(lengths.std()), 4),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "p25": float(lengths.quantile(0.25)),
        "p50": float(lengths.median()),
        "p75": float(lengths.quantile(0.75)),
        "p90": float(lengths.quantile(0.90)),
        "p95": float(lengths.quantile(0.95)),
        "p99": float(lengths.quantile(0.99)),
        "len_eq_1": int((lengths == 1).sum()),
        "len_eq_1_ratio": ratio(lengths == 1),
        "len_le_2_ratio": ratio(lengths <= 2),
        "len_le_3_ratio": ratio(lengths <= 3),
        "len_ge_5_ratio": ratio(lengths >= 5),
        "len_ge_10_ratio": ratio(lengths >= 10),
        "item_share_in_len1": round(singleton_items / total_items, 6) if total_items else 0.0,
        "histogram": {str(k): int((lengths == k).sum()) for k in range(1, 11)},
    }
    stats["histogram"][">=11"] = int((lengths >= 11).sum())

    if "raw_len" in seq_df.columns:
        raw = pd.to_numeric(seq_df["raw_len"], errors="coerce").dropna()
        stats["raw_mean"] = round(float(raw.mean()), 4)
        stats["raw_total_items"] = int(raw.sum())
        stats["dedupe_removed_items"] = int(raw.sum() - total_items)
    return stats


def sample_sequences(seq_df: pd.DataFrame, n: int = 3, seed: int = 42) -> pd.DataFrame:
    """固定种子随机抽 n 条序列，用于人工判读是否像真实购物。"""
    if len(seq_df) <= n:
        return seq_df.copy().reset_index(drop=True)
    return seq_df.sample(n=n, random_state=seed).reset_index(drop=True)


def build_name_map(products: pd.DataFrame) -> dict[str, str]:
    """``StockCode -> 可读名称``（中文优先，退回英文，再退回"未知商品"）。

    具体实现统一放在 :mod:`src.product_catalog`，阶段二与阶段三共用一套口径，
    避免同一个商品在不同阶段显示成不同名称。
    """
    from src.product_catalog import build_name_map as _build_name_map

    return _build_name_map(products)


def describe_items(items: Iterable[str], name_map: Mapping[str, str]) -> list[str]:
    """把商品 ID 列表翻译成 ``ID 名称`` 形式的可读列表。"""
    return [f"{item} {name_map.get(str(item), '未知商品')}" for item in items]


def write_length_report(
    stats: dict[str, Any],
    split_by: str,
    path: Path,
    params: Mapping[str, Any] | None = None,
) -> None:
    """把长度体检结果写成可读文本。"""
    if not stats.get("num_sequences"):
        Path(path).write_text("序列为空，无法体检。\n", encoding="utf-8")
        return

    hist = stats["histogram"]

    def hist_line(label: str) -> str:
        count = hist[label]
        share = count / stats["num_sequences"] * 100
        bar = "#" * max(1, int(round(share / 2))) if count else ""
        return f"    {label:>5} : {count:>7} 条  {share:6.2f}%  {bar}"

    rule = (
        "同一发票内商品构成一条序列"
        if split_by == SPLIT_BY_INVOICE
        else "同一用户内商品构成一条序列"
    )
    lines = [
        "购买序列长度体检报告（阶段二）",
        f"切分规则：{rule}（split_by={split_by}）",
    ]
    if params:
        lines.append(f"其它参数：{dict(params)}")
    lines += [
        "",
        "[总体]",
        f"  序列条数            : {stats['num_sequences']}",
        f"  商品总数（去重后）  : {stats['total_items']}",
        f"  平均长度            : {stats['mean']}",
        f"  标准差              : {stats['std']}",
        f"  最短 / 最长         : {stats['min']} / {stats['max']}",
        "",
        "[分位数]",
        f"  P25 / P50 / P75     : {stats['p25']} / {stats['p50']} / {stats['p75']}",
        f"  P90 / P95 / P99     : {stats['p90']} / {stats['p95']} / {stats['p99']}",
        "",
        "[长度分布]",
    ]
    lines += [hist_line(k) for k in [str(i) for i in range(1, 11)] + [">=11"]]
    lines += [
        "",
        "[长度=1 的影响]",
        f"  长度=1 的序列数     : {stats['len_eq_1']}（占 {stats['len_eq_1_ratio']:.2%}）",
        f"  长度<=2 的序列占比  : {stats['len_le_2_ratio']:.2%}",
        f"  长度<=3 的序列占比  : {stats['len_le_3_ratio']:.2%}",
        f"  长度>=5 的序列占比  : {stats['len_ge_5_ratio']:.2%}",
        f"  长度>=10 的序列占比 : {stats['len_ge_10_ratio']:.2%}",
        f"  落在长度=1 序列中的商品占比: {stats['item_share_in_len1']:.2%}",
    ]

    if "raw_mean" in stats:
        lines += [
            "",
            "[去重前后]",
            f"  去重前平均行数      : {stats['raw_mean']}",
            f"  去重后平均长度      : {stats['mean']}",
            f"  去重掉的重复商品数  : {stats['dedupe_removed_items']}",
        ]

    lines += [
        "",
        "[解读]",
        "  长度=1 的序列在窗口内没有任何邻居，Skip-gram 学不到共现关系；",
        "  其商品占比越高，语料的有效训练信号被稀释得越厉害。",
        "  判断时看'落在长度=1 序列中的商品占比'比看序列占比更准确：",
        "  短序列数量多，但携带的商品可能很少。",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_samples_report(
    samples: pd.DataFrame,
    name_map: Mapping[str, str],
    path: Path,
) -> None:
    """把抽样序列写成可读文本（含商品中文名），供人工判读。"""
    lines = ["抽样序列明细（固定种子，供人工判读）", ""]
    for i, row in enumerate(samples.itertuples(index=False), 1):
        items = parse_items_cell(row.items)
        lines.append(
            f"【样本 {i}】sequence_id={row.sequence_id}  user_id={row.user_id}  "
            f"长度={row.seq_len}（原始行数 {row.raw_len}）"
        )
        lines.append(f"  时间: {row.start_time}")
        for item, text in zip(items, describe_items(items, name_map)):
            lines.append(f"    - {text}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
