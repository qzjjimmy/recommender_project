#!/usr/bin/env python3
"""阶段二入口（一）：把干净交易表切成购买序列。

切分规则：同一发票号 InvoiceNo 内的商品构成一条序列（课堂推荐），
序列内按时间从早到晚排序，商品 ID 一律字符串。

产物：
    data/processed/user_sequences.csv       购买序列（供训练 Item2Vec）
    outputs/stage02/stage02_sequence_stats.txt      长度体检报告
    outputs/stage02/stage02_sequence_samples.txt    固定种子抽样的 3 条序列（含中文名）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config.settings import HYPERPARAMS, PATHS, SEED, set_seed, stage_output_dir
from src import product_catalog as catalog
from src import sequences as seq


def main() -> None:
    set_seed()
    outputs = stage_output_dir(2)
    outputs.mkdir(parents=True, exist_ok=True)
    Path(PATHS["user_sequences"]).parent.mkdir(parents=True, exist_ok=True)

    cfg = HYPERPARAMS.get("sequence", {})
    split_by = cfg.get("split_by", seq.SPLIT_BY_INVOICE)
    dedupe = bool(cfg.get("dedupe", True))
    sample_size = int(cfg.get("sample_size", 3))

    # ---------- 步骤 1～2：生成序列 ----------
    clean = pd.read_csv(PATHS["clean_transactions"], dtype=str)
    print(f"干净交易表 : {len(clean)} 行")

    seq_df = seq.build_sequences(clean, split_by=split_by, dedupe=dedupe)
    seq_df.to_csv(PATHS["user_sequences"], index=False, encoding="utf-8")
    print(f"购买序列   : {len(seq_df)} 条（split_by={split_by}, dedupe={dedupe}）")

    # ---------- 长度体检 ----------
    stats = seq.sequence_length_stats(seq_df)
    seq.write_length_report(stats, split_by, outputs / "stage02_sequence_stats.txt")
    print(
        "长度体检   : "
        f"平均={stats['mean']} 中位={stats['p50']} P90={stats['p90']} "
        f"最长={stats['max']}"
    )
    print(
        "长度=1     : "
        f"{stats['len_eq_1']} 条（{stats['len_eq_1_ratio']:.2%}），"
        f"承载商品占 {stats['item_share_in_len1']:.2%}"
    )

    # ---------- 3 条固定种子抽样（供人工判读） ----------
    products = catalog.load_products()
    name_map = seq.build_name_map(products)
    samples = seq.sample_sequences(seq_df, n=sample_size, seed=SEED)
    seq.write_samples_report(samples, name_map, outputs / "stage02_sequence_samples.txt")

    print("\n抽样序列（固定种子 %d）：" % SEED)
    for i, row in enumerate(samples.itertuples(index=False), 1):
        items = seq.parse_items_cell(row.items)
        print(f"  样本{i}: {row.sequence_id} 长度={row.seq_len} 商品={items[:6]}"
              + (" ..." if len(items) > 6 else ""))

    print("\n已写出：")
    for path in (
        PATHS["user_sequences"],
        outputs / "stage02_sequence_stats.txt",
        outputs / "stage02_sequence_samples.txt",
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
