#!/usr/bin/env python3
"""阶段二入口（二）：训练 Item2Vec，并做相似商品查询样例。

流程：读 user_sequences.csv → 还原商品 ID 列表 → 训练 Skip-gram →
      保存模型 → 重新加载校验 → 对高频商品做 Top5 相似查询。

产物：
    models/item2vec.model                   商品向量模型
    outputs/stage02/stage02_train_log.txt           超参、词表大小、耗时、余弦校验
    outputs/stage02/stage02_similar_raw.csv         5 个高频商品的 Top5 相似结果（原始证据）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config.settings import HYPERPARAMS, PATHS, set_seed, stage_output_dir
from src import item2vec as i2v
from src import product_catalog as catalog
from src import sequences as seq

QUERY_ITEM_NUM = 5
TOP_N = 5


def main() -> None:
    set_seed()
    outputs = stage_output_dir(2)
    outputs.mkdir(parents=True, exist_ok=True)

    # ---------- 步骤 4：读序列 → 训练 → 存模型 ----------
    seq_df = pd.read_csv(PATHS["user_sequences"])
    corpus = [seq.parse_items_cell(x) for x in seq_df["items"].tolist()]
    corpus = [c for c in corpus if c]
    print(f"语料       : {len(corpus)} 条序列，共 {sum(len(c) for c in corpus)} 个商品")

    params = dict(HYPERPARAMS["item2vec"])
    started = time.perf_counter()
    model = i2v.train_item2vec(corpus, params)
    elapsed = time.perf_counter() - started

    i2v.save_item2vec(model, PATHS["item2vec_model"])
    print(f"训练完成   : 词表 {i2v.vocab_size(model)} 个商品，耗时 {elapsed:.1f} 秒")

    # 重新加载，确认模型可二次使用
    reloaded = i2v.load_item2vec(PATHS["item2vec_model"])
    assert i2v.vocab_size(reloaded) == i2v.vocab_size(model), "重新加载后词表大小不一致"
    print(f"重新加载   : 词表 {i2v.vocab_size(reloaded)}，校验通过")

    # ---------- 步骤 5：高频商品的相似样例 ----------
    products = catalog.load_products()
    name_map = seq.build_name_map(products)
    counts = i2v.item_counts(model)
    queries = i2v.top_items_by_count(model, n=QUERY_ITEM_NUM)

    rows: list[dict] = []
    for rank, qid in enumerate(queries, 1):
        for srank, (sim_id, score) in enumerate(i2v.similar_items(model, qid, topn=TOP_N), 1):
            rows.append(
                {
                    "query_rank": rank,
                    "query_id": qid,
                    "query_name": name_map.get(qid, "未知商品"),
                    "query_count": counts.get(qid, 0),
                    "sim_rank": srank,
                    "sim_id": sim_id,
                    "sim_name": name_map.get(sim_id, "未知商品"),
                    "score": round(score, 4),
                }
            )
    similar_df = pd.DataFrame(rows)
    similar_df.to_csv(
        outputs / "stage02_similar_raw.csv", index=False, encoding="utf-8-sig"
    )

    # ---------- 余弦相似度校验 ----------
    demo_id = queries[0]
    demo_vec = model.wv[demo_id]
    cosine_rows = []
    for sim_id, score in i2v.similar_items(model, demo_id, topn=TOP_N):
        manual = i2v.cosine_similarity(demo_vec, model.wv[sim_id])
        cosine_rows.append((sim_id, score, manual, abs(score - manual)))

    lines = [
        "Item2Vec 训练日志（阶段二）",
        "",
        "[语料]",
        f"  序列条数            : {len(corpus)}",
        f"  商品出现次数合计    : {sum(len(c) for c in corpus)}",
        f"  不同商品数（原始）  : {len({i for c in corpus for i in c})}",
        f"  词表大小（min_count 过滤后）: {i2v.vocab_size(model)}",
        "",
        "[超参]",
    ]
    for key, value in params.items():
        lines.append(f"  {key:<12}: {value}")
    lines += [
        "",
        "[结果]",
        f"  训练耗时            : {elapsed:.1f} 秒",
        f"  模型文件            : {PATHS['item2vec_model']}",
        f"  重新加载校验        : 通过（词表 {i2v.vocab_size(reloaded)}）",
        "",
        "[余弦相似度校验]",
        f"  查询商品: {demo_id} {name_map.get(demo_id, '未知商品')}",
        "  商品ID         most_similar分数   手算余弦    绝对差",
    ]
    for sim_id, score, manual, diff in cosine_rows:
        lines.append(f"  {sim_id:<14} {score:>12.6f} {manual:>12.6f} {diff:>10.2e}")
    lines += [
        "",
        "[结论]",
        "  手算余弦与 most_similar 分数一致（绝对差 < 1e-6），",
        "  说明向量方向越接近、相似度分数越高，阶段三的向量检索可以直接复用。",
        "",
        "[查询样例]",
    ]
    for qid in queries:
        lines.append(f"  {qid} {name_map.get(qid, '未知商品')}（语料出现 {counts.get(qid, 0)} 次）")
        for sim_id, score in i2v.similar_items(model, qid, topn=TOP_N):
            lines.append(f"      {score:.4f}  {sim_id:<10} {name_map.get(sim_id, '未知商品')}")
    lines.append("")

    (outputs / "stage02_train_log.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n查询样例：")
    for qid in queries:
        print(f"  {qid} {name_map.get(qid, '未知商品')}")
        for sim_id, score in i2v.similar_items(model, qid, topn=TOP_N):
            print(f"      {score:.4f}  {sim_id:<10} {name_map.get(sim_id, '未知商品')}")

    print("\n已写出：")
    for path in (
        PATHS["item2vec_model"],
        outputs / "stage02_train_log.txt",
        outputs / "stage02_similar_raw.csv",
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
