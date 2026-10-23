#!/usr/bin/env python3
"""阶段三入口（一）：从 Item2Vec 模型导出向量并构建可查询的向量索引。

产物：
    models/vector_index/vectors.npy   归一化后的向量矩阵（float32）
    models/vector_index/items.json    商品 ID 顺序表（与矩阵行一一对应）
    models/vector_index/meta.json     索引元信息（数量、维度、度量）
    outputs/stage03/stage03_index_build.txt   构建日志与命令行试查结果
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config.settings import PATHS, set_seed, stage_output_dir
from src import item2vec as i2v
from src import product_catalog as catalog
from src.item2vec import load_item2vec
from src.vector_index import ItemVectorIndex

DEMO_TOP_N = 5


def export_vectors(model) -> dict:
    """从 Item2Vec 模型导出 ``{item_id: vector}`` 字典。

    直接复用 ``src.item2vec.export_vectors``（内部走底层矩阵切片），
    不在这里重复实现导出逻辑。
    """
    return i2v.export_vectors(model)


def main() -> None:
    set_seed()
    outputs = stage_output_dir(3)
    outputs.mkdir(parents=True, exist_ok=True)

    # ---------- 导出向量 → 构建索引 → 保存 ----------
    model = load_item2vec(PATHS["item2vec_model"])
    vectors = export_vectors(model)
    print(f"从模型导出向量 : {len(vectors)} 个商品，维度 {len(next(iter(vectors.values())))}")

    started = time.perf_counter()
    index = ItemVectorIndex().build(vectors)
    build_seconds = time.perf_counter() - started

    index.save(PATHS["vector_index_dir"])
    print(f"索引构建完成   : {len(index)} 个商品，耗时 {build_seconds:.2f} 秒")
    print(f"索引目录       : {PATHS['vector_index_dir']}")

    # ---------- 重新加载，确认索引可复用 ----------
    reloaded = ItemVectorIndex().load(PATHS["vector_index_dir"])
    assert len(reloaded) == len(index), "重新加载后索引大小不一致"
    assert reloaded.dim == index.dim, "重新加载后向量维度不一致"
    print(f"重新加载校验   : 通过（{len(reloaded)} 个商品，{reloaded.dim} 维）")

    # ---------- 命令行试查 Top5 ----------
    products = catalog.load_products()
    names = catalog.build_name_map(products)
    categories = catalog.build_category_map(products)

    counts = i2v.item_counts(model)
    demo_ids = i2v.top_items_by_count(model, n=3)

    lines = [
        "向量索引构建日志（阶段三）",
        "",
        "[索引]",
        f"  商品数量            : {len(index)}",
        f"  向量维度            : {index.dim}",
        f"  相似度度量          : cosine（向量已 L2 归一化，内积即余弦）",
        f"  检索方式            : 暴力矩阵乘法（商品规模小，无需 faiss/chromadb）",
        f"  构建耗时            : {build_seconds:.2f} 秒",
        f"  索引目录            : {PATHS['vector_index_dir']}",
        f"  重新加载校验        : 通过",
        "",
        "[命令行试查 Top5]",
    ]

    for demo_id in demo_ids:
        print(f"\n{demo_id} {names.get(demo_id, '未知商品')}")
        lines.append(f"  {demo_id} {names.get(demo_id, '未知商品')}（语料出现 {counts.get(demo_id, 0)} 次）")
        for sim_id, score in index.search(demo_id, topn=DEMO_TOP_N):
            name = names.get(sim_id, "未知商品")
            category = categories.get(sim_id, "未知类目")
            print(f"    {score:.4f}  {sim_id:<10} {name}（{category}）")
            lines.append(f"      {score:.4f}  {sim_id:<10} {name}（{category}）")

    # ---------- 未知 ID 的明确处理 ----------
    try:
        index.search("NOT_EXIST_ID", topn=5)
        unknown_result = "未报错（不符合预期）"
    except KeyError as exc:
        unknown_result = f"抛出 KeyError：{exc}"
    print(f"\n未知商品处理   : {unknown_result}")
    lines += [
        "",
        "[未知商品处理]",
        f"  {unknown_result}",
        "  说明：检索到不存在的商品时明确报错，由上层决定提示方式，而不是静默返回空结果。",
        "",
    ]

    (outputs / "stage03_index_build.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已写出：\n  {PATHS['vector_index_dir']}\n  {outputs / 'stage03_index_build.txt'}")


if __name__ == "__main__":
    main()
