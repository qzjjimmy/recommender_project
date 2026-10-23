"""阶段二：Item2Vec（Word2Vec / Skip-gram）。

把每条购买序列当作一个"句子"，用 Skip-gram 学习商品向量，
使经常一起出现在同一个购物篮里的商品，在向量空间中彼此接近。

超参一律从 ``config/settings.py`` 的 ``HYPERPARAMS["item2vec"]`` 读取。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:  # 允许脱离项目根目录单独导入本模块
    from config.settings import SEED
except Exception:  # pragma: no cover
    SEED = 42


def train_item2vec(sequences: list[list[str]], params: dict[str, Any]):
    """训练商品向量模型，返回 ``gensim.models.Word2Vec`` 对象。

    参数：
        sequences: 语料，每个元素是一条商品 ID 列表（一个购物篮）。
        params:    超参数字典，键缺失时使用下面的默认值。

    说明：``workers=1`` 时训练完全可复现（多线程下累加顺序不定，
    即使固定 seed 也可能产生微小差异）。
    """
    from gensim.models import Word2Vec

    cfg = dict(params or {})
    corpus = [list(s) for s in sequences if len(s) > 0]

    model = Word2Vec(
        sentences=corpus,
        vector_size=int(cfg.get("vector_size", 64)),
        window=int(cfg.get("window", 5)),
        min_count=int(cfg.get("min_count", 5)),
        sg=int(cfg.get("sg", 1)),
        negative=int(cfg.get("negative", 5)),
        sample=float(cfg.get("sample", 0.001)),
        epochs=int(cfg.get("epochs", 10)),
        workers=int(cfg.get("workers", 1)),
        seed=int(cfg.get("seed", SEED)),
    )
    return model


def save_item2vec(model, path: Path) -> None:
    """把模型持久化到 ``models/``，之后无需每次重训。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def load_item2vec(path: Path):
    """加载已保存的模型；文件不存在时给出明确提示。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 Item2Vec 模型：{path}\n请先运行 scripts/03_train_item2vec.py 训练模型。"
        )
    from gensim.models import Word2Vec

    return Word2Vec.load(str(path))


def similar_items(model, item_id: str, topn: int = 5) -> list[tuple[str, float]]:
    """查询相似商品，返回 ``[(item_id, score), ...]``（不含自身）。

    多取一个候选再过滤掉自身，避免"自己排第一"占掉名额；
    商品不在词表时抛出带说明的 ``KeyError``，而不是返回空结果或崩溃。
    """
    item_id = str(item_id)
    if item_id not in model.wv:
        raise KeyError(
            f"商品 {item_id!r} 不在词表中：可能被 min_count 过滤，"
            f"或该商品未出现在训练语料里。"
        )

    pairs = model.wv.most_similar(item_id, topn=topn + 1)
    return [(str(word), float(score)) for word, score in pairs if str(word) != item_id][:topn]


def vocab_size(model) -> int:
    """词表大小（通过 min_count 过滤后真正参与训练的商品数）。"""
    return int(len(model.wv))


def item_counts(model) -> dict[str, int]:
    """``商品ID -> 在语料中出现的次数``，用于挑高频商品做查询样例。"""
    wv = model.wv
    return {str(k): int(wv.get_vecattr(k, "count")) for k in wv.index_to_key}


def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    """手算余弦相似度，用于验证"方向越近、分数越高"。

    与 ``most_similar`` 返回的分数应当一致（gensim 用的就是余弦相似度）。
    """
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def export_vectors(model) -> dict[str, np.ndarray]:
    """导出 ``{item_id: 向量}`` 字典，供阶段三构建向量索引使用。

    直接从底层矩阵切片，比逐个 ``model.wv[key]`` 快得多。
    """
    wv = model.wv
    matrix = np.asarray(wv.vectors, dtype=np.float32)
    return {str(key): matrix[i] for i, key in enumerate(wv.index_to_key)}


def top_items_by_count(model, n: int = 5) -> list[str]:
    """按语料出现次数取前 N 个商品，作为相似查询的样例商品。"""
    counts = item_counts(model)
    return [k for k, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n]]
