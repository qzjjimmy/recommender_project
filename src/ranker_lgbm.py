"""阶段五：LightGBM 精排（对比增强）。

与 LR **完全相同的训练集、验证集与特征列**，"换模型不换候选、不换数据"，
这样两者的指标差异才能归因到模型本身，而不是数据口径。

树模型的优势是自动捕捉非线性与特征交叉；代价是可解释性弱、参数多、易过拟合。
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def train_lgbm(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    *,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """训练 LightGBM 二分类器，返回可保存的模型包。"""
    from lightgbm import LGBMClassifier

    x_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[label_col].to_numpy(dtype=int)

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train)

    return {
        "kind": "lgbm",
        "model": model,
        "scaler": None,  # 树模型不需要标准化
        "feature_cols": list(feature_cols),
        "label_col": label_col,
    }


def save_lgbm(model_bundle: dict[str, Any], path: Path) -> None:
    """保存 LightGBM 模型包。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model_bundle, handle)


def load_lgbm(path: Path) -> dict[str, Any]:
    """加载 LightGBM 模型包；文件不存在时给出明确提示。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 LightGBM 精排模型：{path}\n请先运行 scripts/07_train_rankers.py。"
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict_scores(model_bundle: dict[str, Any], feat_df: pd.DataFrame) -> list[float]:
    """对特征表打分，返回"感兴趣"概率。"""
    cols = model_bundle["feature_cols"]
    matrix = feat_df[cols].to_numpy(dtype=float)
    return model_bundle["model"].predict_proba(matrix)[:, 1].tolist()


def feature_importance(model_bundle: dict[str, Any]) -> pd.DataFrame:
    """特征重要性（按 split 次数归一化），降序排列。"""
    model = model_bundle["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])

    total = float(importances.sum()) or 1.0
    return (
        pd.DataFrame(
            {
                "feature": model_bundle["feature_cols"],
                "importance": importances / total,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def describe_importance(model_bundle: dict[str, Any], top_n: int = 5) -> list[str]:
    """把特征重要性解读成可读句子，供报告直接引用。"""
    labels = {
        "u_events": "用户交互次数",
        "u_items": "用户购买过的不同商品数",
        "u_invoices": "用户发票数",
        "u_avg_price": "用户平均消费单价",
        "u_recency_days": "距上次购买的天数",
        "i_users": "商品热度（购买用户数）",
        "i_events": "商品交互次数",
        "i_price": "商品平均单价",
        "i_hot": "是否全局热门",
        "m_vec_sim": "用户向量与商品向量相似度",
        "m_cat_share": "商品类目占用户历史比例",
        "m_cat_match": "用户是否买过该类目",
        "m_price_ratio": "商品价格 / 用户平均价格",
    }
    lines: list[str] = []
    for _, row in feature_importance(model_bundle).head(top_n).iterrows():
        lines.append(
            f"  {row['feature']:<16} 重要性 {row['importance']:.4f}"
            f"（{labels.get(row['feature'], '')}）"
        )
    return lines
