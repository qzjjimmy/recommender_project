"""阶段五：逻辑回归精排（主线必做）。

LR 的价值在于**可解释**：标准化之后，系数就是"该特征每变化一个标准差，
log-odds 变化多少"，能直接说清模型在靠什么做判断。

模型与标准化器**成对保存**：只保存模型而忘记 scaler，
在线打分时特征量纲不一致，离线指标和线上行为就会对不上。
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def train_lr(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """标准化 + 逻辑回归，返回可保存的模型包 ``{model, scaler, feature_cols}``。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[label_col].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)

    model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed, n_jobs=None)
    model.fit(x_scaled, y_train)

    return {
        "kind": "lr",
        "model": model,
        "scaler": scaler,
        "feature_cols": list(feature_cols),
        "label_col": label_col,
    }


def save_lr(model_bundle: dict[str, Any], path: Path) -> None:
    """保存模型包（模型 + 标准化器 + 特征列顺序）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model_bundle, handle)


def load_lr(path: Path) -> dict[str, Any]:
    """加载模型包；文件不存在时给出明确提示。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 LR 精排模型：{path}\n请先运行 scripts/07_train_rankers.py。"
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict_scores(model_bundle: dict[str, Any], feat_df: pd.DataFrame) -> list[float]:
    """对特征表打分，返回"感兴趣"概率。"""
    cols = model_bundle["feature_cols"]
    matrix = feat_df[cols].to_numpy(dtype=float)
    scaler = model_bundle.get("scaler")
    if scaler is not None:
        matrix = scaler.transform(matrix)
    return model_bundle["model"].predict_proba(matrix)[:, 1].tolist()


def coefficients(model_bundle: dict[str, Any]) -> pd.DataFrame:
    """导出特征系数（标准化尺度），按绝对值降序 —— 用于 LR 的可解释性分析。

    标准化后的系数可以直接横向比较：绝对值越大，该特征对判断的影响越强。
    正号表示推高"感兴趣"概率，负号表示压低。
    """
    model = model_bundle["model"]
    weights = model.coef_.ravel()
    return (
        pd.DataFrame(
            {
                "feature": model_bundle["feature_cols"],
                "coefficient": weights,
                "abs_coefficient": np.abs(weights),
                "odds_ratio": np.exp(weights),
            }
        )
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def describe_coefficients(model_bundle: dict[str, Any], top_n: int = 8) -> list[str]:
    """把系数解读成可读的中文句子，供报告直接引用。"""
    labels = {
        "u_events": "用户交互次数（活跃度）",
        "u_items": "用户购买过的不同商品数",
        "u_invoices": "用户发票数（消费频次）",
        "u_avg_price": "用户平均消费单价",
        "u_recency_days": "距上次购买的天数（越小越活跃）",
        "i_users": "商品被多少用户购买（热度）",
        "i_events": "商品交互次数（销量）",
        "i_price": "商品平均单价",
        "i_hot": "商品是否属于全局热门 Top100",
        "m_vec_sim": "用户向量与商品向量的余弦相似度",
        "m_cat_share": "该商品类目占用户历史的比例",
        "m_cat_match": "用户是否买过该商品类目",
        "m_price_ratio": "商品单价 / 用户平均单价",
    }
    table = coefficients(model_bundle).head(top_n)
    lines: list[str] = []
    for _, row in table.iterrows():
        direction = "推高" if row["coefficient"] > 0 else "压低"
        lines.append(
            f"  {row['feature']:<16} 系数 {row['coefficient']:+.4f}"
            f"  → 该特征每增加一个标准差，{direction}「感兴趣」的对数几率"
            f"（{labels.get(row['feature'], '')}）"
        )
    return lines
