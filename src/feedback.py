"""应用层：点赞 / 点踩反馈（SQLite）与偏好合成。

这一块对应指导书里"反馈闭环"的要求：**显式反馈要能影响下一轮排序**。
本项目的做法是：

    1. 点赞 / 点踩按 (用户, 商品) 唯一保存，重复点同一个按钮表示取消；
    2. 点踩的商品直接从结果里剔除；
    3. 点赞商品的向量取平均，与基础用户向量合成"短期偏好向量"，
       让"与点赞商品相似"的候选获得加分 —— 这就是反馈对排序的实际影响。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ACTIONS = ("like", "dislike")
ACTION_LABELS = {"like": "喜欢", "dislike": "不喜欢"}


def init_db(db_path: Path) -> None:
    """建表（幂等）。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT NOT NULL,
                item_id    TEXT NOT NULL,
                action     TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                UNIQUE (username, item_id)
            )
            """
        )
        conn.commit()


def set_feedback(db_path: Path, username: str, item_id: str, action: str) -> str | None:
    """记录一次反馈。

    - 首次点击 → 写入该动作；
    - 再次点击同一个动作 → 视为取消，删除记录；
    - 点击另一个动作 → 覆盖（不喜欢 ↔ 喜欢）。

    返回该商品最终的反馈状态（``"like"`` / ``"dislike"`` / ``None`` 表示已取消）。
    """
    if action not in ACTIONS:
        raise ValueError(f"action 必须是 {ACTIONS} 之一，收到 {action!r}")
    username, item_id = str(username), str(item_id)

    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT action FROM feedback WHERE username = ? AND item_id = ?",
            (username, item_id),
        ).fetchone()
        current = row[0] if row else None

        if current == action:  # 再点一次 = 取消
            conn.execute(
                "DELETE FROM feedback WHERE username = ? AND item_id = ?", (username, item_id)
            )
            final = None
        elif current is None:
            conn.execute(
                "INSERT INTO feedback (username, item_id, action) VALUES (?, ?, ?)",
                (username, item_id, action),
            )
            final = action
        else:
            conn.execute(
                "UPDATE feedback SET action = ?, updated_at = datetime('now', 'localtime') "
                "WHERE username = ? AND item_id = ?",
                (action, username, item_id),
            )
            final = action
        conn.commit()
    return final


def get_feedback(db_path: Path, username: str) -> dict[str, str]:
    """返回 ``{商品ID: 动作}``。"""
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT item_id, action FROM feedback WHERE username = ?", (str(username),)
        ).fetchall()
    return {str(item): str(action) for item, action in rows}


def split_feedback(mapping: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """把 ``{商品ID: 动作}`` 拆成 (点赞列表, 点踩列表)。"""
    liked = [k for k, v in mapping.items() if v == "like"]
    disliked = [k for k, v in mapping.items() if v == "dislike"]
    return liked, disliked


def clear_feedback(db_path: Path, username: str, item_id: str | None = None) -> int:
    """清空该用户的反馈；指定 ``item_id`` 时只清一条。返回删除条数。"""
    with sqlite3.connect(str(db_path)) as conn:
        if item_id is None:
            cursor = conn.execute("DELETE FROM feedback WHERE username = ?", (str(username),))
        else:
            cursor = conn.execute(
                "DELETE FROM feedback WHERE username = ? AND item_id = ?",
                (str(username), str(item_id)),
            )
        conn.commit()
        return int(cursor.rowcount)


def list_feedback(
    db_path: Path, username: str, action: str | None = None
) -> list[dict[str, Any]]:
    """列出该用户的反馈记录（可按动作过滤），最近更新的排在最前。

    ``get_feedback`` 返回的 ``{商品ID: 动作}`` 字典够用来排序，但**不带时间、也不
    保证顺序**；页面上"不喜欢的商品"面板要按"最近点踩的排前面"来展示，所以需要
    这个按时间倒序的列表接口。
    """
    if action is not None and action not in ACTIONS:
        raise ValueError(f"action 必须是 {ACTIONS} 之一，收到 {action!r}")

    sql = "SELECT item_id, action, updated_at FROM feedback WHERE username = ?"
    params: list[Any] = [str(username)]
    if action is not None:
        sql += " AND action = ?"
        params.append(str(action))
    sql += " ORDER BY updated_at DESC, id DESC"

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {"item_id": str(item_id), "action": str(act), "updated_at": str(updated_at)}
        for item_id, act, updated_at in rows
    ]


def clear_feedback_by_action(db_path: Path, username: str, action: str) -> int:
    """一次清空某类反馈（页面上"清空全部不喜欢"用）。返回删除条数。"""
    if action not in ACTIONS:
        raise ValueError(f"action 必须是 {ACTIONS} 之一，收到 {action!r}")
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute(
            "DELETE FROM feedback WHERE username = ? AND action = ?",
            (str(username), str(action)),
        )
        conn.commit()
        return int(cursor.rowcount)


def summarize(mapping: Mapping[str, str]) -> dict[str, Any]:
    """反馈统计，用于界面回显。"""
    liked, disliked = split_feedback(mapping)
    return {"liked": len(liked), "disliked": len(disliked), "total": len(mapping)}


# ------------------------------------------------------------------ 偏好合成
def mean_vector(item_ids: Iterable[str], index) -> np.ndarray | None:
    """取一组商品向量的平均并归一化；全部不在索引里时返回 ``None``。"""
    vectors = [
        np.asarray(index.get_vector(str(i)), dtype=np.float32)
        for i in item_ids
        if index is not None and index.contains(str(i))
    ]
    if not vectors:
        return None
    mean = np.mean(np.vstack(vectors), axis=0)
    norm = float(np.linalg.norm(mean))
    return None if norm == 0.0 else (mean / norm).astype(np.float32)


def build_preference_vector(
    liked: Sequence[str],
    disliked: Sequence[str],
    index,
    base_vector: np.ndarray | None = None,
    feedback_weight: float = 0.8,
    dislike_penalty: float = 0.35,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """合成"基础兴趣 + 短期反馈"的偏好向量。

    ``偏好向量 = normalize( 基础向量 + w·点赞均值 − penalty·点踩均值 )``

    返回 ``(向量, 说明信息)``；没有任何可用信号时向量为 ``None``。
    """
    liked_vec = mean_vector(liked, index)
    disliked_vec = mean_vector(disliked, index)

    parts: list[np.ndarray] = []
    if base_vector is not None:
        base = np.asarray(base_vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(base))
        if norm > 0:
            parts.append(base / norm)
    if liked_vec is not None:
        parts.append(liked_vec * float(feedback_weight))
    if disliked_vec is not None:
        parts.append(-disliked_vec * float(dislike_penalty))

    info = {
        "has_base": base_vector is not None,
        "liked_used": len([i for i in liked if index is not None and index.contains(str(i))]),
        "disliked_used": len([i for i in disliked if index is not None and index.contains(str(i))]),
        "feedback_weight": feedback_weight,
        "dislike_penalty": dislike_penalty,
    }
    if not parts:
        return None, info

    combined = np.sum(np.vstack(parts), axis=0)
    norm = float(np.linalg.norm(combined))
    if norm == 0.0:
        return None, info
    return (combined / norm).astype(np.float32), info
