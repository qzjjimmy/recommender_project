"""应用层：账号管理（SQLite）。

安全约定：
    1. 密码用 **PBKDF2-HMAC-SHA256 加盐哈希**存储，数据库里不出现明文；
    2. 校验用 ``hmac.compare_digest`` 做常数时间比较，避免时序侧信道；
    3. 登录态只放在 Flask 的签名 Cookie 会话里，不把用户信息写到 URL。

本项目是教学演示，数据库仅存本地文件（``data/app_users.db``），不涉及真实用户数据。
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

PBKDF2_ROUNDS = 200_000
USERNAME_RE = re.compile(r"^[\w\u4e00-\u9fa5]{3,20}$")
MIN_PASSWORD_LEN = 6
USER_ROLES = ("user", "admin")


# ------------------------------------------------------------------ 基础
def get_connection(db_path: Path) -> sqlite3.Connection:
    """打开（必要时创建）数据库连接。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    """建表（幂等）。"""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT    NOT NULL UNIQUE,
                display_name TEXT    NOT NULL DEFAULT '',
                salt         TEXT    NOT NULL,
                password     TEXT    NOT NULL,
                customer_id  TEXT    NOT NULL DEFAULT '',
                role         TEXT    NOT NULL DEFAULT 'user',
                created_at   TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
            """
        )
        # 旧数据库平滑升级：CREATE TABLE IF NOT EXISTS 不会补齐新列。
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "role" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'"
            )
        conn.commit()


# ------------------------------------------------------------------ 密码
def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return salt.hex(), digest.hex()


def _check_password(password: str, salt_hex: str, expected_hex: str) -> bool:
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), PBKDF2_ROUNDS
    )
    return hmac.compare_digest(digest.hex(), expected_hex)


def validate_credentials(username: str, password: str) -> str | None:
    """校验格式，返回错误信息；通过则返回 ``None``。"""
    if not USERNAME_RE.match(username or ""):
        return "用户名需为 3~20 位中文、字母、数字或下划线"
    if len(password or "") < MIN_PASSWORD_LEN:
        return f"密码至少 {MIN_PASSWORD_LEN} 位"
    return None


# ------------------------------------------------------------------ 增删改查
def create_user(
    db_path: Path,
    username: str,
    password: str,
    display_name: str = "",
    customer_id: str = "",
) -> int:
    """注册新用户，返回自增主键。用户名重复时抛 ``ValueError``。"""
    error = validate_credentials(username, password)
    if error:
        raise ValueError(error)

    salt, password_hash = _hash_password(password)
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, display_name, salt, password, customer_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, display_name or username, salt, password_hash, str(customer_id or "")),
            )
            conn.commit()
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise ValueError("该用户名已被注册") from exc


def verify_user(db_path: Path, username: str, password: str) -> dict[str, Any] | None:
    """校验账号密码；成功返回用户信息，失败返回 ``None``。"""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        return None
    if not _check_password(password, row["salt"], row["password"]):
        return None
    return _row_to_user(row)


def get_user(db_path: Path, username: str) -> dict[str, Any] | None:
    """按用户名取用户信息（不含密码）。"""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return _row_to_user(row) if row else None


def update_customer(db_path: Path, username: str, customer_id: str) -> None:
    """修改用户关联的演示客户 ID（用于个性化推荐）。"""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE users SET customer_id = ? WHERE username = ?",
            (str(customer_id or ""), username),
        )
        conn.commit()


def count_users(db_path: Path) -> int:
    """已注册用户数。"""
    with get_connection(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"])


def list_users(
    db_path: Path,
    *,
    query: str = "",
    role: str | None = None,
) -> list[dict[str, Any]]:
    """列出用户，供管理员目录检索；不返回密码或盐。"""
    if role is not None and role not in USER_ROLES:
        raise ValueError(f"role 必须是 {USER_ROLES} 之一")

    sql = (
        "SELECT id, username, display_name, customer_id, role, created_at "
        "FROM users"
    )
    clauses: list[str] = []
    params: list[Any] = []
    text = str(query or "").strip()
    if text:
        clauses.append(
            "(username LIKE ? OR display_name LIKE ? OR customer_id LIKE ?)"
        )
        pattern = f"%{text}%"
        params.extend([pattern, pattern, pattern])
    if role is not None:
        clauses.append("role = ?")
        params.append(role)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY CASE role WHEN 'admin' THEN 0 ELSE 1 END, created_at DESC, id DESC"

    with get_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_user(row) for row in rows]


def set_role(db_path: Path, username: str, role: str) -> bool:
    """授予或撤销管理员角色；用户不存在时返回 ``False``。"""
    if role not in USER_ROLES:
        raise ValueError(f"role 必须是 {USER_ROLES} 之一")
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "UPDATE users SET role = ? WHERE username = ?",
            (role, str(username)),
        )
        conn.commit()
        return cursor.rowcount > 0


def count_admins(db_path: Path) -> int:
    """当前管理员账号数。"""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM users WHERE role = 'admin'"
        ).fetchone()
    return int(row["n"])


def _row_to_user(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "username": str(row["username"]),
        "display_name": str(row["display_name"]),
        "customer_id": str(row["customer_id"] or ""),
        "role": str(row["role"] or "user"),
        "created_at": str(row["created_at"]),
    }
