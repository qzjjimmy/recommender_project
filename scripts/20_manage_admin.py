#!/usr/bin/env python3
"""授予或撤销管理员角色。

示例：
    python scripts/20_manage_admin.py qzj --grant
    python scripts/20_manage_admin.py qzj --revoke
    python scripts/20_manage_admin.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import PATHS
from src import user_store


def main() -> int:
    parser = argparse.ArgumentParser(description="管理推荐系统管理员角色")
    parser.add_argument("username", nargs="?", help="要授权或撤销的用户名")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--grant", action="store_true", help="授予管理员角色（默认）")
    action.add_argument("--revoke", action="store_true", help="撤销管理员角色")
    parser.add_argument("--list", action="store_true", help="列出全部用户与角色")
    args = parser.parse_args()

    user_store.init_db(PATHS["user_db"])
    if args.list:
        users = user_store.list_users(PATHS["user_db"])
        if not users:
            print("暂无用户。")
            return 0
        print(f"{'用户名':<20} {'角色':<8} {'显示名称'}")
        for user in users:
            print(
                f"{user['username']:<20} {user['role']:<8} "
                f"{user['display_name'] or '-'}"
            )
        return 0

    if not args.username:
        parser.error("请提供用户名，或使用 --list 查看用户")

    user = user_store.get_user(PATHS["user_db"], args.username)
    if user is None:
        print(f"用户不存在：{args.username}", file=sys.stderr)
        return 1

    role = "user" if args.revoke else "admin"
    user_store.set_role(PATHS["user_db"], args.username, role)
    label = "管理员" if role == "admin" else "普通用户"
    print(f"已将 {args.username} 设置为{label}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
