from __future__ import annotations

import sqlite3

import pandas as pd

from src.admin_dashboard import build_user_insight
from src.purchase_history import PurchaseHistory
from src import user_store
from web import app as web_app


def test_user_role_migration_and_filtering(tmp_path) -> None:
    db_path = tmp_path / "users.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL DEFAULT '',
                salt TEXT NOT NULL,
                password TEXT NOT NULL,
                customer_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
            """
        )

    user_store.init_db(db_path)
    user_store.create_user(db_path, "alice", "secret1", "Alice", "C001")
    user_store.create_user(db_path, "bob", "secret2", "Bob", "C002")
    assert user_store.get_user(db_path, "alice")["role"] == "user"

    assert user_store.set_role(db_path, "alice", "admin")
    assert user_store.count_admins(db_path) == 1
    results = user_store.list_users(db_path, query="C001")
    assert [user["username"] for user in results] == ["alice"]
    assert results[0]["role"] == "admin"


def test_build_user_insight_aggregates_purchases_and_feedback() -> None:
    history = PurchaseHistory(
        pd.DataFrame(
            [
                {
                    "customer_id": "C001",
                    "item_id": "I1",
                    "orders": 2,
                    "qty": 3.0,
                    "first_date": "2026-01-01 10:00",
                    "last_date": "2026-02-01 10:00",
                },
                {
                    "customer_id": "C001",
                    "item_id": "I2",
                    "orders": 1,
                    "qty": 2.0,
                    "first_date": "2026-02-02 10:00",
                    "last_date": "2026-02-02 10:00",
                },
            ]
        )
    )
    insight = build_user_insight(
        {"customer_id": "C001"},
        history,
        {
            "I1": {"name": "收纳盒", "category": "收纳用品", "price": 10.0},
            "I2": {"name": "台灯", "category": "家居装饰", "price": 20.0},
        },
        [
            {"item_id": "I1", "action": "like", "updated_at": "2026-03-01 10:00"},
            {"item_id": "I2", "action": "dislike", "updated_at": "2026-03-02 10:00"},
        ],
    )

    assert insight["metrics"] == {
        "purchase_items": 2,
        "orders": 3,
        "qty": 5.0,
        "reference_spend": 70.0,
        "likes": 1,
        "dislikes": 1,
        "feedback_total": 2,
    }
    assert {row["category"] for row in insight["categories"]} == {
        "收纳用品",
        "家居装饰",
    }
    assert insight["purchases"][0]["item_id"] == "I2"


def test_admin_page_requires_admin_role(monkeypatch) -> None:
    client = web_app.app.test_client()
    with client.session_transaction() as session:
        session["username"] = "normal-user"

    monkeypatch.setattr(
        web_app,
        "current_user",
        lambda: {
            "username": "normal-user",
            "display_name": "普通用户",
            "role": "user",
        },
    )
    assert client.get("/admin").status_code == 403


def test_admin_page_and_csv_export(monkeypatch) -> None:
    client = web_app.app.test_client()
    with client.session_transaction() as session:
        session["username"] = "admin"

    admin = {
        "id": 1,
        "username": "admin",
        "display_name": "管理员",
        "customer_id": "",
        "role": "admin",
        "created_at": "2026-01-01 10:00",
    }
    monkeypatch.setattr(web_app, "current_user", lambda: admin)
    monkeypatch.setattr(
        web_app.user_store,
        "list_users",
        lambda *args, **kwargs: [dict(admin)],
    )
    monkeypatch.setattr(web_app.user_store, "get_user", lambda *args, **kwargs: dict(admin))
    monkeypatch.setattr(web_app.fb, "list_feedback", lambda *args, **kwargs: [])
    monkeypatch.setattr(web_app, "ensure_home", lambda: None)

    response = client.get("/admin")
    assert response.status_code == 200
    assert "用户洞察" in response.get_data(as_text=True)

    exported = client.get("/admin/users/admin/purchases.csv")
    assert exported.status_code == 200
    assert exported.headers["Content-Type"].startswith("text/csv")
    assert "attachment" in exported.headers["Content-Disposition"]
