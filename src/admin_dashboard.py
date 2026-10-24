"""管理员用户洞察的数据整理。

这里只做只读聚合，不写账号、购买或反馈数据。页面层负责补充商品图片，
路由层负责权限校验。
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from src.purchase_history import PurchaseHistory


def build_user_insight(
    user: Mapping[str, Any],
    history: PurchaseHistory | None,
    profile_index: Mapping[str, Mapping[str, Any]],
    feedback_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """把账号、购买历史、商品画像和显式反馈汇总成一个用户视图。"""
    customer_id = str(user.get("customer_id") or "")
    history_items = history.items(customer_id) if history and customer_id else []

    category_counts: Counter[str] = Counter()
    purchases: list[dict[str, Any]] = []
    total_orders = 0
    total_qty = 0.0
    reference_spend = 0.0
    has_reference_price = False

    for item in history_items:
        meta = dict(profile_index.get(str(item.item_id), {}))
        category = str(meta.get("category") or "未知类目")
        price = meta.get("price")
        price = round(float(price), 2) if price is not None else None
        amount = round(float(price) * float(item.qty), 2) if price is not None else None
        if amount is not None:
            reference_spend += amount
            has_reference_price = True
        category_counts[category] += 1
        total_orders += int(item.orders)
        total_qty += float(item.qty)
        purchases.append(
            {
                "item_id": str(item.item_id),
                "name": str(meta.get("name") or "未知商品"),
                "category": category,
                "price": price,
                "amount": amount,
                "orders": int(item.orders),
                "qty": round(float(item.qty), 2),
                "first_date": str(item.first_date),
                "last_date": str(item.last_date),
            }
        )

    item_count = len(history_items)
    categories = [
        {
            "category": category,
            "count": count,
            "share": round(count / item_count, 4) if item_count else 0.0,
        }
        for category, count in category_counts.most_common()
    ]

    feedback: list[dict[str, Any]] = []
    likes = 0
    dislikes = 0
    for record in feedback_records:
        item_id = str(record.get("item_id") or "")
        meta = dict(profile_index.get(item_id, {}))
        action = str(record.get("action") or "")
        if action == "like":
            likes += 1
        elif action == "dislike":
            dislikes += 1
        feedback.append(
            {
                "item_id": item_id,
                "name": str(meta.get("name") or "未知商品"),
                "category": str(meta.get("category") or "未知类目"),
                "price": meta.get("price"),
                "action": action,
                "updated_at": str(record.get("updated_at") or ""),
            }
        )

    return {
        "purchases": purchases,
        "categories": categories,
        "feedback": feedback,
        "metrics": {
            "purchase_items": item_count,
            "orders": total_orders,
            "qty": round(total_qty, 2),
            "reference_spend": round(reference_spend, 2) if has_reference_price else None,
            "likes": likes,
            "dislikes": dislikes,
            "feedback_total": len(feedback),
        },
    }
