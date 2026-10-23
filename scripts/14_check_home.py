#!/usr/bin/env python3
"""推荐首页自检：模块层 + 路由层各测一遍，结果写到 outputs/stage14/stage14_home_check.txt。

分两段测：

    A. 模块层：直接调用 ``HomeRecommender``，检查两个板块是否都能产出结果、
       推荐理由是否带来源、月份切换是否真的改变了档期。
    B. 路由层：用 Flask 的 ``test_client`` 打 ``/home`` 与 ``/api/home``，
       检查匿名/登录两种身份下的页面内容，以及非法月份的处理。

运行（项目根目录）：
    python scripts/14_check_home.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import PATHS, stage_output_dir
from src import user_store
from src.home_recommend import HomeRecommender

DEMO_USER = "demo_user"
DEMO_PASS = "demo123456"

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def check(label: str, ok: bool, detail: str = "") -> bool:
    say(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" —— {detail}" if detail else ""))
    return ok


def ensure_demo_user() -> str:
    """确保有一个可登录的演示账号；返回它的关联客户 ID。"""
    user = user_store.get_user(PATHS["user_db"], DEMO_USER)
    if user is None:
        user_store.create_user(
            PATHS["user_db"], DEMO_USER, DEMO_PASS, "演示用户", "14897"
        )
        user = user_store.get_user(PATHS["user_db"], DEMO_USER)
    return str(user["customer_id"]) if user else ""


def module_checks(rec: HomeRecommender, customer_id: str) -> bool:
    say("A. 模块层：HomeRecommender")
    ok = True

    sep = rec.build(username=None, customer_id=customer_id, month=9)
    ok &= check("9 月档期 = 开学季 + 秋季换季",
                [t["name"] for t in sep["themes"]] == ["9 月开学季", "秋季换季"],
                str([t["name"] for t in sep["themes"]]))

    for_you = sep["sections"][0]
    ok &= check("板块一基于购买过商品产出推荐", len(for_you["rows"]) > 0,
                f"{len(for_you['rows'])} 条，历史 {sep['history_count']} 件")
    ok &= check("板块一每条都带来源商品",
                all(r.get("origin_name") for r in for_you["rows"]),
                "；".join(f"{r['name']} ← {r['origin_name']}" for r in for_you["rows"][:2]))
    purchased = rec.history.item_ids(customer_id)
    ok &= check("板块一不推已经买过的商品",
                all(r["item_id"] not in purchased for r in for_you["rows"]))
    cats = [r["category"] for r in for_you["rows"]]
    ok &= check("板块一做过类目多样化（同一类目 ≤ 2 条）",
                all(cats.count(c) <= 2 for c in set(cats)), str(cats))

    for section in sep["sections"][1:]:
        ok &= check(f"板块「{section['title']}」有商品", len(section["rows"]) > 0,
                    f"{len(section['rows'])} 条，类目池 {'/'.join(section['categories'][:4])}…")
        ok &= check(f"板块「{section['title']}」标出了档期类目的实测指数",
                    bool(section["evidence_rows"]),
                    "、".join(
                        f"{c['category']}={c['lift'] if c['lift'] is not None else '样本不足'}"
                        for c in section["evidence_rows"][:4]
                    ))

    dec = rec.build(username=None, customer_id=customer_id, month=12)
    ok &= check("换到 12 月档期变成圣诞礼赠",
                [t["name"] for t in dec["themes"]] == ["圣诞礼赠", "年终迎新"],
                str([t["name"] for t in dec["themes"]]))

    cold = rec.build(username=None, customer_id="不存在的客户", month=9)
    ok &= check("没有购买历史时不报错、板块一为空并给出提示",
                cold["sections"][0]["rows"] == [] and bool(cold["sections"][0]["empty"]))
    return bool(ok)


def route_checks() -> bool:
    say("\nB. 路由层：Flask test_client")
    import web.app as webapp

    ok = True
    client = webapp.app.test_client()

    resp = client.get("/home")
    html = resp.get_data(as_text=True)
    ok &= check("匿名访问 /home 返回 200", resp.status_code == 200, f"{len(html)} 字节")
    ok &= check("匿名页面含当月档期与登录引导",
                "档期" in html and "登录 / 注册" in html)

    customer_id = ensure_demo_user()
    resp = client.post(
        "/login",
        data={"username": DEMO_USER, "password": DEMO_PASS},
        follow_redirects=False,
    )
    ok &= check("登录后跳转到 /home（不再是 /search）",
                resp.status_code == 302 and "/home" in resp.headers.get("Location", ""),
                resp.headers.get("Location", ""))

    html = client.get("/home?month=9").get_data(as_text=True)
    ok &= check("登录后 /home 出现「猜你还会买」的卡片", 'pcard' in html and "推荐依据" in html,
                f"关联客户 {customer_id}")
    ok &= check("登录后卡片带点赞 / 点踩按钮", 'class="fb"' in html)

    data = json.loads(client.get("/api/home?month=9").get_data(as_text=True))
    ok &= check("/api/home 返回 JSON 且已个性化",
                data["logged_in"] and data["personalized"] and len(data["sections"]) == 3,
                f"month={data['month']} 板块={[s['title'] for s in data['sections']]}")
    ok &= check("/api/home 的每一行都带配图 URL",
                all(r["image"].startswith("/static/img/") for s in data["sections"] for r in s["rows"]))

    resp = client.get("/api/home?month=13")
    ok &= check("非法月份 → 400", resp.status_code == 400, resp.get_data(as_text=True).strip())
    resp = client.get("/home?month=abc")
    ok &= check("非法月份页面 → 200 且提示错误", resp.status_code == 200 and "month" in resp.get_data(as_text=True))
    return bool(ok)


def main() -> None:
    say(f"推荐首页自检 · {date.today().isoformat()}")
    say(f"运行环境：{sys.executable}\n")
    rec = HomeRecommender.load()
    customer_id = ensure_demo_user()

    say(f"数据规模：购买历史 {len(rec.history)} 位客户 / 季节统计 {len(rec.seasons)} 条")
    say(f"演示客户：{customer_id}（历史 {rec.history.count(customer_id)} 件商品）\n")

    ok_a = module_checks(rec, customer_id)
    ok_b = route_checks()

    say(f"\n结论：模块层 {'通过' if ok_a else '未通过'}；路由层 {'通过' if ok_b else '未通过'}")
    out = stage_output_dir(14) / "stage14_home_check.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n自检报告：{out}")


if __name__ == "__main__":
    main()
