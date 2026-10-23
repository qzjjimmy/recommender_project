#!/usr/bin/env python3
"""首页滚动推荐（轮播）自检：模块层 + 路由层 + 图片链路，结果写到
``outputs/stage19/stage19_carousel_check.txt``。

三段各管一件事：

    A. 模块层：``HomeRecommender.carousel_rows()`` 的编队规则
       —— 条数固定、板块轮流取、同类目最多 1 条、不推已买过 / 点过不喜欢的。
    B. 路由层：``/home`` 与 ``/api/home`` 里轮播的字段是否齐全
       （href / image / thumb / 文案），以及 ``/health`` 能不能看到图片资产数量。
    C. 图片链路：类目配图的三级回落（真实商品图 → 联网照片 → 本地示意图）
       是否每一级都指向**真实存在的文件**；照片覆盖了多少个类目。

运行（项目根目录）：
    python scripts/16_check_carousel.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import HYPERPARAMS, PATHS, stage_output_dir
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
        user_store.create_user(PATHS["user_db"], DEMO_USER, DEMO_PASS, "演示用户", "14897")
        user = user_store.get_user(PATHS["user_db"], DEMO_USER)
    return str(user["customer_id"]) if user else ""


def module_checks(rec: HomeRecommender, customer_id: str) -> bool:
    say("A. 模块层：carousel_rows()")
    ok = True
    want = int(HYPERPARAMS["home"]["carousel_slides"])
    cap = int(HYPERPARAMS["home"]["carousel_max_per_category"])

    data = rec.build(username=DEMO_USER, month=9)
    carousel = data["carousel"]
    ok &= check(f"登录后轮播条数 = 配置的 {want} 条", len(carousel) == want,
                "、".join(f"{s['badge']}:{s['name']}" for s in carousel))
    ok &= check("每条都带来源板块、类目、价格、热度与推荐理由",
                all(
                    s["badge"] and s["category"] and s["price"] is not None
                    and s["users"] > 0 and s["reason_detail"]
                    for s in carousel
                ))
    ok &= check("轮播里没有重复商品", len({s["item_id"] for s in carousel}) == len(carousel))
    cats = [s["category"] for s in carousel]
    ok &= check(f"同类目最多 {cap} 条（照片是按类目取的，重复等于图片不换）",
                all(cats.count(c) <= cap for c in set(cats)), str(cats))
    ok &= check("两个板块都上镜（不是被某一个板块包场）",
                {"personal", "season"} <= {s["badge_kind"] for s in carousel},
                str([s["badge_kind"] for s in carousel]))

    purchased = rec.history.item_ids(customer_id)
    ok &= check("不推已经买过的商品",
                all(s["item_id"] not in purchased for s in carousel))
    ok &= check("「猜你还会买」那几条带着来源商品（因为它买过它）",
                all(s["origin_name"] for s in carousel if s["badge_kind"] == "personal"),
                "；".join(f"{s['name']} ← {s['origin_name']}"
                          for s in carousel if s["badge_kind"] == "personal")[:90])

    anon = rec.build(username=None, month=9)["carousel"]
    ok &= check("未登录也有轮播（档期 + 热销兜底）", len(anon) == want,
                "、".join(f"{s['badge']}:{s['name']}" for s in anon))

    # 把轮播的第一条"点踩"，它应该从轮播里消失（反馈闭环对轮播同样生效）
    blocked = carousel[0]["item_id"]
    filtered = rec.build(username=None, month=9)["carousel"]
    manual = rec.carousel_rows(
        rec.build(username=None, month=9)["sections"], disliked=[blocked], slides=want
    )
    ok &= check("点过「不喜欢」的商品不会出现在轮播里",
                blocked not in {s["item_id"] for s in manual},
                f"屏蔽 {blocked} 后仍是 {len(manual)} 条")
    ok &= check("匿名轮播不掺个性化板块", all(s["badge_kind"] != "personal" for s in filtered))

    # 极端情况：没有板块时靠热销补满
    only_popular = rec.carousel_rows([], slides=want)
    ok &= check("板块全空时用全站热销补满（不会少于配置条数）",
                len(only_popular) == want,
                "、".join(s["name"] for s in only_popular))
    return bool(ok)


def route_checks() -> bool:
    say("\nB. 路由层：Flask test_client")
    import web.app as webapp

    ok = True
    client = webapp.app.test_client()

    html = client.get("/home").get_data(as_text=True)
    ok &= check("匿名 /home 含滚动推荐区块", 'id="spot"' in html and "滚动推荐" in html)
    ok &= check("轮播容器带自动滚动间隔与无障碍标注",
                'data-interval="' in html and 'aria-roledescription="carousel"' in html)

    ensure_demo_user()
    client.post("/login", data={"username": DEMO_USER, "password": DEMO_PASS})
    html = client.get("/home?month=9").get_data(as_text=True)
    slide_hits = html.count('class="slide')
    ok &= check("登录后 /home 渲染出 5 条 slide",
                slide_hits >= 5, f"slide 出现 {slide_hits} 次")
    ok &= check("轮播里有大图 <img class=\"slide-img\">", 'slide-img' in html)
    ok &= check("轮播带箭头 / 圆点 / 进度条 / 计数",
                all(token in html for token in ("navbtn", "sdot", "spot-progress", "spot-index")))

    data = json.loads(client.get("/api/home?month=9").get_data(as_text=True))
    slides = data["carousel"]
    ok &= check("/api/home 返回 carousel（5 条）", len(slides) == 5, f"{len(slides)} 条")
    ok &= check("每条 slide 都有 href / thumb",
                all(s["href"].startswith("/similar?item_id=") and s["thumb"] for s in slides))
    ok &= check("每条 slide 的 image 指向静态横幅或为空（空则前端走渐变底）",
                all(s["image"].startswith("/static/img/banners/") or s["image"] == ""
                    for s in slides),
                f"有横幅 {sum(1 for s in slides if s['image'])}/{len(slides)} 条")
    ok &= check("carousel_interval_ms 透传到页面", int(data["carousel_interval_ms"]) >= 2000,
                f"{data['carousel_interval_ms']} ms")

    served = [s["image"] for s in slides if s["image"]]
    if served:
        resp = client.get(served[0])
        ok &= check("横幅图片能被服务真实返回", resp.status_code == 200,
                    f"{served[0]} → {resp.status_code} / {len(resp.get_data())} 字节")

    health = json.loads(client.get("/health").get_data(as_text=True))
    images = health.get("home_images", {})
    ok &= check("/health 能看到图片资产数量",
                images.get("photos", 0) > 0 and images.get("banners", 0) > 0,
                f"照片 {images.get('photos')} / 横幅 {images.get('banners')} / "
                f"示意图 {images.get('category_tiles')}")
    return bool(ok)


def image_chain_checks() -> bool:
    say("\nC. 图片链路：类目配图三级回落")
    import web.app as webapp

    ok = True
    manifest = ROOT / "web" / "static" / "img" / "categories" / "manifest.json"
    categories = list(json.loads(manifest.read_text(encoding="utf-8")))
    stats = {"photos": 0, "tiles": 0, "missing": 0}
    broken: list[str] = []

    for category in categories:
        url = webapp.image_url(f"__no_such_item__{category}", category)
        # URL 里类目名是百分号编码的（中文），落盘前先解码
        file = ROOT / "web" / "static" / unquote(url).removeprefix("/static/")
        if not file.exists():
            stats["missing"] += 1
            broken.append(f"{category}→{url}")
        elif "/photos/" in url:
            stats["photos"] += 1
        else:
            stats["tiles"] += 1

    say(f"  类目 {len(categories)} 个：真实照片 {stats['photos']} / 本地示意图 {stats['tiles']}")
    ok &= check("每个类目都能解析到一张真实存在的图", stats["missing"] == 0,
                "缺失：" + ("、".join(broken) if broken else "无"))
    ok &= check("至少一半类目用上了联网抓的照片",
                stats["photos"] * 2 >= len(categories),
                f"{stats['photos']}/{len(categories)}")

    # 用服务里真正生效的那份目录清单（扫描磁盘得到的），而不是脚写的 manifest.json
    banners = webapp.BANNER_IMAGES
    photos = webapp.PHOTO_IMAGES
    ok &= check("横幅 / 方图的目录清单与磁盘一致",
                bool(banners) and bool(photos) and all(
                    (ROOT / "web" / "static" / "img" / "banners" / f).exists()
                    for f in banners.values()
                ) and all(
                    (ROOT / "web" / "static" / "img" / "photos" / f).exists()
                    for f in photos.values()
                ),
                f"横幅 {len(banners)} 个 / 方图 {len(photos)} 个类目")
    return bool(ok)


def main() -> None:
    say(f"首页滚动推荐自检 · {date.today().isoformat()}")
    say(f"运行环境：{sys.executable}\n")
    rec = HomeRecommender.load()
    customer_id = ensure_demo_user()
    say(f"数据规模：购买历史 {len(rec.history)} 位客户 / 季节统计 {len(rec.seasons)} 条")
    say(f"演示客户：{customer_id}（历史 {rec.history.count(customer_id)} 件商品）\n")

    ok_a = module_checks(rec, customer_id)
    ok_b = route_checks()
    ok_c = image_chain_checks()

    say(
        f"\n结论：模块层 {'通过' if ok_a else '未通过'}；"
        f"路由层 {'通过' if ok_b else '未通过'}；"
        f"图片链路 {'通过' if ok_c else '未通过'}"
    )
    out = stage_output_dir(19) / "stage19_carousel_check.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n自检报告：{out}")


if __name__ == "__main__":
    main()
