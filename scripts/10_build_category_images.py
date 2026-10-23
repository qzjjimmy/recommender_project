#!/usr/bin/env python3
"""生成商品类目示意图（本地生成，不依赖网络与外部服务）。

背景说明
--------
本数据集（Online Retail）**不包含商品照片**，环境里也没有可用的图像生成服务，
所以这里用程序化方式生成"类目示意图"：柔和渐变底 + 大号彩色 emoji + 类目名。
它不是为了冒充真实商品图，而是让列表有视觉锚点、便于快速扫读类目。

真实商品图的接入方式
--------------------
Web 服务取图的优先级是：
    1. ``web/static/img/items/<商品ID>.png``   ← 有就优先用（真实商品图放这里）
    2. ``web/static/img/categories/<类目>.png`` ← 否则回落到类目示意图
所以后续只要把真实商品图按商品 ID 命名丢进 ``items/`` 目录，**不用改代码**就会自动生效。

产物：
    web/static/img/categories/<slug>.png   各类目示意图（240×240 RGBA）
    web/static/img/categories/manifest.json  类目 → 文件名 的映射
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from config.settings import PATHS

SIZE = 240
EMOJI_FONT = r"C:\Windows\Fonts\seguiemj.ttf"
TEXT_FONT = r"C:\Windows\Fonts\msyh.ttc"

# 类目 → (emoji, 主题色)
CATEGORY_STYLE: dict[str, tuple[str, str]] = {
    "家居装饰": ("🏠", "#f59e0b"),
    "装饰品": ("🎀", "#ec4899"),
    "生活用品": ("🧴", "#14b8a6"),
    "厨房用品": ("🍳", "#ef4444"),
    "派对用品": ("🎉", "#8b5cf6"),
    "收纳用品": ("📦", "#0ea5e9"),
    "手工用品": ("✂️", "#22c55e"),
    "购物袋": ("🛍️", "#d946ef"),
    "饰品": ("💍", "#eab308"),
    "园艺用品": ("🪴", "#16a34a"),
    "花园用品": ("🌷", "#f472b6"),
    "玩具": ("🧸", "#fb923c"),
    "宠物用品": ("🐾", "#a16207"),
    "文具用品": ("✏️", "#3b82f6"),
    "办公用品": ("🖇️", "#64748b"),
    "工具": ("🔧", "#475569"),
    "婴儿用品": ("🍼", "#38bdf8"),
    "洗浴用品": ("🛁", "#06b6d4"),
    "浴室用品": ("🚿", "#0891b2"),
    "家具": ("🛋️", "#a855f7"),
    "配饰": ("🎩", "#7c3aed"),
    # ---- 补充数据的 12 个新类目 ----
    "数码配件": ("📱", "#0ea5e9"),
    "家用电器": ("🔌", "#6366f1"),
    "运动户外": ("⛺", "#16a34a"),
    "美妆护肤": ("💄", "#ec4899"),
    "母婴用品": ("👶", "#f472b6"),
    "图书音像": ("📚", "#b45309"),
    "汽车用品": ("🚗", "#475569"),
    "食品饮料": ("🍫", "#d97706"),
    "珠宝首饰": ("💎", "#8b5cf6"),
    "鞋靴箱包": ("👟", "#0d9488"),
    "智能设备": ("⌚", "#2563eb"),
    "乐器音响": ("🎸", "#a16207"),
    # ---- v2 新增（3C 品牌类目 + 校园类目）----
    "手机": ("📱", "#2563eb"),
    "笔记本电脑": ("💻", "#475569"),
    "平板电脑": ("📲", "#0ea5e9"),
    "智能穿戴": ("⌚", "#7c3aed"),
    "耳机音响": ("🎧", "#0891b2"),
    "摄影摄像": ("📷", "#334155"),
    "游戏设备": ("🎮", "#7c3aed"),
    "宿舍用品": ("🛏️", "#0d9488"),
    "其他": ("🎁", "#6b7280"),
    "未知类目": ("❔", "#9ca3af"),
}
DEFAULT_STYLE = ("🎁", "#6b7280")


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))  # type: ignore[return-value]


def draw_tile(category: str, emoji: str, color: str, out_path: Path) -> None:
    """画一张 240×240 的类目示意图：柔和渐变 + 大号 emoji + 类目名。"""
    base = _hex_to_rgb(color)
    white = (255, 255, 255)

    # 1) 垂直渐变底（上浅下稍深）
    gradient = Image.new("RGB", (SIZE, SIZE))
    pixels = gradient.load()
    for y in range(SIZE):
        t = y / (SIZE - 1)
        top = _mix(white, base, 0.10)
        bottom = _mix(white, base, 0.30)
        row = _mix(top, bottom, t)
        for x in range(SIZE):
            pixels[x, y] = row

    # 2) 圆角遮罩
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], radius=22, fill=255)
    tile = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    tile.paste(gradient, (0, 0), mask)

    draw = ImageDraw.Draw(tile)

    # 3) 大号彩色 emoji（居中偏上）
    try:
        emoji_font = ImageFont.truetype(EMOJI_FONT, 104)
        box = draw.textbbox((0, 0), emoji, font=emoji_font, embedded_color=True)
        w, h = box[2] - box[0], box[3] - box[1]
        draw.text(
            ((SIZE - w) / 2 - box[0], 88 - h / 2 - box[1]),
            emoji,
            font=emoji_font,
            embedded_color=True,
        )
    except Exception:
        pass  # 拿不到 emoji 字体时只留底色与文字

    # 4) 类目名（底部居中）
    try:
        text_font = ImageFont.truetype(TEXT_FONT, 22)
    except Exception:
        text_font = ImageFont.load_default()
    box = draw.textbbox((0, 0), category, font=text_font)
    w = box[2] - box[0]
    draw.text(((SIZE - w) / 2 - box[0], 186), category, font=text_font, fill=_mix(base, (0, 0, 0), 0.55))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tile.save(out_path)


def main() -> None:
    out_dir = ROOT / "web" / "static" / "img" / "categories"
    profile_path = Path(PATHS["item_profile"])
    if not profile_path.exists():
        print("缺少商品画像，请先运行 scripts/09_build_app_data.py")
        sys.exit(1)

    import pandas as pd

    profile = pd.read_csv(profile_path, dtype={"item_id": str})
    categories = sorted(profile["category"].fillna("未知类目").astype(str).unique())

    manifest: dict[str, str] = {}
    for category in categories:
        emoji, color = CATEGORY_STYLE.get(category, DEFAULT_STYLE)
        # 直接用品类名做文件名：稳定、可读、可复现。
        # （注意：不能用 Python 的 hash() —— 字符串 hash 每进程随机，会导致重跑文件名变化）
        slug = f"{category}.png"
        draw_tile(category, emoji, color, out_dir / slug)
        manifest[category] = slug
        print(f"  {category:<8} {emoji}  -> {slug}")

    # 兜底图
    if "未知类目" not in manifest:
        draw_tile("未知类目", *DEFAULT_STYLE, out_dir / "unknown.png")
        manifest["未知类目"] = "unknown.png"

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n共生成 {len(manifest)} 张类目示意图 → {out_dir}")
    print("真实商品图放这里即可自动生效：web/static/img/items/<商品ID>.png")


if __name__ == "__main__":
    main()
