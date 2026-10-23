#!/usr/bin/env python3
"""联网抓取真实照片：首页滚动推荐的大图 + 各商品类目的配图。

背景
----
本数据集（Online Retail + 补充数据）**不含商品照片**，之前的类目图是
``scripts/10_build_category_images.py`` 用 PIL 画的"渐变底 + emoji"示意图。
首页新增的**滚动推荐**需要一张能撑满横幅的大图，emoji 示意图放大后会糊，
所以这里改成**联网抓取真实照片**：

    一次搜索 → 一张原图 → 裁两个尺寸
        ├─ web/static/img/photos/<类目>.jpg   512×512   卡片 / 推荐依据 / 缩略图
        └─ web/static/img/banners/<类目>.jpg  1280×720  首页滚动推荐的大图

取图链路（`web/app.py` 里的 `image_url()` / `banner_url()`）会**优先用真实照片**，
没有照片的类目自动回落到 10 号脚本生成的 emoji 示意图，所以本脚本没跑、
或者某个类目抓取失败，页面依然完整可用。

数据来源与合规
--------------
用必应图片搜索的**公开结果页**（`/images/async`）拿到候选图片直链，再逐张下载。
候选图按顺序尝试，只有**能被 PIL 正常解码、且尺寸达标**的才留下；
每个类目最终用的那张图，来源 URL 会记进
``web/static/img/photos/manifest.json`` 与 ``outputs/stage18/stage18_image_sources.csv``，
便于答辩时说明出处（图片版权归原站点所有，本项目只用于课程演示）。

产物
----
    web/static/img/photos/<类目>.jpg         类目方图（512×512）
    web/static/img/banners/<类目>.jpg        类目横幅图（1280×720）
    web/static/img/photos/manifest.json      类目 → 文件名 + 来源 URL
    web/static/img/banners/manifest.json     类目 → 横幅文件名
    outputs/stage18/stage18_image_sources.csv        抓取记录（类目/查询词/来源/尺寸）
    outputs/stage18/stage18_category_photos.png      方图拼版（人工过一眼是否对得上类目）
    outputs/stage18/stage18_banner_photos.png        横幅图拼版

运行（项目根目录，**需要联网**）：

    python scripts/15_fetch_web_images.py                 # 抓全部类目（已有文件跳过）
    python scripts/15_fetch_web_images.py --force         # 全部重抓
    python scripts/15_fetch_web_images.py --only 耳机音响,笔记本电脑
    python scripts/15_fetch_web_images.py --only 手机 --pages 3   # 多翻几页候选

注意：只依赖标准库（``urllib``）+ Pillow，**不装 requests**（本机 PyPI 不通）。
"""
from __future__ import annotations

import argparse
import csv
import http.cookiejar
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from config.settings import PATHS, stage_output_dir

# ----------------------------------------------------------------- 常量
BING_ENDPOINT = "https://www.bing.com/images/async"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

PHOTO_DIR = ROOT / "web" / "static" / "img" / "photos"
BANNER_DIR = ROOT / "web" / "static" / "img" / "banners"
OUT_DIR = stage_output_dir(18)

PHOTO_SIZE = (512, 512)
BANNER_SIZE = (1280, 720)
PHOTO_QUALITY = 88
BANNER_QUALITY = 82

# 候选图至少要这么大才要：太小的图放大成横幅会糊
MIN_EDGE = 520
# 横竖比下限：纯竖图裁成 16:9 会只剩一条，信息量太少
MIN_ASPECT = 0.95

# 这些站点的图要么带水印、要么防盗链（下到手的是占位图或 403），直接跳过
BLOCKED_HOSTS = (
    "699pic.com",
    "tuke88.com",
    "tukuppt.com",
    "xinyuepu.com",
    "yueqiziyuan.com",
    "swarma.org",
    "deviantart.net",
    "artstation.com",
)

TEXT_FONT = r"C:\Windows\Fonts\msyh.ttc"

# 每个类目一条中文检索词 + 一组"标题关键词"（见 CATEGORY_KEYWORDS）。
# 类目名单独搜经常搜到概念图，所以检索词写成产品词（"美妆护肤"→"化妆品护肤品"）。
CATEGORY_QUERY: dict[str, str] = {
    "家居装饰": "家居装饰品摆件",
    "装饰品": "装饰工艺品摆件",
    "生活用品": "日用百货生活用品",
    "厨房用品": "厨房用具锅具",
    "派对用品": "派对装饰气球",
    "收纳用品": "收纳盒收纳箱",
    "手工用品": "手工diy材料",
    "购物袋": "帆布购物袋手提袋",
    "饰品": "项链手链饰品商品",
    "园艺用品": "盆栽花卉园艺",
    "花园用品": "花园庭院花卉装饰",
    "玩具": "儿童玩具积木商品",
    "宠物用品": "宠物用品猫狗",
    "文具用品": "文具笔记本中性笔",
    "办公用品": "办公用品文具",
    "工具": "五金工具扳手螺丝刀",
    "婴儿用品": "婴儿用品奶瓶商品",
    "洗浴用品": "沐浴露洗发水商品",
    "浴室用品": "浴室卫浴毛巾用品",
    "家具": "家具沙发桌椅",
    "配饰": "时尚配饰帽子围巾",
    "数码配件": "充电宝数据线商品",
    "家用电器": "家用电器电饭煲商品",
    "运动户外": "户外运动装备帐篷商品",
    "美妆护肤": "化妆品护肤品",
    "母婴用品": "母婴用品婴儿商品",
    "图书音像": "图书书籍封面",
    "汽车用品": "汽车用品车载商品",
    "食品饮料": "零食饮料包装商品",
    "珠宝首饰": "珠宝首饰项链商品",
    "鞋靴箱包": "运动鞋双肩包商品",
    "智能设备": "智能家居设备音箱",
    "乐器音响": "吉他乐器",
    "手机": "智能手机",
    "笔记本电脑": "笔记本电脑",
    "平板电脑": "平板电脑",
    "智能穿戴": "智能手表",
    "耳机音响": "蓝牙耳机",
    "摄影摄像": "数码相机",
    "游戏设备": "游戏手柄",
    "宿舍用品": "学生宿舍床品收纳",
    "其他": "礼品礼盒商品",
    "未知类目": "礼盒商品",
}

# 标题关键词：用来过滤搜索结果——标题里一个都对不上的当噪声。
# 默认从检索词自动切，下面这些手工写的比自动切准，直接覆盖。
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "家居装饰": ["家居", "装饰", "摆件", "居家"],
    "装饰品": ["装饰", "摆件", "工艺"],
    "生活用品": ["日用", "生活", "百货", "用品"],
    "厨房用品": ["厨房", "锅", "餐具", "厨具"],
    "派对用品": ["派对", "气球", "装饰"],
    "收纳用品": ["收纳", "整理", "箱", "盒"],
    "手工用品": ["手工", "diy", "材料", "编织"],
    "购物袋": ["购物袋", "帆布", "手提袋", "布袋"],
    "饰品": ["项链", "手链", "饰品", "耳环"],
    "园艺用品": ["园艺", "盆栽", "花卉", "绿植"],
    "花园用品": ["花园", "庭院", "花卉", "户外"],
    "玩具": ["玩具", "积木", "儿童"],
    "宠物用品": ["宠物", "猫", "狗", "用品"],
    "文具用品": ["文具", "笔记本", "笔", "本子"],
    "办公用品": ["办公", "文具", "用品"],
    "工具": ["工具", "五金", "扳手", "螺丝", "钳"],
    "婴儿用品": ["婴儿", "奶瓶", "尿布", "宝宝"],
    "洗浴用品": ["沐浴", "洗发", "洗浴", "浴"],
    "浴室用品": ["浴室", "卫浴", "毛巾", "淋浴"],
    "家具": ["家具", "沙发", "桌", "椅"],
    "配饰": ["配饰", "帽", "围巾", "腰带"],
    "数码配件": ["充电", "数据线", "数码", "配件", "耳机"],
    "家用电器": ["电器", "家电", "电饭", "冰箱", "洗衣机"],
    "运动户外": ["运动", "户外", "帐篷", "跑步", "健身"],
    "美妆护肤": ["美妆", "化妆", "护肤", "口红"],
    "母婴用品": ["母婴", "婴儿", "孕妇", "宝宝"],
    "图书音像": ["图书", "书籍", "书", "杂志"],
    "汽车用品": ["汽车", "车载", "车", "用品"],
    "食品饮料": ["食品", "饮料", "零食", "包装"],
    "珠宝首饰": ["珠宝", "首饰", "项链", "戒指"],
    "鞋靴箱包": ["鞋", "靴", "包", "箱"],
    "智能设备": ["智能", "家居", "设备", "音箱"],
    "乐器音响": ["乐器", "吉他", "音响", "钢琴"],
    "手机": ["手机", "智能"],
    "笔记本电脑": ["笔记本", "电脑"],
    "平板电脑": ["平板", "电脑"],
    "智能穿戴": ["智能", "手表", "手环", "穿戴"],
    "耳机音响": ["耳机", "音响", "蓝牙"],
    "摄影摄像": ["相机", "摄影", "摄像", "镜头"],
    "游戏设备": ["游戏", "手柄", "主机", "电竞"],
    "宿舍用品": ["宿舍", "床", "收纳", "被"],
    "其他": ["礼盒", "礼品", "商品", "用品"],
    "未知类目": ["礼盒", "礼品", "商品"],
}

# 结果块：`m="{&quot;sid&quot;:...,&quot;murl&quot;:&quot;图片直链&quot;,...,&quot;t&quot;:&quot;网页标题&quot;}"`
_BLOCK_RE = re.compile(rb'm="(\{&quot;.*?\})"', re.DOTALL)
_MURL_RE = re.compile(rb"murl&quot;:&quot;(.*?)&quot;")
_TITLE_RE = re.compile(rb"&quot;t&quot;:&quot;(.*?)&quot;")


# ----------------------------------------------------------------- 抓取
def _get(url: str, timeout: int = 25) -> bytes:
    """带浏览器 UA + Cookie 会话的 GET；返回响应体。

    必应的图片结果接口对"没有会话的裸请求"经常返回上一次查询的结果，
    所以这里挂一个 cookie jar，并带上来源页 Referer。
    """
    headers = dict(HEADERS)
    headers["Referer"] = "https://www.bing.com/images/search"
    request = urllib.request.Request(url, headers=headers)
    with _OPENER.open(request, timeout=timeout) as response:  # noqa: S310 - 只取公开图源
        return response.read()


_COOKIE_JAR = http.cookiejar.CookieJar()
_OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_COOKIE_JAR))


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", "ignore").replace("\\/", "/").strip()


def search_candidates_360(query: str, pages: int = 2, per_page: int = 30) -> list[dict[str, str]]:
    """360 图片搜索的公开结果接口：一次返回整页 JSON（含标题与原图尺寸）。

    比必应结果页稳：不会被限流成"上一次查询的结果"，而且**标题与原图宽高都在 JSON 里**，
    可以下载前就把明显不合适的候选（太小、太竖、标题不相关）筛掉。
    """
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for page in range(pages):
        params = urllib.parse.urlencode(
            {"q": query, "pn": page * per_page, "rn": per_page, "src": "srp"}
        )
        try:
            payload = json.loads(_get(f"https://image.so.com/j?{params}").decode("utf-8", "ignore"))
        except Exception as exc:  # noqa: BLE001 - 换源或下一轮再试
            print(f"    [warn] 360 搜索失败（第 {page + 1} 页）：{type(exc).__name__}: {exc}")
            continue
        for item in payload.get("list", []):
            title = str(item.get("title", ""))
            try:
                width = int(item.get("width") or 0)
                height = int(item.get("height") or 0)
            except (TypeError, ValueError):
                width = height = 0
            for key, kind in (("img", "原图"), ("thumb", "缩略")):
                url = str(item.get(key) or "").strip()
                if not url.startswith("http") or url in seen:
                    continue
                seen.add(url)
                found.append(
                    {
                        "url": url,
                        "title": title,
                        "width": width,
                        "height": height,
                        "size_kind": kind,
                        "source": "360",
                    }
                )
    return found


def search_candidates_bing(
    query: str,
    pages: int = 2,
    per_page: int = 35,
    *,
    photo_only: bool = True,
) -> list[dict[str, str]]:
    """用必应图片搜索的公开结果页拿候选（图片直链 + 网页标题）。

    ``photo_only`` 会带上"只要照片"的过滤（排掉插画/表情包/3D 渲染），
    候选太少时调用方可以再关掉它重搜一次。

    先打**主结果页**（一次就能拿到整页结果，而且会种下会话 Cookie），
    再用异步接口翻后面的页 —— 直接裸请求异步接口时必应常返回上一次查询的结果。
    """
    found: list[dict[str, str]] = []
    seen: set[str] = set()

    def absorb(html: bytes) -> None:
        for block in _BLOCK_RE.findall(html):
            murl = _MURL_RE.search(block)
            title = _TITLE_RE.search(block)
            if not murl:
                continue
            url = _decode(murl.group(1)).split(",")[0]
            if not url.startswith("http") or url in seen:
                continue
            seen.add(url)
            found.append(
                {
                    "url": url,
                    "title": _decode(title.group(1)) if title else "",
                    "width": 0,
                    "height": 0,
                    "size_kind": "原图",
                    "source": "bing",
                }
            )

    filters = "+filterui:imagesize-large"
    if photo_only:
        filters += "+filterui:photo-photo"
    main_params = urllib.parse.urlencode(
        {"q": query, "form": "HDRSC2", "first": 1, "count": per_page, "qft": filters}
    )
    try:
        absorb(_get(f"https://www.bing.com/images/search?{main_params}"))
    except Exception as exc:  # noqa: BLE001 - 主结果页失败就只剩异步接口
        print(f"    [warn] 主结果页失败：{type(exc).__name__}: {exc}")

    for page in range(pages):
        params = urllib.parse.urlencode(
            {
                "q": query,
                "first": 1 + page * per_page,
                "count": per_page,
                "mmasync": 1,
                "qft": filters,
            }
        )
        try:
            absorb(_get(f"{BING_ENDPOINT}?{params}"))
        except Exception as exc:  # noqa: BLE001 - 搜索失败就少一页候选，不致命
            print(f"    [warn] 搜索失败（第 {page + 1} 页）：{type(exc).__name__}: {exc}")
            continue
    return found


def search_candidates(query: str, pages: int = 2, *, photo_only: bool = True) -> list[dict[str, str]]:
    """先问 360，候选不够再问必应（两个都是公开结果页，不用 API key）。"""
    candidates = search_candidates_360(query, pages=pages)
    if len(candidates) >= 3:
        return candidates
    return candidates + search_candidates_bing(query, pages=pages, photo_only=photo_only)


def keywords_for(category: str, query: str) -> list[str]:
    """类目的标题关键词：手工表优先，否则从检索词里切 2 字滑窗。"""
    manual = CATEGORY_KEYWORDS.get(category)
    if manual:
        return [word.lower() for word in manual]
    words = [part for part in re.split(r"[\s,，]+", query) if len(part) >= 2]
    if words:
        return [word.lower() for word in words]
    return [query[i : i + 2].lower() for i in range(max(1, len(query) - 1))]


def is_relevant(candidate: dict[str, str], keywords: list[str]) -> bool:
    """标题里出现任一关键词才算相关；标题为空时不判负（少一道过滤门槛）。"""
    title = candidate.get("title", "").lower()
    if not title:
        return True
    return any(word in title for word in keywords)


def is_blocked(url: str) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return any(blocked in host for blocked in BLOCKED_HOSTS)


def load_image(data: bytes) -> Image.Image | None:
    """解码图片；不是图片 / 打不开就返回 None。"""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except Exception:  # noqa: BLE001 - 有些直链其实返回的是 HTML 错误页
        return None


def is_usable(image: Image.Image) -> tuple[bool, str]:
    """尺寸门槛：太小、太扁、太竖的都不要。"""
    width, height = image.size
    if min(width, height) < MIN_EDGE:
        return False, f"太小 {width}×{height}"
    if width / max(height, 1) < MIN_ASPECT:
        return False, f"太竖 {width}×{height}"
    return True, f"{width}×{height}"


def average_hash(image: Image.Image, size: int = 8) -> int:
    """感知哈希（均值哈希）：用来过滤"同一张图换了 URL"。"""
    small = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(small.tobytes())  # 等价于 getdata()，但不触发 Pillow 的弃用提示
    avg = sum(pixels) / len(pixels)
    bits = 0
    for index, value in enumerate(pixels):
        if value >= avg:
            bits |= 1 << index
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """等比缩放后居中裁切（CSS ``object-fit: cover`` 的等价物）。"""
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def save_jpeg(image: Image.Image, path: Path, quality: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=quality, optimize=True, progressive=True)
    return path.stat().st_size


# ----------------------------------------------------------------- 主流程
def fetch_one(
    category: str,
    *,
    pages: int,
    tries: int,
    used_hashes: list[int],
    force: bool,
    require_relevant: bool = True,
) -> dict | None:
    """抓一个类目的照片，返回记录（失败返回 None）。"""
    photo_path = PHOTO_DIR / f"{category}.jpg"
    banner_path = BANNER_DIR / f"{category}.jpg"
    if photo_path.exists() and banner_path.exists() and not force:
        print(f"  {category:<8} [skip] 已有照片")
        return {"category": category, "skipped": True}

    query = CATEGORY_QUERY.get(category, f"{category} 商品")
    keywords = keywords_for(category, query)
    candidates = [c for c in search_candidates(query, pages=pages) if not is_blocked(c["url"])]
    if not candidates:
        print(f"  {category:<8} [fail] 搜索没有返回候选图（查询词：{query}）")
        return None

    relevant = [c for c in candidates if is_relevant(c, keywords)] if require_relevant else []
    if require_relevant:
        if relevant:
            candidates = relevant
        else:
            print(f"      [warn] 没有一条标题命中 {keywords}，先用未过滤的候选")

    # 原图尺寸已知时先排掉"太小 / 太竖"的候选（省掉一次没必要的下载）。
    # 排完按"最小边"从大到小：横幅大图优先挑分辨率高的。
    def declared_edge(candidate: dict[str, str]) -> int:
        try:
            width = int(candidate.get("width") or 0)
            height = int(candidate.get("height") or 0)
        except (TypeError, ValueError):
            return 0
        if width <= 0 or height <= 0:
            return 0
        if min(width, height) < MIN_EDGE or width / max(height, 1) < MIN_ASPECT:
            return -1
        return min(width, height)

    sized = [(declared_edge(c), c) for c in candidates]
    keep = [c for edge, c in sized if edge >= 0]
    if keep:
        keep.sort(key=lambda c: -declared_edge(c))
        candidates = keep + [c for edge, c in sized if edge == -1]

    tried = 0
    for candidate in candidates:
        if tried >= tries:
            break
        url = candidate["url"]
        tried += 1
        try:
            data = _get(url, timeout=20)
        except urllib.error.HTTPError as exc:
            print(f"      - HTTP {exc.code} {url[:90]}")
            continue
        except Exception as exc:  # noqa: BLE001 - 单张图失败就换下一张
            print(f"      - {type(exc).__name__} {url[:90]}")
            continue

        image = load_image(data)
        if image is None:
            print(f"      - 不是图片 {url[:90]}")
            continue
        ok, detail = is_usable(image)
        if not ok:
            print(f"      - {detail} {url[:90]}")
            continue

        digest = average_hash(image)
        if any(hamming(digest, h) <= 4 for h in used_hashes):
            print(f"      - 与已有图太像，换一张 {url[:90]}")
            continue

        photo_bytes = save_jpeg(cover(image, PHOTO_SIZE), photo_path, PHOTO_QUALITY)
        banner_bytes = save_jpeg(cover(image, BANNER_SIZE), banner_path, BANNER_QUALITY)
        used_hashes.append(digest)
        print(
            f"  {category:<8} [ok]   {detail}  方图 {photo_bytes // 1024}KB / "
            f"横幅 {banner_bytes // 1024}KB"
        )
        print(f"       标题：{candidate.get('title', '')[:60]}")
        print(f"       {url[:110]}")
        return {
            "category": category,
            "query": query,
            "source_engine": candidate.get("source", ""),
            "source": url,
            "source_title": candidate.get("title", ""),
            "width": image.width,
            "height": image.height,
            "photo": f"{category}.jpg",
            "banner": f"{category}.jpg",
            "photo_bytes": photo_bytes,
            "banner_bytes": banner_bytes,
        }

    print(f"  {category:<8} [fail] 试了 {tried} 张候选，没有一张可用（查询词：{query}）")
    return None


def build_sheets(records: list[dict]) -> None:
    """把抓到的照片拼成两张对照图，方便人工确认"图和类目对得上"。"""
    try:
        font = ImageFont.truetype(TEXT_FONT, 20)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()

    def sheet(items: list[tuple[str, Path]], cell: tuple[int, int], columns: int, out: Path) -> None:
        if not items:
            return
        cell_w, cell_h = cell
        label_h = 30
        rows = (len(items) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * cell_w, rows * (cell_h + label_h)), (241, 236, 227))
        draw = ImageDraw.Draw(canvas)
        for index, (label, path) in enumerate(items):
            if not path.exists():
                continue
            row, column = divmod(index, columns)
            x = column * cell_w
            y = row * (cell_h + label_h)
            with Image.open(path) as image:
                canvas.paste(image.convert("RGB").resize(cell, Image.Resampling.LANCZOS), (x, y))
            draw.text((x + 8, y + cell_h + 5), label, font=font, fill=(28, 27, 24))
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out)
        print(f"  拼版 → {out.relative_to(ROOT)}  （{len(items)} 张）")

    square_items = [(r["category"], PHOTO_DIR / r["photo"]) for r in records if r.get("photo")]
    banner_items = [(r["category"], BANNER_DIR / r["banner"]) for r in records if r.get("banner")]
    sheet(square_items, (170, 170), 7, OUT_DIR / "stage18_category_photos.png")
    sheet(banner_items, (280, 158), 4, OUT_DIR / "stage18_banner_photos.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="联网抓取类目方图与首页横幅图")
    parser.add_argument("--only", default="", help="只抓这些类目（逗号分隔）")
    parser.add_argument("--force", action="store_true", help="已有文件也重新抓")
    parser.add_argument("--pages", type=int, default=2, help="每个类目翻几页候选（默认 2）")
    parser.add_argument("--tries", type=int, default=6, help="每个类目最多试几张候选图（默认 6）")
    parser.add_argument(
        "--no-relevance",
        action="store_true",
        help="关闭标题相关性过滤（候选不够时用）",
    )
    parser.add_argument("--probe", default="", help="只试搜索一个词并打印候选标题，不下载")
    parser.add_argument(
        "--sync-manifest",
        action="store_true",
        help="不联网：只把 photos/ 与 banners/ 目录里现有的图重新写进 manifest.json",
    )
    args = parser.parse_args()

    if args.probe:  # 调检索词用：看看这个词搜出来的都是些什么
        for candidate in search_candidates(args.probe, pages=max(1, args.pages)):
            print(f"  {candidate['title'][:70]}\n      {candidate['url'][:100]}")
        return

    if args.sync_manifest:  # 手工换图之后用：让清单和磁盘一致（页面本来也以磁盘为准）
        for directory in (PHOTO_DIR, BANNER_DIR):
            entries: dict[str, str] = {}
            for path in sorted(directory.glob("*")):
                if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    entries[path.stem] = path.name
            (directory / "manifest.json").write_text(
                json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"{directory.relative_to(ROOT)} → 清单 {len(entries)} 条")
        return

    profile_path = Path(PATHS["item_profile"])
    manifest_path = ROOT / "web" / "static" / "img" / "categories" / "manifest.json"
    if manifest_path.exists():
        categories = list(json.loads(manifest_path.read_text(encoding="utf-8")))
    else:
        print(f"缺少类目清单（{manifest_path}），请先运行 scripts/10_build_category_images.py")
        sys.exit(1)
    print(f"商品画像：{profile_path.name} · 类目清单：{len(categories)} 个")

    if args.only:
        wanted = [c.strip() for c in args.only.split(",") if c.strip()]
        missing = [c for c in wanted if c not in categories]
        if missing:
            print(f"这些类目不在清单里：{missing}")
            sys.exit(1)
        categories = wanted

    records: list[dict] = []
    used_hashes: list[int] = []
    if not args.force:  # 重抓时也要避开与已抓图重复，先把已有图的哈希读进来
        for existing in sorted(PHOTO_DIR.glob("*.jpg")):
            try:
                with Image.open(existing) as image:
                    used_hashes.append(average_hash(image.convert("RGB")))
            except Exception:  # noqa: BLE001
                continue

    started = time.time()
    for category in categories:
        print(f"\n[{category}]")
        record = fetch_one(
            category,
            pages=max(1, args.pages),
            tries=max(1, args.tries),
            used_hashes=used_hashes,
            force=args.force,
            require_relevant=not args.no_relevance,
        )
        if record and not record.get("skipped"):
            records.append(record)

    # ---- 写清单（跳过的不写，保持清单 = 真的抓到的照片） ----
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    BANNER_DIR.mkdir(parents=True, exist_ok=True)
    photo_manifest = {}
    banner_manifest = {}
    if not args.only and not args.force:
        # 全量跑：按目录里实际存在的文件重建清单，避免删过图之后清单还留着
        for category in categories:
            if (PHOTO_DIR / f"{category}.jpg").exists():
                photo_manifest[category] = f"{category}.jpg"
            if (BANNER_DIR / f"{category}.jpg").exists():
                banner_manifest[category] = f"{category}.jpg"
    else:
        old_photo = (
            json.loads((PHOTO_DIR / "manifest.json").read_text(encoding="utf-8"))
            if (PHOTO_DIR / "manifest.json").exists()
            else {}
        )
        old_banner = (
            json.loads((BANNER_DIR / "manifest.json").read_text(encoding="utf-8"))
            if (BANNER_DIR / "manifest.json").exists()
            else {}
        )
        photo_manifest.update(old_photo)
        banner_manifest.update(old_banner)
        for record in records:
            photo_manifest[record["category"]] = record["photo"]
            banner_manifest[record["category"]] = record["banner"]
    (PHOTO_DIR / "manifest.json").write_text(
        json.dumps(photo_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (BANNER_DIR / "manifest.json").write_text(
        json.dumps(banner_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 抓取记录（答辩要能说出"这张图从哪来"） ----
    if records:
        csv_path = OUT_DIR / "stage18_image_sources.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "category",
                    "query",
                    "source_engine",
                    "source_title",
                    "source",
                    "width",
                    "height",
                    "photo_bytes",
                    "banner_bytes",
                ],
            )
            writer.writeheader()
            for record in records:
                writer.writerow({key: record.get(key, "") for key in writer.fieldnames})
        print(f"\n抓取记录 → {csv_path.relative_to(ROOT)}（{len(records)} 条）")

    build_sheets(records)

    have_photo = len(list(PHOTO_DIR.glob("*.jpg")))
    have_banner = len(list(BANNER_DIR.glob("*.jpg")))
    print(
        f"\n完成：本次抓取 {len(records)} 个类目，用时 {time.time() - started:.1f} 秒；"
        f"现有方图 {have_photo} 张、横幅 {have_banner} 张"
    )
    print("取图优先级：web/static/img/items/<商品ID>.* → photos/<类目>.jpg → categories/<类目>.png")
    if records and args.only:
        print("提示：只跑了一部分类目，清单已与旧清单合并。")


if __name__ == "__main__":
    main()
