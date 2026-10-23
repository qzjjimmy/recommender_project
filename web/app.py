"""推荐系统在线服务（Flask）。

启动（在项目根目录）：
    C:\\Users\\qiuzijian\\anaconda3\\envs\\hygc_qzj\\python.exe web\\app.py

页面：
    GET  /home                推荐首页（顶部滚动推荐 + 购买相似 + 当月档期；登录后个性化）
    GET  /                    相似商品查询（阶段三能力，公开）
    GET  /search              智能推荐（需登录；支持"100块以内的家居装饰"这类中文查询）
    GET  /login  /logout      登录 / 注册 / 登出

接口：
    GET  /health                       健康检查
    GET  /api/home?month=              推荐首页（JSON，含顶部滚动推荐 carousel）
    GET  /similar?item_id=&topn=       相似商品查询（JSON）
    GET  /api/items                    索引规模与示例商品
    GET  /api/search?q=&llm=1          智能推荐（JSON，需登录；llm=1 强制大模型重新识别）
    GET  /api/disliked                 当前用户"不喜欢"的商品列表（JSON，需登录）
    POST /feedback                     点赞 / 点踩（JSON，需登录）
    POST /feedback/clear               清空某类反馈（JSON，需登录）
    POST /api/voice                    本地语音识别（WAV，需登录）
    POST /api/gesture                  本地手势识别（JPEG，需登录）
    POST /api/vision/analyze           图片视觉线索（multipart，需登录）
    POST /api/tts                      edge-tts 语音播报（JSON，需登录）

安全说明：
    1. 密码用 PBKDF2 加盐哈希存储，数据库不存明文（见 src/user_store.py）；
    2. 会话密钥首次运行自动生成并保存到 data/.secret_key，不写进代码仓库；
    3. 反馈接口校验登录态，未登录返回 401。
"""
from __future__ import annotations

import functools
import json
import math
import os
import secrets
import sys
from datetime import date
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from flask import (
    Flask,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from config.settings import HYPERPARAMS, PATHS
from src import feedback as fb
from src import home_recommend as hr
from src import item2vec as i2v
from src import multimodal_client as mmc
from src import product_catalog as catalog
from src import smart_search as ss
from src import user_store
from src import visual_intent as vi
from src.home_recommend import HomeRecommender
from src.item2vec import load_item2vec
from src.vector_index import ItemVectorIndex

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["JSON_AS_ASCII"] = False
# 语音/图片是二进制上传，给一个明确上限（约 24MB），避免超大请求拖垮进程
app.config["MAX_CONTENT_LENGTH"] = 24 * 1024 * 1024

DEFAULT_TOPN = 5
MAX_TOPN = 50
PAGE_SIZE = 5  # 智能推荐每页展示多少条
MAX_RANKED = 100  # 单次查询最多排序并保留多少条（用于翻页）


def _server_port() -> int:
    """读取推荐服务端口，默认保持 5000 以兼容原有命令和演示材料。"""
    raw = os.environ.get("RECOMMENDER_PORT", "5000").strip()
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"RECOMMENDER_PORT 必须是整数，当前为 {raw!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"RECOMMENDER_PORT 必须在 1~65535 之间，当前为 {port}")
    return port


SERVER_HOST = os.environ.get("RECOMMENDER_HOST", "127.0.0.1").strip() or "127.0.0.1"
SERVER_PORT = _server_port()
_MULTIMODAL_CFG = HYPERPARAMS.get("search", {}).get("multimodal", {})
MAX_VISION_UPLOAD_BYTES = int(
    _MULTIMODAL_CFG.get("vision_max_upload_bytes", 8 * 1024 * 1024))


def _load_secret_key() -> str:
    """会话密钥：优先环境变量，其次本地文件（首次自动生成）。"""
    env_key = os.environ.get("RECOMMENDER_SECRET_KEY")
    if env_key:
        return env_key
    path = Path(PATHS["user_db"]).parent / ".secret_key"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    key = secrets.token_hex(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key, encoding="utf-8")
    return key


app.secret_key = _load_secret_key()

# ---------------------------------------------------------------- 启动加载
INDEX: ItemVectorIndex | None = None
NAMES: dict[str, str] = {}
CATEGORIES: dict[str, str] = {}
DEMO_IDS: list[str] = []
SEARCHER: ss.SmartSearch | None = None
DEMO_CUSTOMERS: list[dict] = []
PROFILE_INDEX: dict[str, dict] = {}  # 商品ID → 名称/类目/均价/热度（反馈面板要用）
LOAD_ERROR: str | None = None
HOME: HomeRecommender | None = None  # 推荐首页（购买历史 + 季节统计，首次访问 /home 按需加载）
HOME_ERROR: str | None = None

# 项目二多模态桥接服务（本地 ASR / 手势 / 视觉 / 大模型 / TTS）。
# 进程内单例：共享健康探测缓存，未启动时所有调用都会降级，不影响页面其它功能。
MULTIMODAL = mmc.get_client()

# ------------------------------------------------------------ 商品图片
STATIC_IMG_DIR = ROOT / "web" / "static" / "img"
ITEM_IMG_DIR = STATIC_IMG_DIR / "items"
CATEGORY_MANIFEST = STATIC_IMG_DIR / "categories" / "manifest.json"
PHOTO_DIR = STATIC_IMG_DIR / "photos"  # 联网抓来的类目照片（scripts/15_fetch_web_images.py）
BANNER_DIR = STATIC_IMG_DIR / "banners"  # 同一张原图裁的横幅版（首页滚动推荐用）
PHOTO_MANIFEST = PHOTO_DIR / "manifest.json"
BANNER_MANIFEST = BANNER_DIR / "manifest.json"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _load_manifest(path: Path) -> dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 没有清单就退回兜底图
        return {}


def _scan_dir(directory: Path, manifest: dict[str, str]) -> dict[str, str]:
    """目录清单 = 磁盘上真实存在的文件（文件名 = 类目名）∪ 清单里仍能落地的条目。

    以磁盘为准，好处是**手工替换/新增的图片不需要重跑任何脚本**，
    ``/health`` 里的张数也永远和文件系统一致；清单只用来兼容"文件名不等于类目名"的旧产物。
    """
    found: dict[str, str] = {}
    if directory.exists():
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            if path.stem in CATEGORY_IMAGES or path.stem in manifest:
                found.setdefault(path.stem, path.name)
    for category, name in manifest.items():
        if category not in found and (directory / name).exists():
            found[category] = name
    return found


def _find_image(directory: Path, category: str, manifest: dict[str, str]) -> str | None:
    """在目录里按类目名找图：先看磁盘（不依赖清单），再退回清单里的文件名。

    这样**手工替换图片不需要跑任何脚本、也不用改代码**：
    把 ``手机.jpg`` / ``手机.png`` / ``手机.webp`` 丢进对应目录就直接生效
    （清单是 ``scripts/15_fetch_web_images.py`` 抓图时顺手写的，只是备用线索）。
    """
    if not category:
        return None
    for ext in IMAGE_EXTS:
        if (directory / f"{category}{ext}").exists():
            return f"{category}{ext}"
    name = manifest.get(category)
    if name and (directory / name).exists():
        return name
    return None


CATEGORY_IMAGES = _load_manifest(CATEGORY_MANIFEST)
PHOTO_IMAGES = _scan_dir(PHOTO_DIR, _load_manifest(PHOTO_MANIFEST))
BANNER_IMAGES = _scan_dir(BANNER_DIR, _load_manifest(BANNER_MANIFEST))


def image_url(item_id: str, category: str = "") -> str:
    """商品配图地址：真实商品图 → 联网抓的类目照片 → 本地画的类目示意图。

    三级回落，任何一级缺了都不影响页面：

        1. ``web/static/img/items/<商品ID>.png``   真实商品图（有就最优先，放进去即生效）
        2. ``web/static/img/photos/<类目>.jpg``    联网抓的类目照片（scripts/15）
        3. ``web/static/img/categories/<类目>.png``  本地画的类目示意图（scripts/10）
    """
    item_id = str(item_id)
    for ext in IMAGE_EXTS:
        if (ITEM_IMG_DIR / f"{item_id}{ext}").exists():
            return f"/static/img/items/{quote(item_id)}{ext}"
    category = str(category)
    photo = _find_image(PHOTO_DIR, category, PHOTO_IMAGES)
    if photo:
        return f"/static/img/photos/{quote(photo)}"
    slug = CATEGORY_IMAGES.get(category)
    if slug:
        return f"/static/img/categories/{quote(slug)}"
    return "/static/img/categories/unknown.png"


def banner_url(category: str) -> str:
    """首页滚动推荐的大图地址（拿不到照片时返回空串，前端改用渐变底 + 方图）。

    横幅照片是同一张原图按 16:9 裁的，所以它和卡片里的方图**是同源不同裁切**，
    页面上"轮播里的大图"和"卡片里的小图"看起来是一套东西。
    """
    category = str(category or "")
    banner = _find_image(BANNER_DIR, category, BANNER_IMAGES)
    if banner:
        return f"/static/img/banners/{quote(banner)}"
    photo = _find_image(PHOTO_DIR, category, PHOTO_IMAGES)
    if photo:
        return f"/static/img/photos/{quote(photo)}"
    return ""


def load_resources() -> None:
    """一次性加载索引、商品表与智能推荐器；失败时记录原因而不让服务崩溃。"""
    global INDEX, NAMES, CATEGORIES, DEMO_IDS, SEARCHER, DEMO_CUSTOMERS, LOAD_ERROR
    try:
        user_store.init_db(PATHS["user_db"])
        fb.init_db(PATHS["feedback_db"])

        INDEX = ItemVectorIndex().load(PATHS["vector_index_dir"])
        products = catalog.load_products()
        NAMES = catalog.build_name_map(products)
        CATEGORIES = catalog.build_category_map(products)

        if Path(PATHS["item2vec_model"]).exists():
            model = load_item2vec(PATHS["item2vec_model"])
            DEMO_IDS = [i for i in i2v.top_items_by_count(model, n=6) if INDEX.contains(i)]

        SEARCHER = ss.SmartSearch.load()
        DEMO_CUSTOMERS = ss.load_demo_customers()
        # 商品画像的字典视图：反馈面板要按商品ID快速取名称/类目/价格
        PROFILE_INDEX.clear()
        PROFILE_INDEX.update(
            {
                str(row.item_id): {
                    "name": str(row.name),
                    "category": str(row.category),
                    "price": float(row.price),
                    "users": int(row.users),
                }
                for row in SEARCHER.profile.itertuples(index=False)
            }
        )
        LOAD_ERROR = None
    except Exception as exc:  # noqa: BLE001 - 需要把原因原样报给调用方
        INDEX = None
        SEARCHER = None
        LOAD_ERROR = f"{type(exc).__name__}: {exc}"


load_resources()


# ------------------------------------------------------------------ 辅助
def current_user() -> dict | None:
    """当前登录用户（未登录返回 None）。"""
    username = session.get("username")
    if not username:
        return None
    return user_store.get_user(PATHS["user_db"], username)


def nav(active: str = "") -> dict:
    """所有页面共用的导航上下文。"""
    user = current_user()
    return {
        "username": user["username"] if user else "",
        "display_name": user["display_name"] if user else "",
        "active": active,
    }


def disliked_items(username: str) -> list[dict]:
    """当前用户点过「不喜欢」的商品，最近点的排在前面。

    商品名 / 类目 / 价格取自智能推荐的商品画像；画像里没有的商品（例如后来才被
    过滤掉的编码）就退回商品主数据，保证列表里任何一条都能显示得出来。
    """
    if not username:
        return []
    records = fb.list_feedback(PATHS["feedback_db"], username, action="dislike")
    items: list[dict] = []
    for record in records:
        item_id = record["item_id"]
        meta = PROFILE_INDEX.get(item_id, {})
        category = meta.get("category") or CATEGORIES.get(item_id, catalog.UNKNOWN_CATEGORY)
        items.append(
            {
                "item_id": item_id,
                "name": meta.get("name") or NAMES.get(item_id, catalog.UNKNOWN_NAME),
                "category": category,
                "price": meta.get("price"),
                "users": meta.get("users"),
                "updated_at": record.get("updated_at", ""),
                "image": image_url(item_id, category),
            }
        )
    return items


def login_required(view):
    """需要登录的页面：未登录跳转到登录页并记住来源。"""

    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("username"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapper


def api_login_required(view):
    """需要登录的接口：未登录返回 401。"""

    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("username"):
            return jsonify({"error": "请先登录"}), 401
        return view(*args, **kwargs)

    return wrapper


# ------------------------------------------------------------------ 登录
@app.route("/login", methods=["GET", "POST"])
def login():
    """登录 / 注册二合一页面。"""
    # mode 既可能来自表单提交，也可能来自链接上的查询参数（"去注册"按钮）
    mode = request.values.get("mode", "login")
    if mode not in ("login", "register"):
        mode = "login"
    error = ""
    message = ""

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if mode == "register":
            display_name = (request.form.get("display_name") or "").strip()
            customer_id = (request.form.get("customer_id") or "").strip()
            try:
                user_store.create_user(
                    PATHS["user_db"], username, password, display_name, customer_id
                )
                session["username"] = username
                return redirect(request.args.get("next") or url_for("home_page"))
            except ValueError as exc:
                error = str(exc)
        else:
            user = user_store.verify_user(PATHS["user_db"], username, password)
            if user:
                session["username"] = user["username"]
                return redirect(request.args.get("next") or url_for("home_page"))
            error = "用户名或密码不正确"

    return render_template(
        "login.html",
        **nav("login"),
        mode=mode,
        error=error,
        message=message,
        demo_customers=DEMO_CUSTOMERS,
        user_count=user_store.count_users(PATHS["user_db"]),
        load_error=LOAD_ERROR,
    )


@app.get("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("home_page"))


# ------------------------------------------------------- 相似商品（阶段三）
@app.get("/")
def index_page():
    """相似商品查询页面（公开，保持阶段三的交付不变）。"""
    item_id = (request.args.get("item_id") or "").strip()
    raw_topn = request.args.get("topn", str(DEFAULT_TOPN))
    result = None
    query_done = False
    error = LOAD_ERROR

    if item_id and INDEX is not None:
        query_done = True
        try:
            topn = int(raw_topn)
        except (TypeError, ValueError):
            topn = DEFAULT_TOPN
        topn = max(1, min(topn, MAX_TOPN))
        try:
            pairs = INDEX.search(item_id, topn=topn)
            enriched = catalog.enrich_results(pairs, None, names=NAMES, categories=CATEGORIES)
            for row in enriched:
                row["image"] = image_url(row["id"], row.get("category", ""))
            result = {
                "item_id": item_id,
                "item_name": NAMES.get(item_id, catalog.UNKNOWN_NAME),
                "item_category": CATEGORIES.get(item_id, catalog.UNKNOWN_CATEGORY),
                "item_image": image_url(item_id, CATEGORIES.get(item_id, "")),
                "count": len(pairs),
                "results": enriched,
            }
        except KeyError:
            error = (
                f"商品 {item_id} 不在向量索引中。"
                f"该商品可能被 min_count 过滤，或不是有效商品编码。"
            )

    return render_template(
        "index.html",
        **nav("similar"),
        demo_ids=DEMO_IDS,
        index_size=len(INDEX) if INDEX else 0,
        error=error,
        result=result,
        query_done=query_done,
        form_item_id=item_id or "85123A",
        form_topn=raw_topn,
    )


# ------------------------------------------------------------ 推荐首页
def ensure_home() -> HomeRecommender | None:
    """首页推荐器按需加载：首次要读购买历史与季节统计（约 2 秒），之后常驻内存。"""
    global HOME, HOME_ERROR
    if HOME is not None:
        return HOME
    try:
        HOME = hr.HomeRecommender.load()
        HOME_ERROR = None
    except Exception as exc:  # noqa: BLE001 - 把失败原因原样报给页面
        HOME_ERROR = f"{type(exc).__name__}: {exc}"
    return HOME


def parse_month(raw: str | None) -> int:
    """首页档期月份：默认当前月，``?month=9`` 可指定（演示与自测用）。"""
    if raw is None or str(raw).strip() == "":
        return date.today().month
    try:
        month = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"month 必须是 1~12 的整数，收到 {raw!r}") from exc
    if month < 1 or month > 12:
        raise ValueError(f"month 必须在 1~12 之间，收到 {month}")
    return month


def decorate_home(data: dict) -> dict:
    """给首页数据里的每一行商品补上配图（模板直接用现成的 URL）。"""
    for row in data.get("history_rows", []):
        row["image"] = image_url(row["item_id"], row.get("category", ""))
    for section in data.get("sections", []):
        for row in section.get("rows", []):
            row["image"] = image_url(row["item_id"], row.get("category", ""))
    for slide in data.get("carousel", []):
        category = slide.get("category", "")
        slide["image"] = banner_url(category)
        slide["thumb"] = image_url(slide.get("item_id", ""), category)
        # 点一条轮播 = 看它的向量近邻（阶段三那个公开页面，未登录也能看）
        slide["href"] = f"/similar?item_id={quote(str(slide.get('item_id', '')))}&topn=6"
    return data


def build_home(month: int | None = None) -> tuple[dict | None, str | None]:
    """构建首页数据；返回 ``(数据, 错误信息)``，两者必有一为 None。"""
    rec = ensure_home()
    if rec is None:
        return None, HOME_ERROR or "首页推荐未就绪"
    try:
        data = rec.build(username=session.get("username"), month=month)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    return decorate_home(data), None


@app.get("/home")
def home_page():
    """推荐首页：左边是"基于你买过的商品"，上面是"这个月该推什么"。

    未登录也能看：档期板块是公共的，个性化板块会提示登录/关联演示客户。
    """
    errors: list[str] = []
    if LOAD_ERROR:
        errors.append(f"部分资源未就绪（{LOAD_ERROR}），首页可能不完整")
    try:
        month = parse_month(request.args.get("month"))
    except ValueError as exc:
        errors.append(str(exc))
        month = date.today().month

    data, error = build_home(month)
    if error:
        errors.append(error)

    return render_template(
        "home.html",
        **nav("home"),
        data=data,
        errors=errors,
        month=month,
        demo_customers=DEMO_CUSTOMERS,
        feedback_stats=(
            fb.summarize(fb.get_feedback(PATHS["feedback_db"], session["username"]))
            if session.get("username")
            else {"liked": 0, "disliked": 0, "total": 0}
        ),
    )


@app.get("/api/home")
def api_home():
    """推荐首页的 JSON 版本（公开；带登录 Cookie 时会返回个性化板块）。"""
    try:
        month = parse_month(request.args.get("month"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    data, error = build_home(month)
    if data is None:
        return jsonify({"error": "首页推荐未就绪", "detail": error}), 503
    return jsonify(data)


# ------------------------------------------------------------ 智能推荐
@app.get("/search")
@login_required
def search_page():
    """智能推荐页面：解析中文查询并返回带理由的个性化结果。"""
    user = current_user()
    from_vision = request.args.get("from") == "vision"
    query = (request.args.get("q") or "").strip()
    if from_vision:
        # 视觉选品只按图片中的物品与类目推荐；即使手动编辑时残留了价格，
        # 也不让视觉入口生成价格筛选条件。
        query = vi.strip_price_constraints(query)
    # llm=1：用户点了「大模型识别」按钮，这次强制请大模型重新解析一遍
    force_llm = (request.args.get("llm") or "").strip().lower() in ("1", "true", "yes", "on")
    result = None
    error = LOAD_ERROR
    page = 1
    total_pages = 1

    if query and SEARCHER is not None:
        try:
            # 一次多排一些（最多 MAX_RANKED 条），页面上再按页切
            full = SEARCHER.search(
                query, username=user["username"], top_n=MAX_RANKED, force_llm=force_llm
            )
            for row in full["results"]:
                row["image"] = image_url(row["item_id"], row.get("category", ""))

            ranked = full["results"]
            total_pages = max(1, math.ceil(len(ranked) / PAGE_SIZE))
            try:
                page = int(request.args.get("page", 1))
            except (TypeError, ValueError):
                page = 1
            page = max(1, min(page, total_pages))

            full["ranked_total"] = len(ranked)
            full["results"] = ranked[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
            result = full
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

    stats = fb.summarize(fb.get_feedback(PATHS["feedback_db"], user["username"]))
    return render_template(
        "search.html",
        **nav("search"),
        query=query,
        result=result,
        error=error,
        page=page,
        total_pages=total_pages,
        page_size=PAGE_SIZE,
        weights=(SEARCHER.weights if SEARCHER else {}),
        from_voice=(request.args.get("from") == "voice"),
        from_vision=from_vision,
        llm_forced=force_llm,
        disliked_items=disliked_items(user["username"]),
        show_dislike=(request.args.get("show") == "dislike"),
        examples=[
            "100块钱以内的家居装饰",
            "不超过3元的厨房用品",
            "20到50元 收纳",
            "红色的购物袋",
            "派对用品",
        ],
        feedback_stats=stats,
        load_error=LOAD_ERROR,
        # 项目二多模态桥接状态：页面据此决定语音/手势走哪条通道并显示徽章
        multimodal=MULTIMODAL.status(),
    )


# ------------------------------------------------------------ 反馈接口
@app.post("/feedback")
@api_login_required
def feedback_api():
    """记录点赞 / 点踩。重复点同一个动作表示取消。"""
    payload = request.get_json(silent=True) or {}
    item_id = str(payload.get("item_id") or "").strip()
    action = str(payload.get("action") or "").strip()

    if not item_id:
        return jsonify({"error": "缺少 item_id"}), 400
    if action not in fb.ACTIONS:
        return jsonify({"error": f"action 必须是 {fb.ACTIONS} 之一"}), 400

    username = session["username"]
    try:
        state = fb.set_feedback(PATHS["feedback_db"], username, item_id, action)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    stats = fb.summarize(fb.get_feedback(PATHS["feedback_db"], username))
    label = {"like": "喜欢", "dislike": "不喜欢"}.get(state or "", "")
    hint = f"已记录「{label}」，下次搜索会按你的反馈重新排序" if state else "已取消该反馈"
    return jsonify(
        {
            "item_id": item_id,
            "state": state,
            "item_name": NAMES.get(item_id, catalog.UNKNOWN_NAME),
            "stats": stats,
            "hint": hint,
        }
    )


@app.post("/feedback/clear")
@api_login_required
def feedback_clear_api():
    """一次清空某类反馈（页面上"清空全部不喜欢"用）。

    ``action`` 取 ``dislike``（默认）/ ``like`` / ``all``。
    """
    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action") or "dislike").strip()
    if action not in ("dislike", "like", "all"):
        return jsonify({"error": "action 必须是 dislike / like / all 之一"}), 400

    username = session["username"]
    try:
        if action == "all":
            removed = fb.clear_feedback(PATHS["feedback_db"], username)
        else:
            removed = fb.clear_feedback_by_action(PATHS["feedback_db"], username, action)
    except Exception as exc:  # noqa: BLE001 - 把失败原因原样报给页面
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    stats = fb.summarize(fb.get_feedback(PATHS["feedback_db"], username))
    label = {"dislike": "不喜欢", "like": "喜欢", "all": "全部"}[action]
    return jsonify(
        {
            "action": action,
            "removed": removed,
            "stats": stats,
            "hint": f"已清空 {removed} 条「{label}」记录" if removed else f"没有「{label}」记录需要清空",
        }
    )


@app.get("/api/disliked")
@api_login_required
def api_disliked():
    """当前用户"不喜欢"的商品列表（JSON）。"""
    items = disliked_items(session["username"])
    return jsonify({"count": len(items), "results": items})


# ------------------------------------------------- 项目二多模态桥接（语音/手势/播报）
def _multimodal_error(exc: Exception, *, fallback: str = "builtin"):
    """把桥接客户端的异常翻译成 ``(响应, 状态码)``。

    桥接服务没启动（``BridgeUnavailable``）→ 503，并明确告诉前端"用内置通道"；
    桥接服务自己拒绝了请求（``BridgeError``）→ 原样透传它的状态码。
    两种情况页面都不会 500。
    """
    if isinstance(exc, mmc.BridgeUnavailable):
        return (
            jsonify(
                {
                    "error": str(exc),
                    "fallback": fallback,
                    "hint": (
                        "多模态桥接服务未启动；文字与语音推荐仍可正常使用。"
                        "启动方式：cd multimodal_ai_project1 && python bridge_server.py"
                    ),
                }
            ),
            503,
        )
    status = int(getattr(exc, "status", 502) or 502)
    text_only = fallback == "text-only"
    return (
        jsonify(
            {
                "error": str(exc),
                "hint": (
                    "视觉服务未返回有效结果；可继续使用文字或语音搜索。"
                    if text_only
                    else "本地多模态服务返回错误，可继续使用页面内置能力。"
                ),
                "fallback": "text-only" if text_only else "builtin",
            }
        ),
        status if 400 <= status < 600 else 502,
    )


def _binary_body() -> bytes:
    """读取上传的二进制体：既支持 raw body，也兼容 multipart 表单里的文件。"""
    # multipart 的表单正文里还包含 boundary 和字段名，必须先取真正的文件部分。
    if request.files:
        first = next(iter(request.files.values()))
        data = first.read()
    else:
        data = request.get_data(cache=False, as_text=False)
    return data or b""


@app.get("/api/multimodal/status")
@api_login_required
def api_multimodal_status():
    """多模态桥接服务的能力摘要：页面据此决定用哪条通道并渲染徽章。"""
    return jsonify(MULTIMODAL.status())


@app.post("/api/voice")
@api_login_required
def api_voice():
    """本地语音识别（项目二 SenseVoiceSmall）：收 WAV 字节 → 返回识别文本。"""
    audio = _binary_body()
    if len(audio) < 128:
        return jsonify({"error": "没有收到有效的音频数据（WAV 太小或为空）"}), 400
    try:
        result = MULTIMODAL.transcribe(audio)
    except (mmc.BridgeUnavailable, mmc.BridgeError) as exc:
        return _multimodal_error(exc)
    text = str(result.get("text") or "").strip()
    return jsonify(
        {
            "text": text,
            "engine": result.get("engine", "local-asr"),
            "elapsed_ms": result.get("elapsed_ms", 0),
            "empty": not bool(text),
        }
    )


@app.post("/api/gesture")
@api_login_required
def api_gesture():
    """MediaPipe 手势单帧判定（项目二）：收 JPEG 字节 → 返回手势名与中文标签。"""
    image = _binary_body()
    if not image:
        return jsonify({"error": "没有收到图像数据"}), 400
    try:
        result = MULTIMODAL.gesture(image)
    except (mmc.BridgeUnavailable, mmc.BridgeError) as exc:
        return _multimodal_error(exc)
    result["engine"] = "mediapipe"
    return jsonify(result)


@app.post("/api/tts")
@api_login_required
def api_tts():
    """语音合成（项目二 edge-tts）：收 ``{"text": ...}`` → 回传 MP3 字节流。"""
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "缺少 text"}), 400
    try:
        audio = MULTIMODAL.tts(text)
    except (mmc.BridgeUnavailable, mmc.BridgeError) as exc:
        return _multimodal_error(exc)
    response = make_response(audio)
    response.headers["Content-Type"] = "audio/mpeg"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/api/vision/analyze")
@api_login_required
def api_vision_analyze():
    """图片 → 可见场景与类目线索 → 可编辑的推荐查询（不自动搜索）。"""
    if SEARCHER is None:
        return jsonify(
            {"error": "智能推荐未就绪", "hint": "请先准备商品画像与向量索引"}
        ), 503

    declared = request.content_length
    if declared is not None and declared > MAX_VISION_UPLOAD_BYTES + 128 * 1024:
        return jsonify(
            {
                "error": "图片过大",
                "hint": f"请上传不超过 {MAX_VISION_UPLOAD_BYTES // 1024 // 1024}MB 的图片",
            }
        ), 413

    image = _binary_body()
    if not image:
        return jsonify(
            {
                "error": "没有收到图片",
                "hint": "请使用 multipart/form-data 上传字段 image",
            }
        ), 400
    if len(image) > MAX_VISION_UPLOAD_BYTES:
        return jsonify(
            {
                "error": "图片过大",
                "hint": f"请上传不超过 {MAX_VISION_UPLOAD_BYTES // 1024 // 1024}MB 的图片",
            }
        ), 413
    if not (
        image.startswith(b"\xff\xd8\xff")
        or image.startswith(b"\x89PNG\r\n\x1a\n")
    ):
        return jsonify(
            {
                "error": "图片格式不支持",
                "hint": "请上传 JPEG 或 PNG 图片",
            }
        ), 400

    try:
        raw = MULTIMODAL.vision(
            image,
            categories=list(SEARCHER.categories),
        )
    except (mmc.BridgeUnavailable, mmc.BridgeError) as exc:
        return _multimodal_error(exc, fallback="text-only")

    analysis = vi.normalize_analysis(raw, SEARCHER.categories)
    # 视觉结果独立于文字搜索框：不复用其中的价格或旧关键词，只生成物品条件。
    suggested_query = vi.compose_search_query("", analysis)
    model_name = str(raw.get("model") or "")
    if not model_name:
        model_name = str(MULTIMODAL.status().get("vision_model", ""))
    return jsonify(
        {
            "vision": analysis,
            "suggested_query": suggested_query,
            "has_visual_terms": vi.has_visual_terms(analysis),
            "model": model_name,
            "elapsed_ms": int(raw.get("elapsed_ms", 0) or 0),
        }
    )


# ------------------------------------------------------------ JSON 接口
@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok" if INDEX is not None else "degraded",
            "index_loaded": INDEX is not None,
            "index_size": len(INDEX) if INDEX else 0,
            "vector_dim": INDEX.dim if INDEX else 0,
            "products_loaded": len(NAMES),
            "search_ready": SEARCHER is not None,
            "search_items": len(SEARCHER.profile) if SEARCHER else 0,
            "home_ready": HOME is not None,
            "home_loaded_customers": len(HOME.history) if HOME else 0,
            "home_error": HOME_ERROR,
            "home_images": {
                "photos": len(PHOTO_IMAGES),
                "banners": len(BANNER_IMAGES),
                "category_tiles": len(CATEGORY_IMAGES),
            },
            "users": user_store.count_users(PATHS["user_db"]),
            # 项目二多模态桥接：只做一次带缓存的短超时探测，服务没起也不报错
            "multimodal": MULTIMODAL.status(),
            "error": LOAD_ERROR,
        }
    )


@app.get("/api/items")
def api_items():
    if INDEX is None:
        return jsonify({"error": "索引未加载", "detail": LOAD_ERROR}), 503
    return jsonify(
        {
            "index_size": len(INDEX),
            "vector_dim": INDEX.dim,
            "demo_ids": [
                {"id": i, "name": NAMES.get(i, catalog.UNKNOWN_NAME)} for i in DEMO_IDS
            ],
        }
    )


@app.get("/similar")
def similar():
    if INDEX is None:
        return (
            jsonify(
                {
                    "error": "向量索引未加载",
                    "detail": LOAD_ERROR,
                    "hint": "请先运行 scripts/04_build_vector_index.py 构建索引。",
                }
            ),
            503,
        )

    item_id = (request.args.get("item_id") or "").strip()
    if not item_id:
        return jsonify({"error": "缺少参数 item_id", "example": "/similar?item_id=85123A&topn=5"}), 400

    raw_topn = request.args.get("topn", str(DEFAULT_TOPN))
    try:
        topn = int(raw_topn)
    except (TypeError, ValueError):
        return jsonify({"error": f"topn 必须是整数，收到 {raw_topn!r}"}), 400
    if topn <= 0 or topn > MAX_TOPN:
        return jsonify({"error": f"topn 必须在 1~{MAX_TOPN} 之间，收到 {topn}"}), 400

    try:
        pairs = INDEX.search(item_id, topn=topn)
    except KeyError as exc:
        return (
            jsonify(
                {
                    "error": "商品不在向量索引中",
                    "detail": str(exc),
                    "item_id": item_id,
                    "hint": "该商品可能被 min_count=5 过滤，或不是有效商品编码。",
                }
            ),
            404,
        )

    return jsonify(
        {
            "item_id": item_id,
            "item_name": NAMES.get(item_id, catalog.UNKNOWN_NAME),
            "item_category": CATEGORIES.get(item_id, catalog.UNKNOWN_CATEGORY),
            "item_image": image_url(item_id, CATEGORIES.get(item_id, "")),
            "topn": topn,
            "count": len(pairs),
            "results": [
                {**row, "image": image_url(row["id"], row.get("category", ""))}
                for row in catalog.enrich_results(pairs, None, names=NAMES, categories=CATEGORIES)
            ],
        }
    )


@app.get("/api/search")
@api_login_required
def api_search():
    """智能推荐的 JSON 版本，便于外部系统（如项目三）调用。"""
    if SEARCHER is None:
        return jsonify({"error": "智能推荐未就绪", "detail": LOAD_ERROR}), 503

    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "缺少参数 q", "example": "/api/search?q=100块以内的家居装饰"}), 400

    try:
        top_n = int(request.args.get("top_n", MAX_RANKED))
    except (TypeError, ValueError):
        return jsonify({"error": "top_n 必须是整数"}), 400

    force_llm = (request.args.get("llm") or "").strip().lower() in ("1", "true", "yes", "on")
    result = SEARCHER.search(
        query,
        username=session["username"],
        top_n=max(1, min(top_n, 100)),
        force_llm=force_llm,
    )
    for row in result["results"]:
        row["image"] = image_url(row["item_id"], row.get("category", ""))
    return jsonify(result)


@app.errorhandler(404)
def handle_404(_):
    return (
        jsonify(
            {
                "error": "接口不存在",
                "hint": (
                    "可用页面/接口：/home 、/ 、/search 、/login 、/health 、/similar 、"
                    "/api/home 、/api/items 、/api/search 、/api/disliked 、/feedback 、"
                    "/feedback/clear 、/api/multimodal/status 、/api/voice 、"
                    "/api/gesture 、/api/vision/analyze 、/api/tts"
                ),
            }
        ),
        404,
    )


if __name__ == "__main__":
    if LOAD_ERROR:
        print(f"[警告] 资源加载失败：{LOAD_ERROR}")
    else:
        print(
            f"[OK] 向量索引 {len(INDEX)} 个商品 | "
            f"智能推荐 {len(SEARCHER.profile)} 件商品 / {len(SEARCHER.categories)} 个类目 | "
            f"已注册用户 {user_store.count_users(PATHS['user_db'])}"
        )
    print(f"服务启动中： http://{SERVER_HOST}:{SERVER_PORT}/")
    print(
        f"推荐首页： http://{SERVER_HOST}:{SERVER_PORT}/home"
        "   （首次打开会加载购买历史与季节统计）"
    )
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)
