"""季节档期：月份 → 档期主题（业务规则）+ 季节性指数（数据事实）。

推荐首页要回答一个很朴素的问题：**"这个季节该给用户推什么？"**
这件事由两层拼起来，两层各自负责不同的东西：

    1. **档期日历**（``SEASON_THEMES``，人工维护的业务规则）
       每个月有哪些档期（开学季 / 秋季换季 / 双 11 / 圣诞…）、每个档期该覆盖
       哪些类目、页面上用什么话术。这是"推什么"的口径。

    2. **季节性指数**（:class:`SeasonTable`，从交易数据里算出来的事实）
       ``季节指数 = 该类目当月的销量占比 / 该类目全年的销量占比``。
       大于 1 说明"这个类目在这个月比平时更旺"。
       这是"谁更靠前"的口径，也是"9 月数码产品卖得好"这句话的数据证据。

两层的分工是刻意的：日历不会因为数据里某个类目当月恰好没货就漏掉整个档期，
指数也不会因为写日历的人想当然就把一个淡季类目顶到首页。

季节性指数带**样本量保护**：当月成交件数少于 ``season_min_events`` 的类目一律按
指数 1.0（"没有可用信号"）处理，避免"全库只有 3 件商品的小类目"因为
恰好 9 月卖过一次就把指数刷到 9 倍。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from config.settings import HYPERPARAMS, PATHS
from src import product_catalog as catalog
from src.data_cleaning import DATE_FORMAT, TRANSACTIONS_ENCODING

DEFAULT_LIFT = 1.0

# ------------------------------------------------------------------ 档期日历
# 每个月最多两个档期：一个"当下最该推"的主档期，一个顺带的次档期。
# 想调整档期只需要改这里，推荐逻辑不用动。
SEASON_THEMES: dict[int, list[dict[str, Any]]] = {
    1: [
        {
            "key": "winter_warm",
            "name": "冬季保暖",
            "tagline": "1 月最冷，保暖类目是刚需",
            "categories": ["生活用品", "家用电器", "洗浴用品", "鞋靴箱包", "配饰"],
            "keywords": ["暖", "保温", "毯", "手套", "帽", "热水"],
        },
        {
            "key": "newyear_home",
            "name": "新年焕新",
            "tagline": "年前居家焕新与囤货",
            "categories": ["家居装饰", "厨房用品", "收纳用品", "装饰品"],
            "keywords": ["收纳", "灯", "杯", "装饰"],
        },
    ],
    2: [
        {
            "key": "valentine_gift",
            "name": "情人节礼赠",
            "tagline": "2 月礼赠类目溢价高",
            "categories": ["珠宝首饰", "饰品", "美妆护肤", "派对用品", "食品饮料"],
            "keywords": ["心形", "爱心", "礼", "玫瑰", "巧克力"],
        },
        {
            "key": "winter_end",
            "name": "冬末清仓",
            "tagline": "冬品尾货 + 春季预热",
            "categories": ["生活用品", "鞋靴箱包", "配饰", "家居装饰"],
            "keywords": ["暖", "保温", "围巾"],
        },
    ],
    3: [
        {
            "key": "spring_home",
            "name": "春季焕新",
            "tagline": "换季收纳与家居焕新",
            "categories": ["家居装饰", "生活用品", "美妆护肤", "收纳用品", "配饰"],
            "keywords": ["收纳", "收纳盒", "装饰", "花", "清新"],
        },
        {
            "key": "spring_outdoor",
            "name": "春游户外",
            "tagline": "天气转暖，户外与园艺开始起量",
            "categories": ["运动户外", "园艺用品", "花园用品", "工具", "汽车用品"],
            "keywords": ["户外", "花园", "园艺", "运动", "工具"],
        },
    ],
    4: [
        {
            "key": "spring_outdoor",
            "name": "春游户外",
            "tagline": "户外、园艺与野餐用品旺季",
            "categories": ["运动户外", "园艺用品", "花园用品", "食品饮料", "购物袋"],
            "keywords": ["户外", "花园", "园艺", "野餐", "运动"],
        },
        {
            "key": "spring_home",
            "name": "换季收纳",
            "tagline": "厚衣服收起来，收纳与家居整理需求旺",
            "categories": ["收纳用品", "家居装饰", "生活用品", "厨房用品"],
            "keywords": ["收纳", "整理", "储物", "箱"],
        },
    ],
    5: [
        {
            "key": "mothers_day_gift",
            "name": "母亲节礼赠",
            "tagline": "5 月第二个周日，礼赠类目集中出货",
            "categories": ["珠宝首饰", "美妆护肤", "饰品", "食品饮料", "家居装饰"],
            "keywords": ["礼", "花", "心形", "首饰"],
        },
        {
            "key": "early_summer",
            "name": "初夏换装",
            "tagline": "入夏前的换装与出行准备",
            "categories": ["配饰", "鞋靴箱包", "美妆护肤", "运动户外", "洗浴用品"],
            "keywords": ["夏", "凉", "防晒", "轻便"],
        },
    ],
    6: [
        {
            "key": "summer_cool",
            "name": "夏季清凉",
            "tagline": "6 月入夏，清凉类目起量",
            "categories": ["洗浴用品", "食品饮料", "运动户外", "家用电器", "生活用品"],
            "keywords": ["凉", "冰", "风扇", "防晒", "夏"],
        },
        {
            "key": "midyear_sale",
            "name": "年中大促",
            "tagline": "618 前后是囤货窗口",
            "categories": [
                "手机", "笔记本电脑", "耳机音响", "智能穿戴", "家用电器",
                "生活用品", "厨房用品", "收纳用品", "家居装饰",
            ],
            "keywords": ["囤", "套装", "收纳", "大容量"],
        },
    ],
    7: [
        {
            "key": "summer_travel",
            "name": "暑期出行",
            "tagline": "暑假出行带动箱包、户外与汽车用品",
            "categories": ["运动户外", "汽车用品", "鞋靴箱包", "食品饮料", "购物袋"],
            "keywords": ["旅行", "包", "户外", "便携", "车载"],
        },
        {
            "key": "summer_cool",
            "name": "夏季清凉",
            "tagline": "高温持续，清凉类目仍在高位",
            "categories": ["洗浴用品", "食品饮料", "家用电器", "生活用品"],
            "keywords": ["凉", "冰", "风扇", "遮阳"],
        },
    ],
    8: [
        {
            "key": "back_to_school",
            "name": "开学季预热",
            "tagline": "8 月下旬起，数码与文具开始备货",
            "categories": [
                "手机", "笔记本电脑", "平板电脑", "耳机音响", "智能穿戴", "数码配件",
                "智能设备", "图书音像", "文具用品", "办公用品", "家用电器",
            ],
            "keywords": ["笔", "书", "背包", "耳机", "键盘", "充电", "台灯"],
        },
        {
            "key": "summer_cool",
            "name": "夏末清凉",
            "tagline": "暑期尾巴 + 军训出行",
            "categories": ["洗浴用品", "食品饮料", "运动户外", "生活用品"],
            "keywords": ["凉", "防晒", "户外", "水杯"],
        },
    ],
    9: [
        {
            "key": "back_to_school",
            "name": "9 月开学季",
            "tagline": "开学季数码、文具、图书集中出货",
            "categories": [
                "手机", "笔记本电脑", "平板电脑", "耳机音响", "智能穿戴", "数码配件",
                "智能设备", "家用电器", "图书音像", "文具用品", "办公用品",
            ],
            "keywords": ["笔", "书", "背包", "耳机", "键盘", "充电", "台灯", "笔记本"],
        },
        {
            "key": "autumn_change",
            "name": "秋季换季",
            "tagline": "气温转凉，家居、床品与配饰进入换季窗口",
            "categories": [
                "家居装饰", "生活用品", "鞋靴箱包", "配饰", "美妆护肤", "厨房用品", "收纳用品",
            ],
            "keywords": ["暖", "保温", "毯", "围巾", "手套", "拖鞋", "加湿", "收纳"],
        },
    ],
    10: [
        {
            "key": "autumn_change",
            "name": "秋季焕新",
            "tagline": "秋季家居焕新与保暖用品",
            "categories": ["家居装饰", "生活用品", "鞋靴箱包", "配饰", "美妆护肤", "厨房用品"],
            "keywords": ["暖", "保温", "毯", "拖鞋", "加湿", "装饰"],
        },
        {
            "key": "double11_prep",
            "name": "双 11 预热",
            "tagline": "大促前的囤货与备货窗口",
            "categories": [
                "手机", "耳机音响", "智能穿戴", "家用电器",
                "生活用品", "收纳用品", "厨房用品", "食品饮料", "购物袋",
            ],
            "keywords": ["囤", "套装", "收纳", "大容量"],
        },
    ],
    11: [
        {
            "key": "double11",
            "name": "双 11 大促",
            "tagline": "全年成交峰值，囤货与礼赠并行",
            "categories": [
                "手机", "笔记本电脑", "平板电脑", "耳机音响", "智能穿戴", "家用电器",
                "生活用品", "装饰品", "家居装饰", "收纳用品", "厨房用品", "食品饮料",
            ],
            "keywords": ["囤", "套装", "礼", "装饰"],
        },
        {
            "key": "winter_warm",
            "name": "初冬保暖",
            "tagline": "入冬后保暖类目快速上量",
            "categories": ["生活用品", "家用电器", "鞋靴箱包", "配饰", "洗浴用品"],
            "keywords": ["暖", "保温", "毯", "热水", "手套"],
        },
    ],
    12: [
        {
            "key": "christmas",
            "name": "圣诞礼赠",
            "tagline": "12 月装饰、派对与礼赠类目全面爆发",
            "categories": ["装饰品", "派对用品", "玩具", "珠宝首饰", "食品饮料", "家居装饰"],
            "keywords": ["圣诞", "礼", "装饰", "灯", "挂饰"],
        },
        {
            "key": "year_end",
            "name": "年终迎新",
            "tagline": "年底居家焕新与来年备货",
            "categories": ["家居装饰", "厨房用品", "收纳用品", "生活用品", "办公用品"],
            "keywords": ["收纳", "装饰", "杯", "台历"],
        },
    ],
}


def all_themes() -> list[dict[str, Any]]:
    """所有档期主题（去重，保留首次出现的月份）。"""
    seen: dict[str, dict[str, Any]] = {}
    for month in sorted(SEASON_THEMES):
        for theme in SEASON_THEMES[month]:
            seen.setdefault(theme["key"], theme)
    return list(seen.values())


def themes_for(month: int, max_themes: int | None = None) -> list[dict[str, Any]]:
    """返回某个月要展示的档期主题（按日历顺序）。"""
    month = int(month)
    if month < 1 or month > 12:
        raise ValueError(f"month 必须在 1~12 之间，收到 {month}")
    max_themes = int(max_themes if max_themes is not None else HYPERPARAMS["home"]["max_themes"])
    themes = [dict(t) for t in SEASON_THEMES[month]]
    return themes[:max_themes] if max_themes > 0 else themes


def theme_by_key(key: str) -> dict[str, Any] | None:
    for theme in all_themes():
        if theme["key"] == key:
            return dict(theme)
    return None


# ------------------------------------------------------------------ 季节统计
SEASON_STATS_COLUMNS = [
    "month",
    "category",
    "events",
    "qty",
    "amount",
    "days_covered",
    "daily_amount",
    "month_share",
    "lift",
]


# ------------------------------------------------------------------ 统计窗口
def source_window(path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """单个交易文件的时间范围（只读日期列，几十万行也很快）。"""
    try:
        raw = pd.read_csv(
            path, usecols=["InvoiceDate"], dtype=str, encoding=TRANSACTIONS_ENCODING
        )
    except (FileNotFoundError, ValueError):
        return None
    dates = pd.to_datetime(raw["InvoiceDate"], format=DATE_FORMAT, errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


def common_window(
    paths: Sequence[Path],
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """多个数据源**都覆盖**的时间窗口（最晚的开始 ~ 最早的结束）。

    为什么需要它：本项目的交易表 = 原始数据 + 补充数据。原始数据从 2010-12 开始，
    补充数据从 2011-01 才开始，于是 2010-12 这个月**只有原始数据**。如果拿它和
    其他月份比"类目占比"，廉价日用品的份额会凭空变高 —— 这不是季节性，是数据源
    没覆盖造成的假象。所以季节统计只保留所有数据源都覆盖的那段窗口。
    """
    windows = [w for w in (source_window(Path(p)) for p in paths) if w is not None]
    if len(windows) < 2:
        return None  # 只有一个数据源就不存在"覆盖不一致"的问题
    start = max(w[0] for w in windows)
    end = min(w[1] for w in windows)
    return (start, end) if start <= end else None


def build_season_stats(
    clean_df: pd.DataFrame,
    products: pd.DataFrame,
    min_events: int | None = None,
    window: tuple[Any, Any] | None = None,
) -> pd.DataFrame:
    """从干净交易表算"月份 × 类目"的季节性指数。

    口径：
        ``日均成交额 = 该类目当月成交额 / 当月覆盖天数``
        ``季节指数 lift = 日均成交额 / 该类目全年日均成交额``

    指数 1.0 = 这个类目在这个月**和它自己的平时一样**；1.7 = 比平时旺 70%。
    用"日均"而不是"月度总量"，是为了处理两个真实问题：

    1. 月份天数不同（28~31 天）；
    2. 数据集最后一个月只到 2011-12-09，12 月的总量天然偏小 —— 不按天归一，
       12 月会被低估。

    另一列 ``month_share`` 是"该类目成交额占当月全部成交额的比例"，
    衡量的是**相对地位**（旺季里谁更突出），保留下来供对比与排查。
    当月样本（成交行数）少于 ``min_events`` 的类目按 1.0 处理并写进 ``events`` 列，
    方便页面说明"这个类目样本不足，没有季节信号"。

    ``window`` 是统计窗口（一般传 :func:`common_window` 的结果）：传了就把交易表
    裁到窗口内再算，保证每个月的口径可比。
    """
    min_events = int(
        min_events if min_events is not None else HYPERPARAMS["home"]["season_min_events"]
    )

    df = clean_df.copy()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["_dt"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["_month"] = df["_dt"].dt.month
    df["_amount"] = df["Quantity"] * df["UnitPrice"]
    df = df.dropna(subset=["_month"])
    df["_month"] = df["_month"].astype(int)

    if window is not None:
        start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        df = df[(df["_dt"] >= start) & (df["_dt"] <= end)]

    # 每个月实际被数据覆盖了几天：本数据集最后一个月份只到 12-09，
    # 半个月的数据会让"当月占比"偏高，页面上要能说清这一点。
    days_covered = (
        df.groupby("_month")["_dt"].apply(lambda s: int(s.dt.normalize().nunique())).to_dict()
    )
    total_days = int(df["_dt"].dt.normalize().nunique()) or 1

    categories = catalog.build_category_map(products)
    df["category"] = df["StockCode"].map(categories).fillna(catalog.UNKNOWN_CATEGORY)

    grouped = df.groupby(["_month", "category"])
    stats = pd.DataFrame(
        {
            "events": grouped.size(),
            "qty": grouped["Quantity"].sum(),
            "amount": grouped["_amount"].sum(),
        }
    ).reset_index()
    stats = stats.rename(columns={"_month": "month"})

    month_amount = stats.groupby("month")["amount"].transform("sum").replace(0, 1.0)
    stats["days_covered"] = stats["month"].map(days_covered).fillna(0).astype(int)
    stats["days_covered"] = stats["days_covered"].replace(0, 1)
    stats["daily_amount"] = stats["amount"] / stats["days_covered"]
    stats["month_share"] = stats["amount"] / month_amount

    # 该类目的"全年日均成交额"：全年成交额 ÷ 全年覆盖天数
    cat_amount = stats.groupby("category")["amount"].sum()
    cat_daily = (cat_amount / total_days).replace(0, pd.NA)
    stats["lift"] = stats["daily_amount"] / stats["category"].map(cat_daily)

    thin = stats["events"] < min_events
    stats.loc[thin, "lift"] = DEFAULT_LIFT  # 样本太薄的类目不给季节信号
    stats["lift"] = stats["lift"].astype(float).fillna(DEFAULT_LIFT).round(4)
    stats["daily_amount"] = stats["daily_amount"].astype(float).round(2)
    stats["month_share"] = stats["month_share"].astype(float).round(6)
    stats["qty"] = stats["qty"].astype(float).round(2)
    stats["amount"] = stats["amount"].astype(float).round(2)
    stats["events"] = stats["events"].astype(int)
    return stats[SEASON_STATS_COLUMNS].sort_values(
        ["month", "lift"], ascending=[True, False]
    ).reset_index(drop=True)


class SeasonTable:
    """季节统计的查询视图：``(月份, 类目) → 季节指数``。"""

    def __init__(self, stats: pd.DataFrame) -> None:
        self.stats = stats
        self._index: dict[tuple[int, str], dict[str, Any]] = {}
        for row in stats.itertuples(index=False):
            self._index[(int(row.month), str(row.category))] = {
                "events": int(row.events),
                "qty": float(row.qty),
                "amount": float(row.amount),
                "days_covered": int(row.days_covered),
                "daily_amount": float(row.daily_amount),
                "month_share": float(row.month_share),
                "lift": float(row.lift),
            }
        self._days: dict[int, int] = {}
        if "days_covered" in stats.columns:
            for row in stats.itertuples(index=False):
                self._days[int(row.month)] = int(row.days_covered)

    def __len__(self) -> int:
        return len(self.stats)

    @property
    def months(self) -> list[int]:
        return sorted(self.stats["month"].unique().tolist())

    def info(self, month: int, category: str) -> dict[str, Any] | None:
        """某个类目在某个月的季节统计；没有记录时返回 ``None``。"""
        return self._index.get((int(month), str(category)))

    def lift(self, month: int, category: str) -> float:
        """季节指数，缺记录时按 1.0（无季节信号）处理。"""
        record = self.info(month, category)
        return float(record["lift"]) if record else DEFAULT_LIFT

    def coverage(self, month: int) -> dict[str, Any]:
        """某个多月的数据覆盖情况（本数据集按月份统计）。

        ``partial=True`` 表示这个月只有部分日期有数据（例如数据只到 12-09），
        此时当月的"占比"分母偏小，季节指数会整体偏高 —— 页面要把它标出来。
        """
        days = int(self._days.get(int(month), 0))
        return {"month": int(month), "days_covered": days, "partial": 0 < days < 20}

    def top_categories(
        self,
        month: int,
        k: int = 5,
        min_events: int | None = None,
        skip_categories: Sequence[str] = (catalog.UNKNOWN_CATEGORY, "其他", "类别"),
    ) -> list[dict[str, Any]]:
        """某个月季节指数最高的类目。

        默认过滤两类东西：样本不足的类目（指数不可信），以及"其他/未知类目/类别"
        这种兜底类目 —— 它们不是真实的商品品类，出现在榜上只会干扰阅读。
        """
        min_events = int(
            min_events if min_events is not None else HYPERPARAMS["home"]["season_min_events"]
        )
        skip = {str(c) for c in skip_categories}
        rows = [
            {"category": category, **record}
            for (m, category), record in self._index.items()
            if m == int(month) and record["events"] >= min_events and str(category) not in skip
        ]
        rows.sort(key=lambda r: (-r["lift"], -r["amount"]))
        return rows[:k]

    @staticmethod
    def describe(rows: Sequence[Mapping[str, Any]]) -> str:
        """把 Top 类目拼成一句话，页面直接用。"""
        return "、".join(f"{r['category']} ×{float(r['lift']):.1f}" for r in rows)


def load_season_table(
    path: Path | None = None,
    force: bool = False,
    window: tuple[Any, Any] | None = None,
) -> SeasonTable:
    """读取缓存的季节统计；不存在（或 ``force``）时重建并落盘。"""
    path = Path(path or PATHS["season_stats"])
    if path.exists() and not force:
        stats = pd.read_csv(path)
    else:
        clean = pd.read_csv(PATHS["clean_transactions"], dtype=str)
        products = catalog.load_products()
        if window is None:
            window = common_window([PATHS["raw_transactions"], *PATHS["extra_transactions"]])
        stats = build_season_stats(clean, products, window=window)
        path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(path, index=False, encoding="utf-8")
    stats["category"] = stats["category"].astype(str)
    stats["month"] = stats["month"].astype(int)
    return SeasonTable(stats)
