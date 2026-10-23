"""阶段一：读数据、探查、三级清洗。

对应任务书：步骤 3～5。

清洗顺序固定如下，不可调换，否则各级保留比例与全班不可比：
    1) 删除缺失用户（CustomerID 为空）
    2) 删除取消/退货（InvoiceNo 以 C 开头，或 Quantity <= 0）
    3) 删除非法价格（UnitPrice <= 0）

字段与编码约定见 README「清洗规则」小节。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# 实测结论：交易表内含 32 个 0xA3 字节（礼品券描述中的 £ 符号），不是 UTF-8。
# 用 utf-8 / gbk / gb18030 读取都会抛 UnicodeDecodeError，必须用 latin-1（等价 cp1252）。
TRANSACTIONS_ENCODING = "latin-1"
# 商品主数据是正常 UTF-8，加 -sig 以兼容可能的 BOM。
PRODUCTS_ENCODING = "utf-8-sig"

DATE_FORMAT = "%m/%d/%Y %H:%M"


def _is_integral(series: pd.Series) -> bool:
    """判断一个数值列是否全部为整数（用于保留 Quantity 的整数形态）。"""
    values = series.dropna()
    return bool(series.notna().all() and (values % 1 == 0).all())


def load_transactions(path: Path) -> pd.DataFrame:
    """从 data/raw 读入交易表。

    要点：
    - ID 列一律按字符串读取，避免 ``85123A`` 被解析成数字或缺失；
    - 交易表实测非 UTF-8，固定用 ``latin-1`` 读取；
    - ``InvoiceDate`` 解析为 datetime，``Quantity`` / ``UnitPrice`` 转为数值，
      供后续探查、清洗与按时间切分使用。
    """
    df = pd.read_csv(path, dtype=str, encoding=TRANSACTIONS_ENCODING)
    df.columns = [str(c).strip() for c in df.columns]

    for col in ("InvoiceNo", "StockCode", "CustomerID", "Country", "Description"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    raw_date = df["InvoiceDate"]
    parsed = pd.to_datetime(raw_date, format=DATE_FORMAT, errors="coerce")
    if parsed.isna().any():
        # 兜底：若数据中出现其它日期写法，退化为自动推断
        parsed = pd.to_datetime(raw_date, errors="coerce")
    df["InvoiceDate"] = parsed

    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Quantity"] = qty.astype("int64") if _is_integral(qty) else qty
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    return df


def load_products(path: Path) -> pd.DataFrame:
    """读入商品主数据，ID 与文本列去空格后统一为字符串。"""
    df = pd.read_csv(path, dtype=str, encoding=PRODUCTS_ENCODING)
    df.columns = [str(c).strip() for c in df.columns]
    for col in ("StockCode", "Description", "Chinese_Description", "Product_Category"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df


def probe_raw(df: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
    """原始数据体检：规模、缺失、异常、热门集中度、时间跨度。"""
    total = len(df)
    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    price = pd.to_numeric(df["UnitPrice"], errors="coerce")
    invoice = df["InvoiceNo"].fillna("").astype(str).str.upper()

    top_items = df["StockCode"].value_counts().head(top_n)
    top_rows = int(top_items.sum())
    missing_user = int(df["CustomerID"].isna().sum())

    # 同一商品被录入多个描述的情况，反映商品主数据/交易录入的一致性
    desc_per_item = (
        df.dropna(subset=["Description"]).groupby("StockCode")["Description"].nunique()
    )

    return {
        "rows": total,
        "users": int(df["CustomerID"].nunique(dropna=True)),
        "items": int(df["StockCode"].nunique(dropna=True)),
        "invoices": int(df["InvoiceNo"].nunique(dropna=True)),
        "countries": int(df["Country"].nunique(dropna=True)),
        "missing_user_rows": missing_user,
        "missing_user_ratio": round(missing_user / total, 6) if total else 0.0,
        "cancel_invoice_rows": int(invoice.str.startswith("C").sum()),
        "nonpositive_qty_rows": int((qty <= 0).sum()),
        "missing_qty_rows": int(qty.isna().sum()),
        "nonpositive_price_rows": int((price <= 0).sum()),
        "missing_price_rows": int(price.isna().sum()),
        "date_min": str(df["InvoiceDate"].min()),
        "date_max": str(df["InvoiceDate"].max()),
        "top_n": top_n,
        "top_n_item_rows": top_rows,
        "top_n_item_share": round(top_rows / total, 6) if total else 0.0,
        "items_with_multiple_desc": int((desc_per_item > 1).sum()),
    }


def top_countries(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """按交易行数统计 Top N 国家/地区。"""
    grouped = (
        df.groupby("Country", dropna=False)
        .agg(rows=("Country", "size"), users=("CustomerID", "nunique"))
        .sort_values("rows", ascending=False)
        .head(n)
        .reset_index()
    )
    grouped["share"] = (grouped["rows"] / len(df)).round(6) if len(df) else 0.0
    return grouped


def _mode_description(df: pd.DataFrame) -> pd.Series:
    """每个商品取出现次数最多的英文描述。

    原始数据里同一 StockCode 可能对应多个 Description（录入不一致），
    取众数比取首次/末次更稳定，也便于在报告中说明该现象。
    """
    pairs = (
        df.dropna(subset=["Description"])
        .groupby(["StockCode", "Description"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    if pairs.empty:
        return pd.Series(dtype="object")
    top = pairs.sort_values("count", ascending=False).drop_duplicates("StockCode", keep="first")
    return top.set_index("StockCode")["Description"]


def top_items(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """按交易行数统计 Top N 热门商品，并附上英文描述便于判读。"""
    grouped = (
        df.groupby("StockCode", dropna=False)
        .agg(rows=("StockCode", "size"), quantity=("Quantity", "sum"))
        .sort_values("rows", ascending=False)
        .head(n)
        .reset_index()
    )
    grouped["share"] = (grouped["rows"] / len(df)).round(6) if len(df) else 0.0
    grouped["Description"] = grouped["StockCode"].map(_mode_description(df))
    return grouped


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """三级清洗漏斗。

    返回：``(清洗后表, 每步保留统计表)``。

    规则（顺序固定，须与 README 一致）：
      1) 删除 CustomerID 为空的行 —— 没有用户 ID 就无法做用户级推荐与序列；
      2) 删除取消/退货：InvoiceNo 以 C 开头，或 Quantity <= 0 —— 退货不是偏好信号；
      3) 删除非法价格：UnitPrice <= 0 —— 明显异常，不代表真实偏好。

    无法解析为数值的 Quantity / UnitPrice 也归入对应级别一并删除。
    """
    total = len(df)
    stats: list[dict[str, Any]] = []
    current = df

    def apply_step(step: int, name: str, rule: str, drop_mask: pd.Series) -> None:
        """执行一级过滤，并记录删除前后的行数与保留比例。"""
        nonlocal current
        before = len(current)
        current = current[~drop_mask]
        removed = before - len(current)
        stats.append(
            {
                "step": step,
                "step_name": name,
                "rule": rule,
                "rows_before": before,
                "rows_removed": removed,
                "rows_after": len(current),
                "removed_ratio": round(removed / before, 6) if before else 0.0,
                "retention_ratio": round(len(current) / total, 6) if total else 0.0,
            }
        )

    # 第 1 级 —— 缺失用户（含空字符串）
    user = current["CustomerID"].fillna("").astype(str).str.strip()
    apply_step(1, "drop_missing_user", "CustomerID 为空", user == "")

    # 第 2 级 —— 取消 / 退货
    invoice = current["InvoiceNo"].fillna("").astype(str).str.upper()
    qty = pd.to_numeric(current["Quantity"], errors="coerce")
    apply_step(
        2,
        "drop_cancel_return",
        "InvoiceNo 以 C 开头 或 Quantity <= 0",
        invoice.str.startswith("C") | qty.isna() | (qty <= 0),
    )

    # 第 3 级 —— 非法价格
    price = pd.to_numeric(current["UnitPrice"], errors="coerce")
    apply_step(
        3,
        "drop_invalid_price",
        "UnitPrice <= 0",
        price.isna() | (price <= 0),
    )

    clean = current.reset_index(drop=True)
    return clean, pd.DataFrame(stats)


def spotcheck_product_names(
    clean_df: pd.DataFrame, products: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """取清洗后交互最多的 TopN 商品，与商品主数据按 StockCode 关联抽查。"""
    counts = clean_df["StockCode"].value_counts().head(top_n)
    out = pd.DataFrame(
        {
            "StockCode": counts.index.astype(str),
            "interactions": counts.to_numpy(),
        }
    )

    prod = products.copy()
    prod["StockCode"] = prod["StockCode"].astype(str).str.strip()
    prod["_found"] = True
    prod = prod.drop_duplicates(subset="StockCode", keep="first")

    keep = [
        c
        for c in ("StockCode", "Description", "Chinese_Description", "Product_Category", "_found")
        if c in prod.columns
    ]
    out = out.merge(prod[keep], on="StockCode", how="left")

    out["matched"] = out["_found"].notna()
    out = out.drop(columns=["_found"])

    chinese = out["Chinese_Description"] if "Chinese_Description" in out.columns else None
    english = out["Description"] if "Description" in out.columns else None
    display = pd.Series(pd.NA, index=out.index, dtype="string")
    if chinese is not None:
        display = display.fillna(chinese)
    if english is not None:
        display = display.fillna(english)
    out["display_name"] = display.fillna("未知商品")

    out["reason"] = out["matched"].map(
        {True: "匹配到商品主数据", False: "商品主数据中未找到该 StockCode"}
    )
    return out


def write_probe_report(probe: dict[str, Any], path: Path) -> None:
    """把原始数据体检结果写成可读文本，便于验收与写入报告。"""
    pct = lambda v: f"{v * 100:.2f}%"  # noqa: E731
    lines = [
        "原始数据体检报告（阶段一）",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "[规模]",
        f"  交易行数          : {probe['rows']}",
        f"  用户数            : {probe['users']}",
        f"  商品数            : {probe['items']}",
        f"  发票数            : {probe['invoices']}",
        f"  国家/地区数       : {probe['countries']}",
        f"  时间跨度          : {probe['date_min']} ~ {probe['date_max']}",
        "",
        "[数据质量]",
        f"  CustomerID 缺失   : {probe['missing_user_rows']} 行（{pct(probe['missing_user_ratio'])}）",
        f"  InvoiceNo 以 C 开头: {probe['cancel_invoice_rows']} 行",
        f"  Quantity <= 0     : {probe['nonpositive_qty_rows']} 行",
        f"  Quantity 无法解析 : {probe['missing_qty_rows']} 行",
        f"  UnitPrice <= 0    : {probe['nonpositive_price_rows']} 行",
        f"  UnitPrice 无法解析: {probe['missing_price_rows']} 行",
        "",
        "[热门集中度]",
        f"  热门 Top{probe['top_n']} 商品占总行数: {pct(probe['top_n_item_share'])}"
        f"（{probe['top_n_item_rows']} 行）",
        "",
        "[一致性]",
        f"  同一商品存在多个描述的商品数: {probe['items_with_multiple_desc']}"
        "（展示名称时统一取出现次数最多的描述）",
        "",
        "[结论]",
        "  热门集中度低说明数据长尾很重，热门基线之外还需覆盖长尾才能看出差距；",
        "  缺失用户与退货行必须先剔除，否则会污染共现与协同过滤。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------ 补充数据源
def load_extra_sources(
    paths: Iterable[Any],
    loader,
) -> list[tuple[str, pd.DataFrame]]:
    """读取"额外数据源"（如 ``data/generated/`` 下的补充数据）。

    这些数据与原始数据结构完全一致，所以**复用同一个 loader 与同一套清洗规则**。
    路径不存在时自动跳过，不会影响只跑原始数据的流程。

    参数：
        paths:  文件路径列表（可来自 ``config/settings.py`` 的 ``extra_transactions``）
        loader: 读取函数，通常是 :func:`load_transactions` 或 :func:`load_products`

    返回：``[(数据源名称, DataFrame), ...]``
    """
    out: list[tuple[str, pd.DataFrame]] = []
    for raw_path in paths or []:
        path = Path(raw_path)
        if not path.exists():
            continue
        out.append((path.stem, loader(path)))
    return out
