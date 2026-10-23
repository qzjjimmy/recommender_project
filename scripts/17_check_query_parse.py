#!/usr/bin/env python3
"""查询理解自检（关键词清洗 / 价格口径 / 大模型回显），结果写到
``outputs/stage21/stage21_query_parse_check.txt``。

分三段：

    A. 规则层：不依赖大模型，断言关键词清洗与价格解析的确定性行为
       —— "我要买8000块钱以内的华为手机" 的关键词必须是「华为」，
       不能出现「我要买」「的华为」这种碎片。
    B. 合并层：价格"整对以规则为准"（大模型的假下限不能漏进来），
       以及"X以内被读成 min=max"的退化区间会被校验层纠正。
    C. 端到端：真实跑一遍查询矩阵（大模型可用时），检查来源、命中数与
       回显条件；再验证大模型不可用时的静默降级不会报错。

运行（项目根目录）：
    python scripts/17_check_query_parse.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import PATHS, stage_output_dir  # noqa: E402
from src import query_parser as qp  # noqa: E402
from src import smart_search as ss  # noqa: E402

# 关键词里绝不该出现的碎片（规则切分留下的请求词/语气词）
NOISE_WORDS = {"我要买", "的华为", "我要", "我想", "的", "了", "买", "我"}

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def check(label: str, ok: bool, detail: str = "") -> bool:
    say(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" —— {detail}" if detail else ""))
    return ok


# ------------------------------------------------------------------ A 规则层
def rule_checks(searcher: ss.SmartSearch) -> bool:
    say("A. 规则层：关键词清洗与价格解析（不依赖大模型）")
    ok = True
    cats, vocab = searcher.categories, searcher._vocabulary

    c = qp.parse_query_rules("我要买8000块钱以内的华为手机。", cats, vocabulary=vocab)
    ok &= check(
        "「我要买8000块钱以内的华为手机」→ 品类=手机、价格≤8000、关键词=华为",
        c.categories == ["手机"] and c.max_price == 8000.0 and c.min_price is None
        and c.keywords == ["华为"],
        f"cats={c.categories} min={c.min_price} max={c.max_price} kw={c.keywords}",
    )
    ok &= check(
        "关键词里没有「我要买 / 的华为」这类碎片",
        not (set(c.keywords) & NOISE_WORDS),
        f"kw={c.keywords}",
    )

    c2 = qp.parse_query_rules("红色的购物袋", cats, vocabulary=vocab)
    ok &= check("「红色的购物袋」→ 关键词「红色」（尾缀的“的”被剥掉）",
                c2.keywords == ["红色"] and c2.categories == ["购物袋"],
                f"cats={c2.categories} kw={c2.keywords}")

    c3 = qp.parse_query_rules("不超过3元的厨房用品", cats, vocabulary=vocab)
    ok &= check("「不超过3元的厨房用品」→ 价格上限 3、关键词为空",
                c3.max_price == 3.0 and c3.min_price is None and c3.keywords == [],
                f"min={c3.min_price} max={c3.max_price} kw={c3.keywords}")

    c4 = qp.parse_query_rules("HUAWEI Mate 60 手机", cats, vocabulary=vocab)
    ok &= check("英文品牌归一：HUAWEI → 华为",
                "华为" in c4.keywords and "HUAWEI" not in c4.keywords,
                f"kw={c4.keywords}")

    c5 = qp.parse_query_rules("家居小礼物送女朋友", cats, vocabulary=vocab)
    ok &= check("全库不存在的词不进关键词（留着只会被静默忽略）",
                all(searcher.keyword_hits(w) > 0 for w in c5.keywords),
                f"kw={c5.keywords}")
    return ok


# ------------------------------------------------------------------ B 合并层
def merge_checks() -> bool:
    say("\nB. 合并层：价格整对以规则为准 / 退化区间纠正")
    ok = True

    rules = qp.QueryConstraints(raw="q", categories=["手机"], min_price=None, max_price=8000.0)
    # 大模型把"8000以内"读成"=8000"，这就是线上那次候选清零的元凶
    llm = qp.QueryConstraints(raw="q", categories=["手机"], min_price=8000.0,
                              max_price=8000.0, keywords=["华为"], source="llm")
    merged = qp.merge_constraints(rules, llm)
    ok &= check("规则有价格时，大模型的假下限不会漏进合并结果",
                merged.min_price is None and merged.max_price == 8000.0,
                f"min={merged.min_price} max={merged.max_price}")

    empty = qp.merge_constraints(
        qp.QueryConstraints(raw="q"), 
        qp.QueryConstraints(raw="q", min_price=None, max_price=300.0, source="llm"),
    )
    ok &= check("规则读不出价格时，仍然采纳大模型的价格",
                empty.max_price == 300.0, f"max={empty.max_price}")

    from src.llm_parser import LLMQueryParser

    parser = LLMQueryParser(enabled=True, backend="openai", base_url="x", model="y",
                            api_key="k")
    got = parser._validate(
        {"categories": ["手机"], "min_price": 8000, "max_price": 8000, "keywords": ["华为"]},
        {"手机"},
        "我要买8000块钱以内的华为手机。",
    )
    ok &= check("「X以内」被读成 min=max 时，校验层只保留上限",
                got is not None and got.min_price is None and got.max_price == 8000.0,
                None if got is None else f"min={got.min_price} max={got.max_price}")

    dedup = parser._validate(
        {"categories": ["家居装饰"], "min_price": None, "max_price": 100,
         "keywords": ["家居", "装饰", "HUAWEI", "60"]},
        {"家居装饰"},
        "100块钱以内的家居装饰",
    )
    ok &= check("被类目名包含的关键词、纯数字、英文品牌都被清理",
                dedup is not None and dedup.keywords == ["华为"],
                None if dedup is None else f"kw={dedup.keywords}")
    return ok


# --------------------------------------------------------------- C 端到端
def end_to_end_checks(searcher: ss.SmartSearch) -> tuple[bool, bool]:
    say("\nC. 端到端：真实查询矩阵")
    cases = [
        ("我要买8000块钱以内的华为手机。", 15, {"华为"}, None, 8000.0),
        ("不超过3元的厨房用品", 308, None, None, 3.0),
        ("5000以上的苹果手机", 24, {"苹果"}, 5000.0, None),
        ("100块钱以内的家居装饰", 874, None, None, 100.0),
        ("200块左右的蓝牙耳机", 4, None, None, 200.0),
        ("HUAWEI Mate 60 手机", 26, {"华为", "Mate"}, None, None),
    ]

    llm_ready = bool(searcher.llm is not None and searcher.llm.available)
    if not llm_ready:
        say("  [SKIP] 大模型当前不可用（bridge/Ollama 没起来），跳过端到端矩阵")
    ok = True
    for query, want_hits, want_kw, want_min, want_max in cases:
        result = searcher.search(query, force_llm=llm_ready)
        cons, parser = result["constraints"], result["parser"]
        detail = (f"source={parser['source']} 命中={result['filtered_items']} "
                  f"条件={cons['text']}")
        problems = []
        if result["filtered_items"] <= 0:
            problems.append("候选为 0")
        if cons["min_price"] is not None and cons["min_price"] == cons["max_price"]:
            problems.append("出现 min=max 的退化价格")
        if want_kw and not want_kw.issubset(set(cons["keywords"] or [])):
            problems.append(f"缺关键词 {sorted(want_kw)}")
        if want_min is not None and cons["min_price"] != want_min:
            problems.append(f"价格下限应为 {want_min}")
        if want_max is not None and cons["max_price"] != want_max:
            problems.append(f"价格上限应为 {want_max}")
        if llm_ready and result["filtered_items"] != want_hits:
            # 命中数依赖大模型的措辞，只做提示不做判失败
            detail += f"（期望约 {want_hits}）"
        if not problems and llm_ready:
            say(f"  [PASS] {query} —— {detail}")
        elif not problems:
            say(f"  [PASS] {query}（规则通道）—— {detail}")
        else:
            ok &= check(query, False, detail + "；" + "、".join(problems))
    return llm_ready, ok


def degrade_checks(searcher: ss.SmartSearch) -> bool:
    say("\nD. 降级回归：大模型不可用时页面依然可用")
    ok = True
    backup = searcher.llm
    try:
        searcher.llm = None
        for query in ("我要买8000块钱以内的华为手机。", "便宜又好看的家居小礼物"):
            result = searcher.search(query, force_llm=True)
            ok &= check(f"（无大模型）{query} 正常返回",
                        result["parser"]["source"] == "rules" and bool(result["results"]),
                        f"source={result['parser']['source']} 命中={result['filtered_items']}")

        class DeadLLM:
            """模拟：桥接在跑但每次调用都失败。"""

            calls, failures = 2, 2
            model = "dead"

            @property
            def available(self) -> bool:
                return False

            def describe(self) -> str:
                return "大模型当前不可用"

            def parse(self, *a, **k):
                return None

        searcher.llm = DeadLLM()
        result = searcher.search("我要买8000块钱以内的华为手机。", force_llm=True)
        notes = " ".join(result["notes"])
        ok &= check("大模型不可用时给出一句可读的降级说明（不是 0 件也不是报错）",
                    "大模型" in notes and result["filtered_items"] == 15,
                    f"命中={result['filtered_items']}")
    finally:
        searcher.llm = backup
    return ok


def main() -> None:
    say(f"查询理解自检 · {date.today().isoformat()}")
    say(f"运行环境：{sys.executable}")
    searcher = ss.SmartSearch.load()
    say(f"数据规模：{len(searcher.profile)} 件商品 / {len(searcher.categories)} 个类目")
    say(f"大模型：{searcher.llm.describe() if searcher.llm is not None else '未接入'}\n")

    ok_a = rule_checks(searcher)
    ok_b = merge_checks()
    _llm_ready, ok_c = end_to_end_checks(searcher)
    ok_d = degrade_checks(searcher)

    say(f"\n结论：规则层 {'通过' if ok_a else '未通过'}；"
        f"合并层 {'通过' if ok_b else '未通过'}；"
        f"端到端 {'通过' if ok_c else '未通过'}；"
        f"降级 {'通过' if ok_d else '未通过'}")
    out = stage_output_dir(21) / "stage21_query_parse_check.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n自检报告：{out}")
    if not all((ok_a, ok_b, ok_c, ok_d)):
        sys.exit(1)


if __name__ == "__main__":
    main()
