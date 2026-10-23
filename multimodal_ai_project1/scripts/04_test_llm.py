"""阶段03 — 文本 LLM 测试脚本。

测试 Qwen3-4B-Instruct-2507（GGUF int4 / llama.cpp）的对话能力与流式生成效果。
"""

from __future__ import annotations

import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PATHS, resolve_text_model  # noqa: E402


def main():
    """测试文本 LLM。"""
    print("=" * 50)
    print("文本 LLM 测试")
    print("=" * 50)

    model_path, backend = resolve_text_model()
    print(f"模型路径: {model_path}")
    print(f"推理后端: {backend}")
    print("-" * 50)

    from src.text_llm import create_text_llm

    try:
        llm = create_text_llm()
    except Exception as exc:
        print(f"✗ 没有可用的文本大模型：{exc}")
        return
    print(f"实际后端: {type(llm).__name__}")
    if backend == "none":
        print("（本地无 GGUF / transformers 模型，使用 Ollama 后端）")

    t0 = time.time()
    try:
        llm.load()
    except Exception as exc:
        print(f"✗ 模型加载失败：{exc}")
        return
    load_s = time.time() - t0
    print(f"✓ 加载完成，用时 {load_s:.2f}s")

    rows: list[dict] = []

    # ---- 1) 基础对话 ----
    query = "请用一句话介绍一下你自己。"
    t0 = time.time()
    answer = llm.generate(query)
    cost = time.time() - t0
    print(f"\n[基础对话] {query}\n  → {answer}\n  耗时 {cost:.2f}s")
    rows.append({"case": "基础对话", "prompt": query, "seconds": round(cost, 2),
                 "chars": len(answer), "chars_per_s": round(len(answer) / max(cost, 1e-6), 1)})

    # ---- 2) 流式生成：首 token 延迟 + 总耗时 ----
    query2 = "什么是多模态AI？请分点说明。"
    print(f"\n[流式生成] {query2}")
    t0 = time.time()
    first_at = None
    pieces: list[str] = []
    for piece in llm.stream_generate(query2):
        if first_at is None:
            first_at = time.time() - t0
        pieces.append(piece)
        print(piece, end="", flush=True)
    total = time.time() - t0
    text2 = "".join(pieces)
    print(f"\n  首 token {first_at or 0:.2f}s / 总耗时 {total:.2f}s / {len(text2)} 字")
    rows.append({"case": "流式生成", "prompt": query2, "seconds": round(total, 2),
                 "chars": len(text2), "chars_per_s": round(len(text2) / max(total, 1e-6), 1)})

    # ---- 3) 上下文模式：验证模型确实基于"参考资料"作答 ----
    context = "本项目共6个阶段，每阶段8课时，项目名称是多模态AI助手。"
    query3 = "本项目共几个阶段？每阶段多少课时？"
    t0 = time.time()
    answer3 = llm.generate(query3, context=context)
    cost3 = time.time() - t0
    print(f"\n[上下文模式] 参考={context}\n  问：{query3}\n  → {answer3}\n  耗时 {cost3:.2f}s")
    hit = ("6" in answer3 or "六" in answer3)
    print(f"  {'✓' if hit else '⚠'} 回答中{'包含' if hit else '未包含'}参考资料里的阶段数")
    rows.append({"case": "上下文模式", "prompt": query3, "seconds": round(cost3, 2),
                 "chars": len(answer3), "chars_per_s": round(len(answer3) / max(cost3, 1e-6), 1)})

    out_csv = Path(PATHS["outputs"]) / "llm_benchmark.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["case", "prompt", "seconds", "chars", "chars_per_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ 基准数据已写入 {out_csv}")

    print("\n测试完成。")


if __name__ == "__main__":
    main()
