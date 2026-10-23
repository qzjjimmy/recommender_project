"""阶段03 — 知识库 RAG 测试脚本。

测试文档导入、向量检索和 RAG 增强问答。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PATHS  # noqa: E402

# 默认知识库文档：优先用仓库里已有的课程材料（无需你额外准备）
_REPO = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = [
    _REPO / "阶段01_环境搭建与音视频采集_8课时.md",
    _REPO / "阶段05_人体姿态与手势识别_8课时.md",
    _REPO / "README.md",
]


def main():
    """测试知识库 RAG 功能。"""
    print("=" * 50)
    print("知识库 RAG 测试")
    print("=" * 50)

    from src.knowledge_base import VectorStore
    from src.rag import RAGPipeline
    from src.text_llm import create_text_llm

    try:
        text_llm = create_text_llm()
        text_llm.load()
    except Exception as exc:
        print(f"✗ 文本大模型不可用：{exc}")
        return

    store = VectorStore()
    rag = RAGPipeline(store, text_llm)

    docs = [d for d in DEFAULT_DOCS if d.exists()]
    if not docs:
        print("✗ 没有找到可用的测试文档，请把 PDF/TXT 放进 data/ 后重试。")
        return

    print("\n--- 导入文档 ---")
    for doc in docs:
        added = rag.import_document(str(doc))
        print(f"  ✓ {doc.name}: 新增 {added} 块（当前共 {len(store.documents)} 块）")

    query = "阶段05 里手势识别支持哪几种手势？"
    print(f"\n--- 检索测试 ---\n问题：{query}")
    hits = store.search(query)
    for i, hit in enumerate(hits, 1):
        print(f"  #{i} 距离={hit.get('score', 0):.3f} "
              f"来源={hit.get('source')} 内容={hit.get('text', '')[:50]}…")

    print("\n--- 有知识库 ---")
    answer_kb = rag.answer(query)
    print(f"  → {answer_kb}")

    print("\n--- 无知识库（对照）---")
    answer_plain = text_llm.generate(query)
    print(f"  → {answer_plain}")

    out = Path(PATHS["outputs"]) / "kb_test_log.txt"
    out.write_text(
        "\n".join([
            "知识库 RAG 测试记录",
            f"导入文档：{', '.join(d.name for d in docs)}",
            f"知识库总块数：{len(store.documents)}",
            "",
            f"问题：{query}",
            f"检索命中：{len(hits)} 块",
            f"有知识库回答：{answer_kb}",
            f"无知识库回答：{answer_plain}",
        ]), encoding="utf-8")
    print(f"\n✓ 测试记录已写入 {out}")

    print("\n测试完成。")


if __name__ == "__main__":
    main()
