"""阶段03 — RAG (检索增强生成) 管线。

串联文档导入、向量检索和 LLM 增强回答的完整流程。
"""

from __future__ import annotations

from pathlib import Path

from src.knowledge_base import DocumentParser, VectorStore
from src.text_llm import TextLLM


class RAGPipeline:
    """检索增强生成管线。

    提供 import_document 和 answer 两个核心接口。
    """

    def __init__(self, vector_store: VectorStore, text_llm: TextLLM):
        """初始化 RAG 管线。

        Args:
            vector_store: 向量存储实例。
            text_llm: 文本 LLM 实例（TextLLM 或 TextLLMTransformers，接口一致）。
        """
        self.vector_store = vector_store
        self.text_llm = text_llm
        self.parser = DocumentParser()

    def import_document(self, file_path: str) -> int:
        """导入一个文档到知识库。

        自动识别 PDF/TXT 格式，解析、分块、嵌入并入库。

        Args:
            file_path: 文档文件路径。

        Returns:
            导入的文本块数量。
        """
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            chunks = self.parser.parse_pdf(file_path)
        elif suffix in {".txt", ".md", ".csv"}:
            chunks = self.parser.parse_txt(file_path)
        else:
            raise ValueError(f"不支持的文档格式：{suffix}（仅支持 PDF / TXT / MD）")
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def answer(self, query: str, k: int = 3) -> str:
        """基于知识库增强的问答。

        检索相关文档 → 构建上下文 → LLM 生成回答。

        Args:
            query: 用户问题。
            k: 检索文档数。

        Returns:
            LLM 生成的增强回答。
        """
        # 知识库为空就直接普通问答，避免让模型面对一个空的【参考资料】
        if not self.vector_store.documents:
            return self.text_llm.generate(query)
        results = self.vector_store.search(query, k=k)
        if not results:
            return self.text_llm.generate(query)
        context = self.build_context(results)
        # 统一走 (query, context) 形式：各后端内部自己拼 system prompt，
        # RAGPipeline 因此无需关心底层是 llama.cpp / transformers / Ollama。
        return self.text_llm.generate(query, context=context)

    def build_context(self, search_results: list[dict]) -> str:
        """将检索结果拼接为上下文字符串。

        Args:
            search_results: 检索结果列表。

        Returns:
            格式化的上下文字符串。
        """
        if not search_results:
            return ""
        blocks: list[str] = []
        for i, item in enumerate(search_results, 1):
            source = item.get("source", "未知来源")
            page = item.get("page", -1)
            where = f"（第{page}页）" if isinstance(page, int) and page > 0 else ""
            blocks.append(
                f"【参考资料{i}】来源: {source}{where}\n内容: {item.get('text', '')}")
        return "\n\n".join(blocks)
