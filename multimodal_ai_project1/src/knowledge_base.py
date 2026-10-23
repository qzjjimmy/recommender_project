"""阶段03 — 知识库系统模块。

实现文档解析 (PDF/TXT)、文本分块、向量嵌入和 FAISS 向量检索。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config.settings import KB_CONFIG
from config.settings import MODEL_PATHS


class DocumentParser:
    """文档解析器。

    支持 PDF (通过 PyMuPDF) 和 TXT 格式。
    """

    CHUNK_SIZE = KB_CONFIG["chunk_size"]
    CHUNK_OVERLAP = KB_CONFIG["chunk_overlap"]

    def parse_pdf(self, file_path: str) -> list[dict]:
        """解析 PDF 文件并返回文本块列表。

        Args:
            file_path: PDF 文件路径。

        Returns:
            列表，每项 dict: {"text": str, "source": str, "page": int, "chunk_id": int}
        """
        try:
            import pymupdf as fitz   # PyMuPDF 1.24+ 的新包名
        except Exception:
            import fitz              # 兼容旧版本

        docs: list[dict] = []
        with fitz.open(file_path) as pdf:
            for page_no, page in enumerate(pdf):
                docs.extend(self.chunk_text(page.get_text(), {
                    "source": Path(file_path).name,
                    "page": page_no + 1,
                }))
        for i, d in enumerate(docs):
            d["chunk_id"] = i
        return docs

    def parse_txt(self, file_path: str) -> list[dict]:
        """解析 TXT 文件并返回文本块列表。

        Args:
            file_path: TXT 文件路径。

        Returns:
            列表，每项 dict: {"text": str, "source": str, "page": int, "chunk_id": int}
        """
        raw = Path(file_path).read_bytes()
        text = ""
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        docs = self.chunk_text(text, {
            "source": Path(file_path).name, "page": -1})
        for i, d in enumerate(docs):
            d["chunk_id"] = i
        return docs

    def chunk_text(self, text: str, metadata: dict) -> list[dict]:
        """将长文本切分为有重叠的文本块。

        Args:
            text: 原始文本。
            metadata: 附加元数据 (source, page 等)。

        Returns:
            文本块列表。
        """
        text = (text or "").strip()
        if not text:
            return []
        size = self.CHUNK_SIZE
        step = max(1, size - self.CHUNK_OVERLAP)
        chunks: list[dict] = []
        for i, start in enumerate(range(0, len(text), step)):
            piece = text[start:start + size].strip()
            if not piece:
                continue
            chunk = {"text": piece}
            chunk.update(metadata)
            chunk["chunk_id"] = metadata.get("chunk_id", i)
            chunks.append(chunk)
            if start + size >= len(text):
                break
        return chunks


class VectorStore:
    """向量存储与检索。

    使用 Sentence-Transformer 生成嵌入向量，FAISS 索引进行相似度检索。
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """初始化向量存储。

        Args:
            embedding_model_name: Sentence-Transformer 模型名。
        """
        # 懒加载：SentenceTransformer 构造要读权重，放在这里会拖慢 import
        self.embedding_model = None
        self.embedding_model_name = embedding_model_name
        self.dimension = 384  # all-MiniLM-L6-v2 输出维度

        self.index: faiss.IndexFlatL2 | None = None
        self.documents: list[dict] = []
        self._lock = threading.Lock()

    def build_index(self):
        """创建 FAISS 索引。"""
        self.index = faiss.IndexFlatL2(self.dimension)

    def _ensure_embedding_model(self):
        """按需加载嵌入模型。

        优先用 config 里配置的本地目录（MODEL_PATHS["embedding"]）；
        如果传入的是自定义模型名，则直接用该名字让 sentence-transformers 处理。
        """
        if self.embedding_model is not None:
            return self.embedding_model
        target = self.embedding_model_name
        configured = str(MODEL_PATHS["embedding"])
        if self.embedding_model_name == "all-MiniLM-L6-v2" and Path(configured).exists():
            target = configured
        self.embedding_model = SentenceTransformer(target)
        return self.embedding_model

    def _ensure_index(self):
        if self.index is None:
            self.build_index()
        return self.index

    def add_documents(self, documents: list[dict]):
        """将文档块编码为向量并加入索引。

        Args:
            documents: 文档块列表，每项含 "text" 和元数据。
        """
        if not documents:
            return
        model = self._ensure_embedding_model()
        with self._lock:
            index = self._ensure_index()
            texts = [d.get("text", "") for d in documents]
            vectors = model.encode(texts, convert_to_numpy=True,
                                   show_progress_bar=False)
            vectors = np.asarray(vectors, dtype="float32")
            index.add(vectors)
            self.documents.extend(documents)

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """检索与查询最相关的 top-k 文档块。

        Args:
            query: 查询文本。
            k: 返回结果数。

        Returns:
            列表，每项 dict: {"text": str, "score": float, "source": str, ...}
        """
        k = k or KB_CONFIG["search_top_k"]
        with self._lock:
            if self.index is None or not self.documents:
                return []
            k = min(k, len(self.documents))
            model = self._ensure_embedding_model()
            qv = np.asarray(model.encode([query], convert_to_numpy=True,
                                         show_progress_bar=False), dtype="float32")
            distances, indices = self.index.search(qv, k)
            results: list[dict] = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                item = dict(self.documents[idx])
                item["score"] = float(dist)   # L2 距离，越小越相关
                results.append(item)
        return results

    def clear(self):
        """清空索引和文档。"""
        with self._lock:
            self.index = None
            self.documents.clear()
            self.build_index()

    def save(self, filepath: str):
        """将索引和文档保存到磁盘。"""
        with self._lock:
            if self.index is None:
                return
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(path))
            path.with_suffix(path.suffix + ".docs.json").write_text(
                json.dumps(self.documents, ensure_ascii=False), encoding="utf-8")

    def load(self, filepath: str):
        """从磁盘加载索引和文档。"""
        path = Path(filepath)
        docs_file = path.with_suffix(path.suffix + ".docs.json")
        if not (path.exists() and docs_file.exists()):
            return False
        with self._lock:
            self.index = faiss.read_index(str(path))
            self.documents = json.loads(docs_file.read_text(encoding="utf-8"))
        return True
