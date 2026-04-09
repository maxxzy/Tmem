"""
嵌入向量服务
支持两种后端:
  1. sentence-transformers (本地加载 HuggingFace 模型)
  2. ollama (通过 Ollama REST API 调用)
"""

import logging

import numpy as np
import requests

import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    文本嵌入服务：将文本编码为稠密向量
    根据 config.EMBEDDING_BACKEND 自动选择后端
    """

    def __init__(self):
        self.backend = config.EMBEDDING_BACKEND

        if self.backend == "ollama":
            self.ollama_url = config.OLLAMA_EMBEDDING_URL
            self.ollama_model = config.OLLAMA_EMBEDDING_MODEL
            logger.info(f"Embedding backend: Ollama ({self.ollama_model} @ {self.ollama_url})")
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.EMBEDDING_MODEL)
            logger.info(f"Embedding backend: sentence-transformers ({config.EMBEDDING_MODEL})")

    def encode(self, text: str) -> np.ndarray:
        """编码单条文本为向量（L2 归一化）"""
        if self.backend == "ollama":
            return self._ollama_encode([text])[0]
        return self.model.encode(text, normalize_embeddings=True)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """批量编码文本为向量矩阵，每行为一条文本的向量（L2 归一化）"""
        if self.backend == "ollama":
            return self._ollama_encode(texts)
        return self.model.encode(texts, normalize_embeddings=True)

    def _ollama_encode(self, texts: list[str]) -> np.ndarray:
        """通过 Ollama /api/embed 接口获取嵌入向量"""
        resp = requests.post(
            self.ollama_url,
            json={"model": self.ollama_model, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        embeddings = np.array(resp.json()["embeddings"], dtype=np.float32)
        # L2 归一化，与 sentence-transformers normalize_embeddings=True 行为一致
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        由于 encode 已经做了 L2 归一化，直接点积即为余弦相似度
        """
        return float(np.dot(a, b))

    @staticmethod
    def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        计算两组向量之间的余弦相似度矩阵
        A: (m, d), B: (n, d) -> 返回 (m, n)
        """
        return A @ B.T
