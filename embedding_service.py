"""
嵌入向量服务
封装 sentence-transformers 模型，提供文本向量化能力
"""

import numpy as np
from sentence_transformers import SentenceTransformer

import config


class EmbeddingService:
    """
    文本嵌入服务：将文本编码为稠密向量
    使用 sentence-transformers 模型进行编码
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        """编码单条文本为向量"""
        return self.model.encode(text, normalize_embeddings=True)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """批量编码文本为向量矩阵，每行为一条文本的向量"""
        return self.model.encode(texts, normalize_embeddings=True)

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
