"""
Qdrant 向量数据库服务

存储和检索记忆/主题的嵌入向量：
  - memories 集合：记忆向量 + payload（内容、主题标签、关键词等）
  - topics 集合：主题的标签嵌入和摘要嵌入
支持按主题 id 进行标量过滤的向量检索
"""

import logging
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

import config

logger = logging.getLogger(__name__)


class QdrantService:
    """Qdrant 向量数据库操作封装"""

    def __init__(
        self,
        host: str = config.QDRANT_HOST,
        port: int = config.QDRANT_PORT,
    ):
        self.client = QdrantClient(host=host, port=port, check_compatibility=False)
        self._ensure_collections()
        logger.info(f"Qdrant 连接成功: {host}:{port}")

    def _ensure_collections(self):
        """确保所需的集合存在"""
        existing = {c.name for c in self.client.get_collections().collections}

        # 记忆向量集合
        if config.QDRANT_COLLECTION_MEMORIES not in existing:
            self.client.create_collection(
                collection_name=config.QDRANT_COLLECTION_MEMORIES,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            # 创建 payload 索引，加速按主题过滤
            self.client.create_payload_index(
                collection_name=config.QDRANT_COLLECTION_MEMORIES,
                field_name="topic_ids",
                field_schema="keyword",
            )
            logger.info(f"创建 Qdrant 集合: {config.QDRANT_COLLECTION_MEMORIES}")

        # 主题向量集合（存储标签嵌入和摘要嵌入）
        if config.QDRANT_COLLECTION_TOPICS not in existing:
            self.client.create_collection(
                collection_name=config.QDRANT_COLLECTION_TOPICS,
                vectors_config={
                    "label": VectorParams(
                        size=config.EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                    "summary": VectorParams(
                        size=config.EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                },
            )
            logger.info(f"创建 Qdrant 集合: {config.QDRANT_COLLECTION_TOPICS}")

    def clear_all(self):
        """清空所有集合数据"""
        self.client.delete_collection(config.QDRANT_COLLECTION_MEMORIES)
        self.client.delete_collection(config.QDRANT_COLLECTION_TOPICS)
        self._ensure_collections()
        logger.info("Qdrant 数据已清空")

    # ======================== 记忆 CRUD ========================

    def upsert_memory(
        self,
        memory_id: str,
        embedding: np.ndarray,
        payload: dict,
    ):
        """插入或更新一条记忆向量"""
        self.client.upsert(
            collection_name=config.QDRANT_COLLECTION_MEMORIES,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding.tolist(),
                    payload={**payload, "memory_id": memory_id},
                )
            ],
        )

    def upsert_memories_batch(
        self,
        memories: list[dict],
    ):
        """
        批量插入记忆向量

        Args:
            memories: [{"memory_id": str, "embedding": np.ndarray, "payload": dict}, ...]
        """
        if not memories:
            return
        points = [
            PointStruct(
                id=m["memory_id"],
                vector=m["embedding"].tolist(),
                payload={**m["payload"], "memory_id": m["memory_id"]},
            )
            for m in memories
        ]
        # 分批上传，每批 100 条
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=config.QDRANT_COLLECTION_MEMORIES,
                points=points[i : i + batch_size],
            )
        logger.info(f"批量上传 {len(points)} 条记忆向量到 Qdrant")

    def search_memories(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        topic_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        向量检索记忆

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            topic_ids: 可选的主题 id 列表过滤（利用标量过滤缩小搜索范围）

        Returns:
            [{"memory_id": str, "score": float, "payload": dict}, ...]
        """
        query_filter = None
        if topic_ids:
            # 使用 Qdrant 的标量过滤：topic_ids 字段包含任一指定主题
            query_filter = Filter(
                should=[
                    FieldCondition(
                        key="topic_ids",
                        match=MatchValue(value=tid),
                    )
                    for tid in topic_ids
                ]
            )

        results = self.client.query_points(
            collection_name=config.QDRANT_COLLECTION_MEMORIES,
            query=query_embedding.tolist(),
            query_filter=query_filter,
            limit=top_k,
        ).points

        return [
            {
                "memory_id": hit.payload.get("memory_id", ""),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results
        ]

    def search_memories_by_topics(
        self,
        query_embedding: np.ndarray,
        topic_ids: list[str],
        top_k: int = 10,
    ) -> list[dict]:
        """
        按主题过滤的向量检索（主题路由后的主题内检索）

        利用 Qdrant 的标量过滤，仅在指定主题下的记忆中执行向量搜索
        """
        return self.search_memories(
            query_embedding=query_embedding,
            top_k=top_k,
            topic_ids=topic_ids,
        )

    # ======================== 主题向量 CRUD ========================

    def upsert_topic_vectors(
        self,
        topic_id: str,
        label_embedding: Optional[np.ndarray],
        summary_embedding: Optional[np.ndarray],
        payload: dict,
    ):
        """插入或更新主题的标签嵌入和摘要嵌入"""
        vectors = {}
        if label_embedding is not None:
            vectors["label"] = label_embedding.tolist()
        if summary_embedding is not None:
            vectors["summary"] = summary_embedding.tolist()
        else:
            # 摘要不可用时用标签向量填充，避免 named vector 缺失
            if label_embedding is not None:
                vectors["summary"] = label_embedding.tolist()

        if not vectors:
            return

        self.client.upsert(
            collection_name=config.QDRANT_COLLECTION_TOPICS,
            points=[
                PointStruct(
                    id=topic_id,
                    vector=vectors,
                    payload={**payload, "topic_id": topic_id},
                )
            ],
        )

    def search_topics_by_label(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> list[dict]:
        """用 query 向量在主题标签嵌入空间中检索最相似的主题"""
        results = self.client.query_points(
            collection_name=config.QDRANT_COLLECTION_TOPICS,
            query=query_embedding.tolist(),
            using="label",
            limit=top_k,
        ).points
        return [
            {
                "topic_id": hit.payload.get("topic_id", ""),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results
        ]

    def search_topics_by_summary(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> list[dict]:
        """用 query 向量在主题摘要嵌入空间中检索最相似的主题"""
        results = self.client.query_points(
            collection_name=config.QDRANT_COLLECTION_TOPICS,
            query=query_embedding.tolist(),
            using="summary",
            limit=top_k,
        ).points
        return [
            {
                "topic_id": hit.payload.get("topic_id", ""),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results
        ]

    # ======================== 工具方法 ========================

    def get_collection_count(self, collection_name: str) -> int:
        """获取集合中的向量数量"""
        info = self.client.get_collection(collection_name)
        return info.points_count
