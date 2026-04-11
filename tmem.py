"""
TMem - 主题感知的 Agent 记忆系统

门面类：整合主题抽取、DAG 构建、关联图构建和检索四大模块，
提供统一的记忆写入和检索接口。
集成 Neo4j（图数据库）和 Qdrant（向量数据库）。

系统架构：
  对话 → [主题感知记忆抽取] → 带主题标签的记忆 → Qdrant (向量存储)
                                     ↓
                             [主题层次索引构建]
                               ├── 主题 DAG → Neo4j (图存储)
                               └── 主题关联图 → Neo4j (图存储)
                                     ↓
                 查询 → [主题路由记忆检索] → Qdrant (向量检索 + 标量过滤)
                                          → Neo4j (DAG 遍历 + PPR)
"""

import logging
from datetime import datetime
from typing import Optional

import config
from models import DialogueTurn, Memory, Topic, RetrievalResult
from embedding_service import EmbeddingService
from llm_service import LLMService
from topic_extractor import TopicExtractor
from topic_dag import TopicDAG
from topic_graph import TopicAssociationGraph
from topic_retrieval import TopicRetriever
from neo4j_service import Neo4jService
from qdrant_service import QdrantService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class TMem:
    """
    TMem 主题感知记忆系统

    使用流程：
    1. 初始化：TMem(llm_api_key="...", neo4j_password="...", qdrant_host="...")
    2. 写入记忆：tmem.add_dialogue(turns) — 处理一段对话并抽取记忆
    3. 构建索引：tmem.build_index() — 构建/重构主题 DAG 和关联图
    4. 检索记忆：tmem.retrieve(query) — 根据查询检索相关记忆
    """

    def __init__(
        self,
        llm_api_key: str = "",
        llm_model: str = config.LLM_MODEL,
        llm_base_url: str = config.LLM_BASE_URL,
        embedding_model: str = config.EMBEDDING_MODEL,
        # Neo4j 配置
        neo4j_uri: str = config.NEO4J_URI,
        neo4j_user: str = config.NEO4J_USER,
        neo4j_password: str = config.NEO4J_PASSWORD,
        # Qdrant 配置
        qdrant_host: str = config.QDRANT_HOST,
        qdrant_port: int = config.QDRANT_PORT,
        # 是否启用外部数据库（设为 False 则仅使用内存）
        use_neo4j: bool = True,
        use_qdrant: bool = True,
    ):
        # 基础服务
        self.emb_service = EmbeddingService()
        self.llm_service = LLMService(
            model=llm_model, base_url=llm_base_url, api_key=llm_api_key,
        )

        # 外部数据库服务
        self.neo4j: Optional[Neo4jService] = None
        self.qdrant: Optional[QdrantService] = None

        if use_neo4j:
            try:
                self.neo4j = Neo4jService(
                    uri=neo4j_uri, user=neo4j_user, password=neo4j_password,
                )
            except Exception as e:
                logger.warning(f"Neo4j 连接失败，回退到纯内存模式: {e}")

        if use_qdrant:
            try:
                self.qdrant = QdrantService(host=qdrant_host, port=qdrant_port)
            except Exception as e:
                logger.warning(f"Qdrant 连接失败，回退到纯内存模式: {e}")

        # 全局存储（内存副本，始终保持最新）
        self.topics: dict[str, Topic] = {}
        self.memories: dict[str, Memory] = {}

        # 子模块（延迟初始化）
        self._extractor: Optional[TopicExtractor] = None
        self._dag: Optional[TopicDAG] = None
        self._graph: Optional[TopicAssociationGraph] = None
        self._retriever: Optional[TopicRetriever] = None

        # 统计
        self._memory_count_since_rebuild = 0

        logger.info(
            f"TMem 初始化完成 "
            f"(Neo4j={'ON' if self.neo4j else 'OFF'}, "
            f"Qdrant={'ON' if self.qdrant else 'OFF'})"
        )

    def close(self):
        """关闭数据库连接"""
        if self.neo4j:
            self.neo4j.close()

    # ======================== 内部模块的懒加载 ========================

    @property
    def extractor(self) -> TopicExtractor:
        if self._extractor is None:
            self._extractor = TopicExtractor(
                self.emb_service, self.llm_service, self.topics,
            )
        return self._extractor

    @property
    def dag(self) -> TopicDAG:
        if self._dag is None:
            self._dag = TopicDAG(
                self.topics, self.memories, self.emb_service, self.llm_service,
            )
        return self._dag

    @property
    def graph(self) -> TopicAssociationGraph:
        if self._graph is None:
            self._graph = TopicAssociationGraph(
                self.topics, self.memories, self.emb_service, self.llm_service,
            )
        return self._graph

    @property
    def retriever(self) -> TopicRetriever:
        if self._retriever is None:
            self._retriever = TopicRetriever(
                self.topics, self.memories, self.emb_service,
                self.dag, self.graph,
                qdrant_service=self.qdrant,
                neo4j_service=self.neo4j,
            )
        return self._retriever

    # ======================== 核心 API ========================

    def add_dialogue(self, turns: list[DialogueTurn]) -> list[Memory]:
        """
        处理一段对话，抽取主题感知的结构化记忆

        1. 主题边界检测 → 分割为主题段
        2. 为每个主题段生成/归并主题标签
        3. 从主题段中抽取结构化记忆
        4. 同步到 Qdrant（向量）和 Neo4j（图关系）
        """
        segments, new_memories = self.extractor.process_dialogue(turns)

        # 存入内存
        for mem in new_memories:
            self.memories[mem.memory_id] = mem

        # 存入 Qdrant
        if self.qdrant and new_memories:
            batch = [
                {
                    "memory_id": mem.memory_id,
                    "embedding": mem.embedding,
                    "payload": mem.to_payload(),
                }
                for mem in new_memories
                if mem.embedding is not None
            ]
            self.qdrant.upsert_memories_batch(batch)

        self._memory_count_since_rebuild += len(new_memories)

        # 增量更新关联图
        if self._graph is not None:
            for mem in new_memories:
                self.graph.incremental_update(mem)

        # 更新主题摘要
        updated_topics = set()
        for mem in new_memories:
            for tid in mem.topic_ids:
                updated_topics.add(tid)
        for tid in updated_topics:
            topic = self.topics.get(tid)
            if topic and self._dag is not None:
                self.dag.update_topic_summary(topic)

        # 自动触发全局重构
        if self._memory_count_since_rebuild >= config.GLOBAL_REBUILD_INTERVAL:
            logger.info(f"累计新增 {self._memory_count_since_rebuild} 条记忆，触发全局索引重构")
            self.build_index()
            self._memory_count_since_rebuild = 0

        logger.info(
            f"对话处理完成: 新增 {len(new_memories)} 条记忆, "
            f"总计 {len(self.memories)} 条, {len(self.topics)} 个主题"
        )
        return new_memories

    def build_index(self):
        """
        构建/重构主题索引体系，并同步到 Neo4j 和 Qdrant

        1. 主题 DAG 构建（HAC 聚类 + 多父边）
        2. 主题关联图构建（共现 + LLM 因果 + 剪枝）
        3. 检测并处理主题合并
        4. 同步 Neo4j 图数据
        5. 同步 Qdrant 主题向量
        """
        logger.info("====== 开始构建主题索引 ======")

        # 重建 DAG
        self._dag = TopicDAG(
            self.topics, self.memories, self.emb_service, self.llm_service,
        )
        self.dag.build()

        # 重建关联图
        self._graph = TopicAssociationGraph(
            self.topics, self.memories, self.emb_service, self.llm_service,
        )
        self.graph.build()

        # 主题合并
        self.dag.check_and_merge_topics()

        # 同步到 Neo4j
        if self.neo4j:
            self.neo4j.sync_dag_from_memory(self.topics, self.graph.edges)

        # 同步主题向量到 Qdrant
        if self.qdrant:
            for tid, topic in self.topics.items():
                if topic.label_embedding is not None:
                    self.qdrant.upsert_topic_vectors(
                        topic_id=tid,
                        label_embedding=topic.label_embedding,
                        summary_embedding=topic.summary_embedding,
                        payload={
                            "label": topic.label,
                            "keywords": list(topic.keywords),
                            "is_virtual": topic.is_virtual,
                            "depth": topic.depth,
                        },
                    )

        # 重置检索器（使用最新数据）
        self._retriever = None

        logger.info("====== 主题索引构建完成 ======")

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """主题路由的记忆检索"""
        return self.retriever.retrieve(query, top_k=top_k)

    # ======================== 便捷方法 ========================

    def add_conversation(self, messages: list[dict]) -> list[Memory]:
        """从 [{"role": "user", "content": "..."}, ...] 格式添加对话"""
        turns = [
            DialogueTurn(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg.get("timestamp", datetime.now()),
                dia_id=msg.get("dia_id", ""),
                session_id=msg.get("session_id", ""),
            )
            for msg in messages
        ]
        return self.add_dialogue(turns)

    def add_locomo_session(self, turns: list[DialogueTurn]) -> list[Memory]:
        """
        添加 LoCoMo 的一个 session 的对话

        保留 dia_id 用于后续评测时计算 evidence recall
        """
        segments, new_memories = self.extractor.process_dialogue(turns)

        # 将 source_dia_ids 设置为段内所有对话 turn 的 dia_id
        seg_dia_map = {}
        for seg in segments:
            dia_ids = [t.dia_id for t in seg.turns if t.dia_id]
            seg_dia_map[seg.segment_id] = dia_ids

        for mem in new_memories:
            mem.source_dia_ids = seg_dia_map.get(mem.source_segment_id, [])
            self.memories[mem.memory_id] = mem

        # 同步到 Qdrant
        if self.qdrant and new_memories:
            batch = [
                {
                    "memory_id": mem.memory_id,
                    "embedding": mem.embedding,
                    "payload": mem.to_payload(),
                }
                for mem in new_memories
                if mem.embedding is not None
            ]
            self.qdrant.upsert_memories_batch(batch)

        self._memory_count_since_rebuild += len(new_memories)
        return new_memories

    def clear_all_data(self):
        """清空所有数据（内存 + 数据库）"""
        self.topics.clear()
        self.memories.clear()
        self._extractor = None
        self._dag = None
        self._graph = None
        self._retriever = None
        self._memory_count_since_rebuild = 0

        if self.neo4j:
            self.neo4j.clear_all()
        if self.qdrant:
            self.qdrant.clear_all()

        logger.info("所有数据已清空")

    def get_topic_tree_str(self, topic_id: str = None, indent: int = 0) -> str:
        """以缩进文本打印主题 DAG 结构"""
        from topic_dag import ROOT_TOPIC_ID
        if topic_id is None:
            topic_id = ROOT_TOPIC_ID

        topic = self.topics.get(topic_id)
        if not topic:
            return ""

        prefix = "  " * indent
        mem_count = len(topic.memory_ids)
        line = f"{prefix}{'[V] ' if topic.is_virtual else ''}{topic.label} ({mem_count} mems)\n"

        for cid in sorted(topic.child_ids):
            line += self.get_topic_tree_str(cid, indent + 1)

        return line

    def get_stats(self) -> dict:
        """获取系统统计信息"""
        stats = {
            "total_memories": len(self.memories),
            "total_topics": len([
                t for t in self.topics.values()
                if not t.is_virtual and t.topic_id != "__ROOT__"
            ]),
            "virtual_topics": len([t for t in self.topics.values() if t.is_virtual]),
            "association_edges": len(self.graph.edges) if self._graph else 0,
        }
        if self.qdrant:
            try:
                stats["qdrant_memories"] = self.qdrant.get_collection_count(
                    config.QDRANT_COLLECTION_MEMORIES
                )
                stats["qdrant_topics"] = self.qdrant.get_collection_count(
                    config.QDRANT_COLLECTION_TOPICS
                )
            except Exception:
                pass
        return stats
