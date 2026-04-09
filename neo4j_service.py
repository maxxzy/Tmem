"""
Neo4j 图数据库服务

将主题 DAG 和主题关联图持久化到 Neo4j 中：
  - Topic 节点：存储主题属性（标签、摘要、关键词、深度等）
  - PARENT_OF 关系：DAG 中的父子层次边
  - ASSOCIATED_WITH 关系：主题关联图中的加权有向边
  - HAS_MEMORY 关系：主题与记忆的归属关系
"""

import logging
from datetime import datetime

from neo4j import GraphDatabase

import config

logger = logging.getLogger(__name__)


class Neo4jService:
    """Neo4j 图数据库操作封装"""

    def __init__(
        self,
        uri: str = config.NEO4J_URI,
        user: str = config.NEO4J_USER,
        password: str = config.NEO4J_PASSWORD,
        database: str = config.NEO4J_DATABASE,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._ensure_constraints()
        logger.info(f"Neo4j 连接成功: {uri}")

    def close(self):
        self.driver.close()

    def _run(self, query: str, **params):
        """执行一条 Cypher 语句"""
        with self.driver.session(database=self.database) as session:
            return session.run(query, **params).data()

    def _ensure_constraints(self):
        """创建唯一约束和索引"""
        with self.driver.session(database=self.database) as session:
            # Topic 节点的 topic_id 唯一约束
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE"
            )
            # Memory 节点的 memory_id 唯一约束
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE"
            )

    def clear_all(self):
        """清空数据库中所有 TMem 相关数据"""
        self._run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j 数据已清空")

    # ======================== Topic 节点 CRUD ========================

    def upsert_topic(
        self,
        topic_id: str,
        label: str,
        keywords: list[str],
        summary: str = "",
        is_virtual: bool = False,
        depth: int = 0,
        memory_count: int = 0,
    ):
        """创建或更新 Topic 节点"""
        self._run(
            """
            MERGE (t:Topic {topic_id: $topic_id})
            SET t.label = $label,
                t.keywords = $keywords,
                t.summary = $summary,
                t.is_virtual = $is_virtual,
                t.depth = $depth,
                t.memory_count = $memory_count,
                t.updated_at = datetime()
            """,
            topic_id=topic_id,
            label=label,
            keywords=keywords,
            summary=summary,
            is_virtual=is_virtual,
            depth=depth,
            memory_count=memory_count,
        )

    def delete_topic(self, topic_id: str):
        """删除 Topic 节点及其所有关系"""
        self._run(
            "MATCH (t:Topic {topic_id: $topic_id}) DETACH DELETE t",
            topic_id=topic_id,
        )

    def get_topic(self, topic_id: str) -> dict | None:
        """获取单个 Topic 节点"""
        result = self._run(
            "MATCH (t:Topic {topic_id: $topic_id}) RETURN t",
            topic_id=topic_id,
        )
        return result[0]["t"] if result else None

    def get_all_topics(self) -> list[dict]:
        """获取所有 Topic 节点"""
        return self._run("MATCH (t:Topic) RETURN t")

    # ======================== DAG 父子关系 ========================

    def add_parent_edge(self, child_id: str, parent_id: str):
        """添加 DAG 父子关系边 (parent)-[:PARENT_OF]->(child)"""
        self._run(
            """
            MATCH (p:Topic {topic_id: $parent_id})
            MATCH (c:Topic {topic_id: $child_id})
            MERGE (p)-[:PARENT_OF]->(c)
            """,
            parent_id=parent_id,
            child_id=child_id,
        )

    def remove_parent_edge(self, child_id: str, parent_id: str):
        """移除 DAG 父子关系边"""
        self._run(
            """
            MATCH (p:Topic {topic_id: $parent_id})-[r:PARENT_OF]->(c:Topic {topic_id: $child_id})
            DELETE r
            """,
            parent_id=parent_id,
            child_id=child_id,
        )

    def get_children(self, topic_id: str) -> list[str]:
        """获取一个主题的所有子节点 id"""
        result = self._run(
            """
            MATCH (p:Topic {topic_id: $topic_id})-[:PARENT_OF]->(c:Topic)
            RETURN c.topic_id AS child_id
            """,
            topic_id=topic_id,
        )
        return [r["child_id"] for r in result]

    def get_parents(self, topic_id: str) -> list[str]:
        """获取一个主题的所有父节点 id"""
        result = self._run(
            """
            MATCH (p:Topic)-[:PARENT_OF]->(c:Topic {topic_id: $topic_id})
            RETURN p.topic_id AS parent_id
            """,
            topic_id=topic_id,
        )
        return [r["parent_id"] for r in result]

    def get_siblings(self, topic_id: str) -> list[str]:
        """获取兄弟节点 id"""
        result = self._run(
            """
            MATCH (p:Topic)-[:PARENT_OF]->(c:Topic {topic_id: $topic_id})
            MATCH (p)-[:PARENT_OF]->(s:Topic)
            WHERE s.topic_id <> $topic_id
            RETURN DISTINCT s.topic_id AS sibling_id
            """,
            topic_id=topic_id,
        )
        return [r["sibling_id"] for r in result]

    def get_descendants(self, topic_id: str) -> list[str]:
        """获取所有后代节点（可变深度路径）"""
        result = self._run(
            """
            MATCH (p:Topic {topic_id: $topic_id})-[:PARENT_OF*1..]->(d:Topic)
            RETURN DISTINCT d.topic_id AS desc_id
            """,
            topic_id=topic_id,
        )
        return [r["desc_id"] for r in result]

    def get_ancestors(self, topic_id: str) -> list[str]:
        """获取所有祖先节点"""
        result = self._run(
            """
            MATCH (a:Topic)-[:PARENT_OF*1..]->(c:Topic {topic_id: $topic_id})
            RETURN DISTINCT a.topic_id AS anc_id
            """,
            topic_id=topic_id,
        )
        return [r["anc_id"] for r in result]

    # ======================== 关联图边 ========================

    def upsert_association_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float,
        edge_type: str = "co",
        npmi_mem: float = 0.0,
        llm_score: float = 0.0,
        npmi_temp: float = 0.0,
    ):
        """创建或更新关联图中的有向边"""
        self._run(
            """
            MATCH (s:Topic {topic_id: $source_id})
            MATCH (t:Topic {topic_id: $target_id})
            MERGE (s)-[r:ASSOCIATED_WITH]->(t)
            SET r.weight = $weight,
                r.edge_type = $edge_type,
                r.npmi_mem = $npmi_mem,
                r.llm_score = $llm_score,
                r.npmi_temp = $npmi_temp,
                r.last_updated = datetime()
            """,
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            edge_type=edge_type,
            npmi_mem=npmi_mem,
            llm_score=llm_score,
            npmi_temp=npmi_temp,
        )

    def remove_association_edge(self, source_id: str, target_id: str):
        """移除关联图边"""
        self._run(
            """
            MATCH (s:Topic {topic_id: $source_id})-[r:ASSOCIATED_WITH]->(t:Topic {topic_id: $target_id})
            DELETE r
            """,
            source_id=source_id,
            target_id=target_id,
        )

    def get_association_edges(self, source_id: str) -> list[dict]:
        """获取一个主题的所有出边"""
        return self._run(
            """
            MATCH (s:Topic {topic_id: $source_id})-[r:ASSOCIATED_WITH]->(t:Topic)
            RETURN t.topic_id AS target_id, r.weight AS weight,
                   r.edge_type AS edge_type, r.npmi_mem AS npmi_mem,
                   r.llm_score AS llm_score, r.npmi_temp AS npmi_temp
            ORDER BY r.weight DESC
            """,
            source_id=source_id,
        )

    def get_all_association_edges(self) -> list[dict]:
        """获取所有关联图边"""
        return self._run(
            """
            MATCH (s:Topic)-[r:ASSOCIATED_WITH]->(t:Topic)
            RETURN s.topic_id AS source_id, t.topic_id AS target_id,
                   r.weight AS weight, r.edge_type AS edge_type,
                   r.npmi_mem AS npmi_mem, r.llm_score AS llm_score,
                   r.npmi_temp AS npmi_temp
            """
        )

    def clear_association_edges(self):
        """清除所有关联图边（保留 DAG 边）"""
        self._run("MATCH ()-[r:ASSOCIATED_WITH]->() DELETE r")

    # ======================== 主题-记忆归属关系 ========================

    def upsert_memory_node(self, memory_id: str, content: str):
        """创建或更新 Memory 节点（轻量节点，详细数据在 Qdrant 中）"""
        self._run(
            """
            MERGE (m:Memory {memory_id: $memory_id})
            SET m.content = $content
            """,
            memory_id=memory_id,
            content=content,
        )

    def add_topic_memory_edge(self, topic_id: str, memory_id: str):
        """建立主题与记忆的归属关系"""
        self._run(
            """
            MATCH (t:Topic {topic_id: $topic_id})
            MATCH (m:Memory {memory_id: $memory_id})
            MERGE (t)-[:HAS_MEMORY]->(m)
            """,
            topic_id=topic_id,
            memory_id=memory_id,
        )

    def get_memory_ids_by_topic(self, topic_id: str) -> list[str]:
        """获取主题下所有记忆 id"""
        result = self._run(
            """
            MATCH (t:Topic {topic_id: $topic_id})-[:HAS_MEMORY]->(m:Memory)
            RETURN m.memory_id AS memory_id
            """,
            topic_id=topic_id,
        )
        return [r["memory_id"] for r in result]

    def get_topic_ids_by_memory(self, memory_id: str) -> list[str]:
        """获取一条记忆归属的所有主题 id"""
        result = self._run(
            """
            MATCH (t:Topic)-[:HAS_MEMORY]->(m:Memory {memory_id: $memory_id})
            RETURN t.topic_id AS topic_id
            """,
            memory_id=memory_id,
        )
        return [r["topic_id"] for r in result]

    # ======================== DAG 批量同步 ========================

    def sync_dag_from_memory(self, topics: dict, edges_assoc: dict):
        """
        将内存中的 DAG 和关联图批量同步到 Neo4j

        Args:
            topics: topic_id -> Topic 对象字典
            edges_assoc: (source_id, target_id) -> AssociationEdge 对象字典
        """
        # 先清空再重建（全量同步策略，适合 build_index 场景）
        self.clear_all()

        # 1. 创建所有 Topic 节点
        for tid, topic in topics.items():
            self.upsert_topic(
                topic_id=tid,
                label=topic.label,
                keywords=list(topic.keywords),
                summary=topic.summary,
                is_virtual=topic.is_virtual,
                depth=topic.depth,
                memory_count=len(topic.memory_ids),
            )

        # 2. 创建 DAG 父子边
        for tid, topic in topics.items():
            for pid in topic.parent_ids:
                if pid in topics:
                    self.add_parent_edge(child_id=tid, parent_id=pid)

        # 3. 创建 Memory 节点和归属边
        memory_ids_seen = set()
        for tid, topic in topics.items():
            for mid in topic.memory_ids:
                if mid not in memory_ids_seen:
                    self.upsert_memory_node(mid, "")
                    memory_ids_seen.add(mid)
                self.add_topic_memory_edge(tid, mid)

        # 4. 创建关联图边
        for (src, tgt), edge in edges_assoc.items():
            self.upsert_association_edge(
                source_id=src,
                target_id=tgt,
                weight=edge.weight,
                edge_type=edge.edge_type,
                npmi_mem=edge.npmi_mem,
                llm_score=edge.llm_score,
                npmi_temp=edge.npmi_temp,
            )

        logger.info(
            f"Neo4j 同步完成: {len(topics)} 个主题, "
            f"{len(memory_ids_seen)} 条记忆, "
            f"{len(edges_assoc)} 条关联边"
        )
