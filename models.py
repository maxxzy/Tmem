"""
TMem 数据模型定义
定义记忆、主题、对话等核心数据结构
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid
import numpy as np


@dataclass
class DialogueTurn:
    """单轮对话"""
    role: str              # "user" 或 "assistant"（LoCoMo 中为 speaker 名字）
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    embedding: Optional[np.ndarray] = None  # 对话内容的向量表示
    dia_id: str = ""       # LoCoMo 格式的对话轮次 ID，如 "D1:3"
    session_id: str = ""   # 所属 session 标识


@dataclass
class TopicSegment:
    """
    主题段：由相邻对话组成的、主题一致的连续片段
    是主题分段的输出单元
    """
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turns: list[DialogueTurn] = field(default_factory=list)
    # 该主题段关联的主题 id 集合（一个主题段可关联多个主题）
    topic_ids: list[str] = field(default_factory=list)
    # LLM 为该段生成的原始主题标签及其关键词
    raw_labels: list[dict] = field(default_factory=list)
    # 格式: [{"label": "工作压力", "keywords": ["加班", "deadline"]}, ...]
    session_id: str = ""   # 来源 session


@dataclass
class Memory:
    """
    结构化记忆：从主题段中由 LLM 抽取的最小记忆单元
    每条记忆携带多个主题标签，用于后续检索
    """
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""                         # 记忆的文本内容
    topic_ids: list[str] = field(default_factory=list)  # 关联的主题 id 列表
    keywords: list[str] = field(default_factory=list)    # 关键词
    embedding: Optional[np.ndarray] = None    # 记忆内容的向量表示
    created_at: datetime = field(default_factory=datetime.now)
    source_segment_id: str = ""               # 来源主题段 id
    importance: float = 0.5                   # 重要性评分 [0,1]
    # LoCoMo 特有：关联的 dialogue turn IDs（用于评估 evidence recall）
    source_dia_ids: list[str] = field(default_factory=list)

    def to_payload(self) -> dict:
        """序列化为 Qdrant payload（不含向量）"""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "topic_ids": self.topic_ids,
            "keywords": self.keywords,
            "importance": self.importance,
            "source_segment_id": self.source_segment_id,
            "source_dia_ids": self.source_dia_ids,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Topic:
    """
    主题节点：主题 DAG 和关联图中的基本元素
    每个主题包含标签嵌入、关键词、记忆集合、摘要等多视角表征
    """
    topic_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""                           # 主题标签短语（如 "工作压力"）
    label_embedding: Optional[np.ndarray] = None   # 标签嵌入向量 e_i
    keywords: set[str] = field(default_factory=set)  # 关键词集合 K_i
    memory_ids: set[str] = field(default_factory=set)  # 归属记忆 id 集合 M_i
    summary: str = ""                         # 主题摘要 S_i（1-3 句概括）
    summary_embedding: Optional[np.ndarray] = None  # 摘要嵌入 e^s_i
    # DAG 结构
    parent_ids: set[str] = field(default_factory=set)   # 父主题 id 集合（DAG 允许多父）
    child_ids: set[str] = field(default_factory=set)    # 子主题 id 集合
    is_virtual: bool = False                  # 是否为聚类产生的虚拟节点
    depth: int = 0                            # 在 DAG 中的深度层级
    # 统计信息
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    memory_count_since_summary: int = 0       # 自上次摘要更新后新增记忆数
    # 关键词频率分布（用于主题漂移检测）
    keyword_distribution: dict[str, float] = field(default_factory=dict)


@dataclass
class AssociationEdge:
    """
    主题关联图中的有向边
    表示两个主题之间的非层次关系（因果、共现、时序等）
    """
    source_id: str = ""          # 源主题 id
    target_id: str = ""          # 目标主题 id
    weight: float = 0.0          # 边权重 [0, 1]
    edge_type: str = "co"        # 关联类型: co(共现), causal(因果), conditional, complementary, temporal
    # 各来源的分项权重
    npmi_mem: float = 0.0        # 记忆共现 NPMI
    llm_score: float = 0.0       # LLM 评分
    npmi_temp: float = 0.0       # 时序共现 NPMI
    # 时间信息（用于半衰期衰减）
    last_updated: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RetrievalResult:
    """检索结果：一条被召回的记忆及其评分信息"""
    memory: Memory
    score: float = 0.0           # 综合得分
    source_type: str = "intra"   # "intra"(主题内) 或 "cross"(跨主题扩展)
    matched_topics: list[str] = field(default_factory=list)  # 命中的主题
