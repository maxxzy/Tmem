"""
主题感知的记忆抽取模块

核心思路：用主题边界检测替代固定窗口，确保每个抽取单元是完整的主题段。
流程：
  1. 相邻对话计算语义相似度，结合转折词检测主题边界 → 分割为主题段
  2. 每个主题段由 LLM 生成主题标签和关键词，与已有主题匹配或创建新主题
  3. 主题段内用 LLM 生成结构化记忆，记忆从该段的主题集合中选取关联主题
"""

import logging
from datetime import datetime

import re

import numpy as np

import config
from models import DialogueTurn, TopicSegment, Memory, Topic
from embedding_service import EmbeddingService
from llm_service import LLMService

logger = logging.getLogger(__name__)


class TopicExtractor:
    """主题感知的记忆抽取器"""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        existing_topics: dict[str, Topic] | None = None,
    ):
        self.emb = embedding_service
        self.llm = llm_service
        # 已有主题集合（topic_id -> Topic），用于新主题标签的归并判断
        self.topics = existing_topics if existing_topics is not None else {}

    # ======================== 步骤 1：主题边界检测与分段 ========================

    def detect_topic_boundaries(
        self, turns: list[DialogueTurn]
    ) -> list[int]:
        """
        检测对话中的主题边界位置

        方法：
        - 计算相邻对话的嵌入余弦相似度，低于阈值时标记为候选边界
        - 同时检测转折词作为辅助信号

        Args:
            turns: 按时间排序的对话列表

        Returns:
            边界位置索引列表（表示在该索引之前切分）
        """
        if len(turns) <= 1:
            return []

        # 确保每轮对话都有嵌入向量
        for turn in turns:
            if turn.embedding is None:
                turn.embedding = self.emb.encode(turn.content)

        boundaries = []
        for i in range(1, len(turns)):
            # 计算相邻对话的语义相似度
            sim = EmbeddingService.cosine_similarity(
                turns[i - 1].embedding, turns[i].embedding
            )
            # 检测当前对话是否包含转折词（单词边界 + 大小写不敏感）
            has_transition = any(
                re.search(r'\b' + re.escape(tw) + r'\b', turns[i].content, re.IGNORECASE)
                for tw in config.TRANSITION_WORDS
            )
            # 相似度低于阈值，或包含转折词且相似度不太高 → 主题边界
            if sim < config.TOPIC_BOUNDARY_SIMILARITY_THRESHOLD:
                boundaries.append(i)
            elif has_transition and sim < config.TOPIC_BOUNDARY_SIMILARITY_THRESHOLD + 0.15:
                # 转折词出现时适当放宽相似度阈值
                boundaries.append(i)

        return boundaries

    def segment_dialogue(
        self, turns: list[DialogueTurn]
    ) -> list[TopicSegment]:
        """
        将对话序列按主题边界分割为多个主题段

        Args:
            turns: 对话列表

        Returns:
            主题段列表，每个段包含连续的、主题一致的对话
        """
        boundaries = self.detect_topic_boundaries(turns)

        # 过滤掉导致过短段的边界（最小段长度约束）
        min_turns = config.MIN_SEGMENT_TURNS
        filtered = []
        prev = 0
        for b in boundaries:
            if b - prev >= min_turns:
                filtered.append(b)
                prev = b
        # 确保最后一段也不会太短：如果最后一段 < min_turns 则移除最后一个边界
        if filtered and len(turns) - filtered[-1] < min_turns:
            filtered.pop()

        segments = []
        split_points = [0] + filtered + [len(turns)]

        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i + 1]
            if start < end:
                seg = TopicSegment(turns=turns[start:end])
                segments.append(seg)

        logger.info(f"对话共 {len(turns)} 轮，分割为 {len(segments)} 个主题段")
        return segments

    # ======================== 步骤 2：主题标签生成与归并 ========================

    def _segment_to_text(self, segment: TopicSegment) -> str:
        """将主题段的对话拼接为纯文本，供 LLM 处理"""
        lines = []
        for turn in segment.turns:
            role = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role}: {turn.content}")
        return "\n".join(lines)

    def _match_existing_topic(
        self, label: str, label_emb: np.ndarray, keywords: set[str] | None = None,
    ) -> str | None:
        """
        将新生成的主题标签与已有主题比较（多信号融合）

        综合相似度 = α1·cos(label_emb) + α2·cos(summary_emb) + α3·Jaccard(keywords) + α4·token_overlap
        当 summary 不可用时，其权重转移给 label embedding。
        若最高相似度超过阈值，归入已有主题并返回其 id；否则返回 None。
        """
        a1, a2, a3, a4 = config.TOPIC_SIM_WEIGHTS
        if keywords is None:
            keywords = set()

        new_tokens = set(label.lower().split())

        best_sim = 0.0
        best_topic_id = None
        for tid, topic in self.topics.items():
            if topic.label_embedding is None:
                continue
            sim = 0.0

            # 标签嵌入相似度
            label_sim = EmbeddingService.cosine_similarity(label_emb, topic.label_embedding)
            sim += a1 * label_sim

            # 摘要嵌入相似度（新标签无摘要，权重转移给 label）
            if topic.summary_embedding is not None:
                summary_sim = EmbeddingService.cosine_similarity(label_emb, topic.summary_embedding)
                sim += a2 * summary_sim
            else:
                sim += a2 * label_sim

            # 关键词 Jaccard
            if keywords or topic.keywords:
                intersection = keywords & topic.keywords
                union = keywords | topic.keywords
                jaccard = len(intersection) / len(union) if union else 0.0
                sim += a3 * jaccard

            # Token 重叠率（containment 度量，捕捉子串关系）
            existing_tokens = set(topic.label.lower().split())
            if new_tokens and existing_tokens:
                overlap = len(new_tokens & existing_tokens) / min(len(new_tokens), len(existing_tokens))
                sim += a4 * overlap

            if sim > best_sim:
                best_sim = sim
                best_topic_id = tid

        if best_sim >= config.TOPIC_MERGE_SIMILARITY_THRESHOLD:
            logger.debug(f"主题 '{label}' 归入已有主题 '{self.topics[best_topic_id].label}' (sim={best_sim:.3f})")
            return best_topic_id
        return None

    def assign_topics_to_segment(self, segment: TopicSegment) -> list[str]:
        """
        为主题段分配主题 id 列表

        流程：
        1. LLM 生成主题标签和关键词
        2. 每个标签与已有主题匹配：匹配上则归入，否则创建新主题

        Returns:
            该主题段关联的 topic_id 列表
        """
        text = self._segment_to_text(segment)
        raw_labels = self.llm.generate_topic_labels(text)
        segment.raw_labels = raw_labels

        topic_ids = []
        for item in raw_labels:
            # 容错：跳过非 dict/str 类型的元素
            if isinstance(item, str):
                item = {"label": item, "keywords": []}
            if not isinstance(item, dict):
                continue
            label = item.get("label", "未知主题")
            keywords = set(item.get("keywords", []))
            label_emb = self.emb.encode(label)

            # 尝试归入已有主题
            matched_id = self._match_existing_topic(label, label_emb, keywords)
            if matched_id:
                # 已有主题：合并关键词
                self.topics[matched_id].keywords |= keywords
                topic_ids.append(matched_id)
            else:
                # 创建新主题
                new_topic = Topic(
                    label=label,
                    label_embedding=label_emb,
                    keywords=keywords,
                )
                self.topics[new_topic.topic_id] = new_topic
                topic_ids.append(new_topic.topic_id)
                logger.info(f"创建新主题: '{label}' (id={new_topic.topic_id[:8]})")

        # 去重（多个 LLM 标签可能匹配到同一已有主题）
        topic_ids = list(dict.fromkeys(topic_ids))
        segment.topic_ids = topic_ids
        return topic_ids

    # ======================== 步骤 3：结构化记忆抽取 ========================

    def extract_memories_from_segment(
        self, segment: TopicSegment
    ) -> list[Memory]:
        """
        从主题段中抽取结构化记忆

        LLM 被要求从该段关联的主题集合中选取每条记忆涉及的主题，
        确保不会产生多余的新主题。

        Args:
            segment: 已完成主题分配的主题段

        Returns:
            结构化记忆列表
        """
        text = self._segment_to_text(segment)
        # 收集该主题段关联的所有主题标签
        topic_labels = [
            self.topics[tid].label
            for tid in segment.topic_ids
            if tid in self.topics
        ]
        # LLM 抽取记忆
        raw_memories = self.llm.extract_memories(text, topic_labels)

        if not raw_memories:
            logger.warning(f"主题段 {segment.segment_id[:8]}: LLM 返回空记忆列表")
        else:
            valid_count = sum(1 for r in raw_memories if isinstance(r, dict) and r.get("content"))
            if valid_count == 0:
                logger.warning(
                    f"主题段 {segment.segment_id[:8]}: LLM 返回 {len(raw_memories)} 条记录但均无有效 content, "
                    f"样例: {raw_memories[0] if raw_memories else 'N/A'}"
                )

        memories = []
        for raw in raw_memories:
            if not isinstance(raw, dict):
                continue
            content = raw.get("content", "")
            if not content:
                continue

            # 将 LLM 返回的主题标签名映射回 topic_id
            mem_topic_ids = []
            for topic_name in raw.get("topics", []):
                matched_tid = None
                topic_name_stripped = topic_name.strip()
                topic_name_lower = topic_name_stripped.lower()
                # 第一级：大小写不敏感 + strip 的字符串匹配
                for tid in segment.topic_ids:
                    if tid in self.topics and self.topics[tid].label.strip().lower() == topic_name_lower:
                        matched_tid = tid
                        break
                # 第二级：embedding 相似度匹配
                if matched_tid is None and topic_name_stripped:
                    topic_name_emb = self.emb.encode(topic_name_stripped)
                    best_sim = 0.0
                    best_tid = None
                    for tid in segment.topic_ids:
                        if tid in self.topics and self.topics[tid].label_embedding is not None:
                            sim = EmbeddingService.cosine_similarity(topic_name_emb, self.topics[tid].label_embedding)
                            if sim > best_sim:
                                best_sim = sim
                                best_tid = tid
                    if best_tid and best_sim >= config.TOPIC_MERGE_SIMILARITY_THRESHOLD:
                        matched_tid = best_tid
                if matched_tid:
                    mem_topic_ids.append(matched_tid)
            # 如果没有成功映射任何主题，则使用段的全部主题
            if not mem_topic_ids:
                mem_topic_ids = list(segment.topic_ids)

            importance_raw = raw.get("importance", 0.5)
            try:
                importance = float(importance_raw)
            except (ValueError, TypeError):
                importance = 0.5

            memory = Memory(
                content=content,
                topic_ids=mem_topic_ids,
                keywords=raw.get("keywords", []),
                embedding=self.emb.encode(content),
                importance=importance,
                source_segment_id=segment.segment_id,
            )
            memories.append(memory)

            # 将记忆 id 注册到对应主题
            for tid in mem_topic_ids:
                if tid in self.topics:
                    self.topics[tid].memory_ids.add(memory.memory_id)
                    self.topics[tid].memory_count_since_summary += 1

        logger.info(f"从主题段 {segment.segment_id[:8]} 抽取 {len(memories)} 条记忆")
        return memories

    # ======================== 完整抽取流程 ========================

    def process_dialogue(
        self, turns: list[DialogueTurn]
    ) -> tuple[list[TopicSegment], list[Memory]]:
        """
        完整的主题感知记忆抽取流程

        Args:
            turns: 一段对话的全部轮次

        Returns:
            (主题段列表, 记忆列表)
        """
        # 1. 主题分段
        segments = self.segment_dialogue(turns)

        all_memories = []
        for seg in segments:
            # 2. 为每个段分配主题
            self.assign_topics_to_segment(seg)
            # 3. 从每个段抽取结构化记忆
            memories = self.extract_memories_from_segment(seg)
            all_memories.extend(memories)

        logger.info(
            f"对话处理完成: {len(segments)} 个主题段, "
            f"{len(all_memories)} 条记忆, {len(self.topics)} 个主题"
        )
        return segments, all_memories
