"""
主题路由的记忆检索模块

核心流程：先主题路由 → 后主题内检索 → 再跨主题扩展
  1. 识别 query 的相关主题（Qdrant 向量检索 + 关键词匹配）
  2. 利用主题 DAG（Neo4j）调整候选主题集（扩展或缩减）
  3. 在候选主题下的记忆集中检索（Qdrant 标量过滤 + 向量检索）
  4. 利用主题关联图跨主题扩展（PPR spreading activation）
"""

import logging

import numpy as np

import config
from models import Topic, Memory, RetrievalResult
from embedding_service import EmbeddingService
from qdrant_service import QdrantService
from neo4j_service import Neo4jService
from topic_dag import TopicDAG, ROOT_TOPIC_ID
from topic_graph import TopicAssociationGraph

logger = logging.getLogger(__name__)


class TopicRetriever:
    """主题路由的记忆检索器"""

    def __init__(
        self,
        topics: dict[str, Topic],
        memories: dict[str, Memory],
        embedding_service: EmbeddingService,
        topic_dag: TopicDAG,
        topic_graph: TopicAssociationGraph,
        qdrant_service: QdrantService | None = None,
        neo4j_service: Neo4jService | None = None,
    ):
        self.topics = topics
        self.memories = memories
        self.emb = embedding_service
        self.dag = topic_dag
        self.graph = topic_graph
        self.qdrant = qdrant_service
        self.neo4j = neo4j_service

    # ======================== 步骤 1：主题路由 ========================

    def route_to_topics(self, query: str, query_emb: np.ndarray) -> list[tuple[str, float]]:
        """
        识别 query 的相关主题

        当 Qdrant 可用时，利用向量数据库快速检索最匹配的主题；
        否则回退到内存遍历方式。

        Returns:
            [(topic_id, match_score), ...] 按匹配得分降序
        """
        query_lower = query.lower()

        if self.qdrant is not None:
            return self._route_via_qdrant(query_emb, query_lower)
        return self._route_via_memory(query_emb, query_lower)

    def _route_via_qdrant(
        self, query_emb: np.ndarray, query_lower: str
    ) -> list[tuple[str, float]]:
        """
        利用 Qdrant 向量检索进行主题路由

        分别在标签嵌入和摘要嵌入空间中检索，合并得分
        """
        top_n = config.MAX_ROUTED_TOPICS * 3  # 多检索一些用于后续筛选

        # 标签向量检索
        label_hits = self.qdrant.search_topics_by_label(query_emb, top_k=top_n)
        # 摘要向量检索
        summary_hits = self.qdrant.search_topics_by_summary(query_emb, top_k=top_n)

        # 合并得分
        score_map: dict[str, float] = {}
        for hit in label_hits:
            tid = hit["topic_id"]
            if tid in self.topics and not self.topics[tid].is_virtual:
                score_map[tid] = score_map.get(tid, 0) + 0.4 * hit["score"]

        for hit in summary_hits:
            tid = hit["topic_id"]
            if tid in self.topics and not self.topics[tid].is_virtual:
                score_map[tid] = score_map.get(tid, 0) + 0.35 * hit["score"]

        # 关键词匹配加分
        for tid, topic in self.topics.items():
            if tid not in score_map or topic.is_virtual:
                continue
            if topic.keywords:
                matched = sum(1 for kw in topic.keywords if kw.lower() in query_lower)
                score_map[tid] += 0.25 * (matched / len(topic.keywords))

        # 过滤掉没有记忆的空主题（路由到空主题毫无意义）
        score_map = {
            tid: s for tid, s in score_map.items()
            if self.topics[tid].memory_ids
        }

        result = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return result

    def _route_via_memory(
        self, query_emb: np.ndarray, query_lower: str
    ) -> list[tuple[str, float]]:
        """回退方式：遍历内存中的主题进行主题路由（无 Qdrant 时使用）"""
        scored_topics = []
        for tid, topic in self.topics.items():
            if topic.is_virtual or tid == ROOT_TOPIC_ID:
                continue
            # 跳过没有记忆的空主题
            if not topic.memory_ids:
                continue

            score = 0.0
            if topic.label_embedding is not None:
                label_sim = EmbeddingService.cosine_similarity(query_emb, topic.label_embedding)
                score += 0.4 * max(0, label_sim)

            if topic.summary_embedding is not None:
                summary_sim = EmbeddingService.cosine_similarity(query_emb, topic.summary_embedding)
                score += 0.35 * max(0, summary_sim)
            elif topic.label_embedding is not None:
                score += 0.35 * max(0, label_sim)

            if topic.keywords:
                matched = sum(1 for kw in topic.keywords if kw.lower() in query_lower)
                score += 0.25 * (matched / len(topic.keywords))

            if score > 0:
                scored_topics.append((tid, score))

        scored_topics.sort(key=lambda x: x[1], reverse=True)
        return scored_topics

    # ======================== 步骤 2：DAG 调整候选主题集 ========================

    def adjust_topics_by_dag(
        self, scored_topics: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """
        利用主题 DAG 调整候选主题集：
        - 不足时沿 DAG 向上扩展兄弟主题
        - 过多时缩减有子主题已入选的父主题
        """
        if not scored_topics:
            return scored_topics

        topic_scores = {tid: s for tid, s in scored_topics}

        # 扩展：主题不足时加入兄弟主题
        if len(topic_scores) < config.MIN_ROUTED_TOPICS:
            for tid in list(topic_scores.keys()):
                if self.neo4j:
                    siblings = self.neo4j.get_siblings(tid)
                else:
                    siblings = list(self.dag.get_siblings(tid))
                for sib_id in siblings:
                    if sib_id not in topic_scores:
                        topic_scores[sib_id] = topic_scores.get(tid, 0) * 0.7

        # 缩减：删除有后代已入选的父主题
        current_ids = set(topic_scores.keys())
        if len(current_ids) > config.MAX_ROUTED_TOPICS:
            to_remove = set()
            for tid in current_ids:
                if self.neo4j:
                    descendants = set(self.neo4j.get_descendants(tid))
                else:
                    descendants = self.dag.get_descendants(tid)
                if any(d in current_ids and d != tid for d in descendants):
                    to_remove.add(tid)
            for tid in to_remove:
                topic_scores.pop(tid, None)

        result = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return result[:config.MAX_ROUTED_TOPICS]

    # ======================== 步骤 3：主题内记忆检索 ========================

    def retrieve_intra_topic(
        self,
        query: str,
        query_emb: np.ndarray,
        candidate_topics: list[tuple[str, float]],
        scoring_topic_ids: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """
        在候选主题下的记忆集中检索

        当 Qdrant 可用时，利用标量过滤 + 向量检索实现高效主题内检索；
        否则回退到内存遍历方式。
        """
        if self.qdrant is not None:
            return self._intra_via_qdrant(query, query_emb, candidate_topics, scoring_topic_ids)
        return self._intra_via_memory(query, query_emb, candidate_topics, scoring_topic_ids)

    def _intra_via_qdrant(
        self,
        query: str,
        query_emb: np.ndarray,
        candidate_topics: list[tuple[str, float]],
        scoring_topic_ids: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """利用 Qdrant 的标量过滤进行主题内向量检索"""
        topic_ids = [tid for tid, _ in candidate_topics]
        topic_score_map = {tid: s for tid, s in candidate_topics}

        # 在 Qdrant 中按主题过滤检索
        hits = self.qdrant.search_memories_by_topics(
            query_embedding=query_emb,
            topic_ids=topic_ids,
            top_k=config.INTRA_TOPIC_TOP_K * 2,  # 多取一些供后续排序
        )

        query_lower = query.lower()
        results = []
        for hit in hits:
            mid = hit["memory_id"]
            mem = self.memories.get(mid)
            if not mem:
                continue

            vec_sim = hit["score"]

            # 关键词匹配加分
            keyword_bonus = 0.0
            keywords = hit["payload"].get("keywords", [])
            if keywords:
                matched = sum(1 for kw in keywords if kw.lower() in query_lower)
                keyword_bonus = 0.1 * (matched / max(len(keywords), 1))

            # 主题路由得分加成（取命中主题中的最高得分）
            mem_topics = hit["payload"].get("topic_ids", [])
            best_topic_score = max(
                (topic_score_map.get(t, 0) for t in mem_topics), default=0
            )

            importance = hit["payload"].get("importance", 0.5)

            # 仅评分主题内的记忆获得 topic bonus
            mem_topic_set = set(mem_topics)
            if scoring_topic_ids and mem_topic_set & scoring_topic_ids:
                score = vec_sim + 0.20 * best_topic_score + keyword_bonus + 0.08 * importance
            else:
                score = vec_sim

            matched_ts = [t for t in mem_topics if t in topic_score_map]
            results.append(RetrievalResult(
                memory=mem,
                score=score,
                source_type="intra",
                matched_topics=matched_ts,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:config.INTRA_TOPIC_TOP_K]

    def _intra_via_memory(
        self,
        query: str,
        query_emb: np.ndarray,
        candidate_topics: list[tuple[str, float]],
        scoring_topic_ids: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """回退方式：遍历内存中的记忆进行主题内检索"""
        query_lower = query.lower()
        results = []
        seen = set()

        for tid, topic_score in candidate_topics:
            topic = self.topics.get(tid)
            if not topic:
                continue
            for mid in topic.memory_ids:
                if mid in seen:
                    continue
                seen.add(mid)
                mem = self.memories.get(mid)
                if not mem or mem.embedding is None:
                    continue

                vec_sim = EmbeddingService.cosine_similarity(query_emb, mem.embedding)
                keyword_bonus = 0.0
                if mem.keywords:
                    matched = sum(1 for kw in mem.keywords if kw.lower() in query_lower)
                    keyword_bonus = 0.1 * (matched / max(len(mem.keywords), 1))

                # 仅评分主题内的记忆获得 topic bonus
                if scoring_topic_ids and tid in scoring_topic_ids:
                    score = vec_sim + 0.20 * topic_score + keyword_bonus + 0.08 * mem.importance
                else:
                    score = vec_sim

                results.append(RetrievalResult(
                    memory=mem, score=score, source_type="intra", matched_topics=[tid],
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:config.INTRA_TOPIC_TOP_K]

    # ======================== 步骤 4：跨主题扩展 ========================

    def should_cross_topic_expand(
        self, query: str, intra_results: list[RetrievalResult]
    ) -> bool:
        """判断是否需要跨主题扩展"""
        # 主题内无结果时必须扩展
        if not intra_results:
            return True
        query_lower = query.lower()
        for kw in config.REASONING_KEYWORDS:
            if kw in query_lower:
                return True
        for kw in config.ENUMERATION_KEYWORDS:
            if kw in query_lower:
                return True
        if intra_results:
            avg_score = sum(r.score for r in intra_results) / len(intra_results)
            if avg_score < config.LOW_SCORE_THRESHOLD:
                return True
        return False

    def retrieve_cross_topic(
        self,
        query_emb: np.ndarray,
        seed_topic_ids: list[str],
        existing_memory_ids: set[str],
    ) -> list[RetrievalResult]:
        """利用 PPR 在关联图上进行跨主题扩展检索"""
        expanded = self.graph.personalized_pagerank(seed_topic_ids)
        if not expanded:
            return []

        results = []
        for tid, activation in expanded:
            topic = self.topics.get(tid)
            if not topic:
                continue

            # Qdrant 按单主题过滤检索
            if self.qdrant is not None:
                hits = self.qdrant.search_memories_by_topics(
                    query_embedding=query_emb,
                    topic_ids=[tid],
                    top_k=5,
                )
                for hit in hits:
                    mid = hit["memory_id"]
                    if mid in existing_memory_ids:
                        continue
                    mem = self.memories.get(mid)
                    if not mem:
                        continue
                    score = hit["score"] * activation * config.CROSS_TOPIC_WEIGHT_DISCOUNT
                    results.append(RetrievalResult(
                        memory=mem, score=score, source_type="cross", matched_topics=[tid],
                    ))
            else:
                for mid in topic.memory_ids:
                    if mid in existing_memory_ids:
                        continue
                    mem = self.memories.get(mid)
                    if not mem or mem.embedding is None:
                        continue
                    vec_sim = EmbeddingService.cosine_similarity(query_emb, mem.embedding)
                    score = vec_sim * activation * config.CROSS_TOPIC_WEIGHT_DISCOUNT
                    results.append(RetrievalResult(
                        memory=mem, score=score, source_type="cross", matched_topics=[tid],
                    ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:config.PPR_TOP_K]

    # ======================== MMR 多样性重排 ========================

    def _mmr_select(
        self, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """
        MMR (Maximal Marginal Relevance) 多样性重排

        兼顾相关性和主题多样性，避免 top-k 结果集中在同一主题段。
        """
        if len(candidates) <= top_k:
            return candidates

        max_score = max(r.score for r in candidates)
        if max_score <= 0:
            return candidates[:top_k]

        selected: list[int] = []
        remaining = list(range(len(candidates)))

        # 第一条：取最高分
        best_idx = max(remaining, key=lambda i: candidates[i].score)
        selected.append(best_idx)
        remaining.remove(best_idx)

        lam = config.MMR_LAMBDA
        topic_w = config.MMR_TOPIC_WEIGHT

        for _ in range(top_k - 1):
            if not remaining:
                break
            best_mmr = -float('inf')
            best_i = remaining[0]
            for i in remaining:
                relevance = candidates[i].score / max_score
                max_redundancy = 0.0
                topics_i = set(candidates[i].memory.topic_ids)
                for j in selected:
                    topics_j = set(candidates[j].memory.topic_ids)
                    union = topics_i | topics_j
                    topic_overlap = len(topics_i & topics_j) / len(union) if union else 0
                    emb_sim = 0.0
                    if candidates[i].memory.embedding is not None and candidates[j].memory.embedding is not None:
                        emb_sim = max(0, float(np.dot(candidates[i].memory.embedding, candidates[j].memory.embedding)))
                    redundancy = topic_w * topic_overlap + (1 - topic_w) * emb_sim
                    if redundancy > max_redundancy:
                        max_redundancy = redundancy
                mmr = lam * relevance - (1 - lam) * max_redundancy
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_i = i
            selected.append(best_i)
            remaining.remove(best_i)

        return [candidates[i] for i in selected]

    # ======================== 完整检索流程 ========================

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """
        主题路由的完整记忆检索流程

        1. 主题路由 → 2. DAG 调整 → 3. 主题内检索 → 4. 全局 Dense 融合 → 5. 跨主题扩展
        """
        query_emb = self.emb.encode(query)

        # 1. 主题路由
        scored_topics = self.route_to_topics(query, query_emb)
        if scored_topics:
            logger.info(
                f"主题路由: {[(self.topics[t].label, f'{s:.3f}') for t, s in scored_topics[:5] if t in self.topics]}"
            )

        # 2. DAG 调整
        adjusted_topics = self.adjust_topics_by_dag(scored_topics)

        # 提取评分主题：仅 top-N 高置信度主题的记忆获得 topic bonus
        scoring_topic_ids = {tid for tid, _ in adjusted_topics[:config.SCORING_TOPIC_COUNT]}

        # 3. 主题内检索
        intra_results = self.retrieve_intra_topic(query, query_emb, adjusted_topics, scoring_topic_ids)
        logger.info(f"主题内检索: {len(intra_results)} 条结果")

        # 4. 全局 Dense 融合（始终执行，作为主题路由的安全网）
        all_results = list(intra_results)
        existing_ids = {r.memory.memory_id for r in all_results}

        if self.qdrant is not None:
            global_hits = self.qdrant.search_memories(
                query_embedding=query_emb,
                top_k=top_k * 2,  # 扩大候选池，增强安全网覆盖
                topic_ids=None,  # 不过滤主题
            )
            global_count = 0
            for hit in global_hits:
                mid = hit["memory_id"]
                dense_score = hit["score"] * config.GLOBAL_DENSE_WEIGHT
                if mid in existing_ids:
                    # 已存在的记忆取较高分（确保不低于 Dense）
                    for r in all_results:
                        if r.memory.memory_id == mid:
                            r.score = max(r.score, dense_score)
                            break
                    continue
                mem = self.memories.get(mid)
                if not mem:
                    continue
                all_results.append(RetrievalResult(
                    memory=mem, score=dense_score, source_type="global_dense", matched_topics=[],
                ))
                existing_ids.add(mid)
                global_count += 1
            if global_count:
                logger.info(f"全局 Dense 融合: 补充 {global_count} 条结果")

        # 5. 跨主题扩展
        seed_ids = [tid for tid, _ in adjusted_topics]
        cross_count = 0

        if self.should_cross_topic_expand(query, intra_results):
            cross_results = self.retrieve_cross_topic(query_emb, seed_ids, existing_ids)
            all_results.extend(cross_results)
            cross_count = len(cross_results)
            logger.info(f"跨主题扩展: {cross_count} 条结果")

        all_results.sort(key=lambda r: r.score, reverse=True)
        if config.MMR_ENABLED:
            return self._mmr_select(all_results, top_k)
        return all_results[:top_k]
