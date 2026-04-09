"""
主题关联图构建与维护模块

核心思路：独立于主题 DAG，构建加权有向图表示主题间的非层次关系（因果、共现、时序等）
边来源：
  1. 记忆共现边（NPMI 度量）
  2. LLM 推断的因果/语义边
  3. 时序共现边（session 级别 NPMI）
支持：
  - 边权重统一计算与归一化
  - 剪枝与稀疏化
  - 半衰期衰减与强化更新
  - Personalized PageRank 跨主题扩展
"""

import math
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np

import config
from models import Topic, Memory, AssociationEdge
from embedding_service import EmbeddingService
from llm_service import LLMService

logger = logging.getLogger(__name__)


class TopicAssociationGraph:
    """
    主题关联图 G = (V, E, w)

    V: 所有主题节点
    E: 关联边（有向）
    w: 边权重函数 [0, 1]
    """

    def __init__(
        self,
        topics: dict[str, Topic],
        memories: dict[str, Memory],
        embedding_service: EmbeddingService,
        llm_service: LLMService,
    ):
        self.topics = topics
        self.memories = memories
        self.emb = embedding_service
        self.llm = llm_service
        # 关联边存储: (source_id, target_id) -> AssociationEdge
        self.edges: dict[tuple[str, str], AssociationEdge] = {}
        # 共现计数缓存: (tid_a, tid_b) -> 共现次数（无序对）
        self._cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
        # 记忆总数（用于计算概率）
        self._total_memories: int = 0

    # ======================== 3.1 来源一：记忆共现边 ========================

    def _ordered_pair(self, a: str, b: str) -> tuple[str, str]:
        """返回排序后的主题对，用于无序共现计数"""
        return (min(a, b), max(a, b))

    def compute_cooccurrence(self):
        """
        统计所有记忆的主题共现频次
        由于每条记忆标注了多个主题，同一记忆的不同主题天然形成共现
        """
        self._cooccurrence.clear()
        self._total_memories = len(self.memories)

        for mem in self.memories.values():
            topics = mem.topic_ids
            # 同一记忆内的所有主题两两共现
            for i in range(len(topics)):
                for j in range(i + 1, len(topics)):
                    pair = self._ordered_pair(topics[i], topics[j])
                    self._cooccurrence[pair] += 1

    def compute_npmi(self, tid_a: str, tid_b: str) -> float:
        """
        计算两个主题的归一化逐点互信息（NPMI）

        NPMI(t_i, t_j) = PMI(t_i, t_j) / (-log P(t_i, t_j))
        其中 PMI(t_i, t_j) = log(P(t_i, t_j) / (P(t_i) * P(t_j)))

        NPMI ∈ [-1, 1]，+1 完全共现，0 独立，-1 互斥
        """
        if self._total_memories == 0:
            return 0.0

        pair = self._ordered_pair(tid_a, tid_b)
        cooccur = self._cooccurrence.get(pair, 0)
        if cooccur == 0:
            return 0.0

        # 计算各概率
        topic_a = self.topics.get(tid_a)
        topic_b = self.topics.get(tid_b)
        if not topic_a or not topic_b:
            return 0.0

        n = self._total_memories
        p_a = len(topic_a.memory_ids) / n
        p_b = len(topic_b.memory_ids) / n
        p_ab = cooccur / n

        if p_a == 0 or p_b == 0 or p_ab == 0:
            return 0.0

        pmi = math.log(p_ab / (p_a * p_b))
        neg_log_p_ab = -math.log(p_ab)

        if neg_log_p_ab == 0:
            return 0.0

        return pmi / neg_log_p_ab

    def build_cooccurrence_edges(self):
        """
        根据记忆共现构建关联边
        建边条件：NPMI > tau_co 且 共现次数 >= n_min
        """
        self.compute_cooccurrence()

        for (tid_a, tid_b), count in self._cooccurrence.items():
            if count < config.MIN_COOCCURRENCE:
                continue

            npmi = self.compute_npmi(tid_a, tid_b)
            if npmi <= config.NPMI_THRESHOLD:
                continue

            # 根据条件概率的不对称性确定边方向
            topic_a = self.topics[tid_a]
            topic_b = self.topics[tid_b]
            n_a = max(len(topic_a.memory_ids), 1)
            n_b = max(len(topic_b.memory_ids), 1)
            # direction = P(b|a) - P(a|b)
            direction_score = count / n_a - count / n_b

            if direction_score > config.DIRECTION_EPSILON:
                # a → b（从 a 更容易联想到 b）
                self._add_or_update_edge(tid_a, tid_b, npmi_mem=npmi, edge_type="co")
            elif direction_score < -config.DIRECTION_EPSILON:
                # b → a
                self._add_or_update_edge(tid_b, tid_a, npmi_mem=npmi, edge_type="co")
            else:
                # 双向边
                self._add_or_update_edge(tid_a, tid_b, npmi_mem=npmi, edge_type="co")
                self._add_or_update_edge(tid_b, tid_a, npmi_mem=npmi, edge_type="co")

        logger.info(f"共现边构建完成，共 {len(self.edges)} 条边")

    # ======================== 3.1 来源二：LLM 因果/语义边 ========================

    def build_llm_causal_edges(self):
        """
        对共现数据不足但向量相似度处于中等区间的主题对，
        调用 LLM 判断是否存在因果/语义关联

        仅检查 cos(e_i, e_j) ∈ (0.3, 0.6) 的主题对，
        相似度太高已被共现覆盖或属于同一 DAG 分支，太低大概率无关
        """
        topic_list = [
            t for t in self.topics.values()
            if not t.is_virtual and t.label_embedding is not None
        ]

        checked = set()
        for i, ta in enumerate(topic_list):
            for j, tb in enumerate(topic_list):
                if i >= j:
                    continue
                pair = self._ordered_pair(ta.topic_id, tb.topic_id)
                if pair in checked:
                    continue
                checked.add(pair)

                # 已有共现边的主题对不再重复检查
                if (ta.topic_id, tb.topic_id) in self.edges or \
                   (tb.topic_id, ta.topic_id) in self.edges:
                    continue

                # 计算向量相似度，仅在中等区间调用 LLM
                sim = EmbeddingService.cosine_similarity(
                    ta.label_embedding, tb.label_embedding
                )
                if not (config.LLM_CAUSAL_SIM_LOW < sim < config.LLM_CAUSAL_SIM_HIGH):
                    continue

                result = self.llm.judge_association(ta.label, tb.label)
                if result is None:
                    continue

                score = result.get("score", 0.0)
                etype = result.get("type", "causal")
                direction = result.get("direction", "both")

                if direction == "a->b":
                    self._add_or_update_edge(
                        ta.topic_id, tb.topic_id,
                        llm_score=score, edge_type=etype,
                    )
                elif direction == "b->a":
                    self._add_or_update_edge(
                        tb.topic_id, ta.topic_id,
                        llm_score=score, edge_type=etype,
                    )
                else:
                    self._add_or_update_edge(
                        ta.topic_id, tb.topic_id,
                        llm_score=score, edge_type=etype,
                    )
                    self._add_or_update_edge(
                        tb.topic_id, ta.topic_id,
                        llm_score=score, edge_type=etype,
                    )

        logger.info(f"LLM 因果边构建完成，当前共 {len(self.edges)} 条边")

    # ======================== 边管理 ========================

    def _add_or_update_edge(
        self,
        source_id: str,
        target_id: str,
        npmi_mem: float = 0.0,
        llm_score: float = 0.0,
        npmi_temp: float = 0.0,
        edge_type: str = "co",
    ):
        """
        添加或更新一条关联边

        统一权重公式：
        w = w1 * NPMI_mem + w2 * s_llm + w3 * NPMI_temp
        """
        w1, w2, w3 = config.EDGE_WEIGHT_SOURCES
        weight = w1 * npmi_mem + w2 * llm_score + w3 * npmi_temp
        # 归一化到 [0, 1]
        weight = max(0.0, min(1.0, weight))

        key = (source_id, target_id)
        if key in self.edges:
            edge = self.edges[key]
            # 更新分项得分
            edge.npmi_mem = max(edge.npmi_mem, npmi_mem)
            edge.llm_score = max(edge.llm_score, llm_score)
            edge.npmi_temp = max(edge.npmi_temp, npmi_temp)
            # 重新计算统一权重
            edge.weight = (
                w1 * edge.npmi_mem + w2 * edge.llm_score + w3 * edge.npmi_temp
            )
            edge.weight = max(0.0, min(1.0, edge.weight))
            edge.last_updated = datetime.now()
            if edge_type != "co":
                edge.edge_type = edge_type
        else:
            self.edges[key] = AssociationEdge(
                source_id=source_id,
                target_id=target_id,
                weight=weight,
                edge_type=edge_type,
                npmi_mem=npmi_mem,
                llm_score=llm_score,
                npmi_temp=npmi_temp,
            )

    # ======================== 3.3 剪枝与稀疏化 ========================

    def prune_edges(self):
        """
        图的剪枝策略：
        1. 权重阈值剪枝：移除 w < tau_w 的边
        2. 度数约束：每个节点最多保留 top-K 条出边
        """
        # 1. 权重阈值剪枝
        to_remove = [
            key for key, edge in self.edges.items()
            if edge.weight < config.EDGE_WEIGHT_PRUNE_THRESHOLD
        ]
        for key in to_remove:
            del self.edges[key]

        # 2. 度数约束：每个节点保留权重最高的 top-K 出边
        out_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for (src, tgt) in self.edges:
            out_edges[src].append((src, tgt))

        for src, edge_keys in out_edges.items():
            if len(edge_keys) <= config.MAX_OUT_DEGREE:
                continue
            # 按权重降序排序，删除超出的边
            edge_keys.sort(key=lambda k: self.edges[k].weight, reverse=True)
            for key in edge_keys[config.MAX_OUT_DEGREE:]:
                del self.edges[key]

        logger.info(f"剪枝后剩余 {len(self.edges)} 条边")

    # ======================== 3.4 动态更新 ========================

    def incremental_update(self, new_memory: Memory):
        """
        增量更新：当新记忆写入时

        1. 更新涉及主题对的共现计数
        2. 重新计算涉及边的 NPMI
        3. 应用衰减-强化更新
        """
        self._total_memories += 1
        topic_ids = new_memory.topic_ids

        # 更新共现计数
        for i in range(len(topic_ids)):
            for j in range(i + 1, len(topic_ids)):
                pair = self._ordered_pair(topic_ids[i], topic_ids[j])
                self._cooccurrence[pair] += 1

                # 重新计算 NPMI
                npmi = self.compute_npmi(topic_ids[i], topic_ids[j])
                if npmi > config.NPMI_THRESHOLD and self._cooccurrence[pair] >= config.MIN_COOCCURRENCE:
                    # 确定方向
                    ta = self.topics.get(topic_ids[i])
                    tb = self.topics.get(topic_ids[j])
                    if ta and tb:
                        n_a = max(len(ta.memory_ids), 1)
                        n_b = max(len(tb.memory_ids), 1)
                        count = self._cooccurrence[pair]
                        dir_score = count / n_a - count / n_b

                        if dir_score > config.DIRECTION_EPSILON:
                            self._decay_and_reinforce(topic_ids[i], topic_ids[j], npmi)
                        elif dir_score < -config.DIRECTION_EPSILON:
                            self._decay_and_reinforce(topic_ids[j], topic_ids[i], npmi)
                        else:
                            self._decay_and_reinforce(topic_ids[i], topic_ids[j], npmi)
                            self._decay_and_reinforce(topic_ids[j], topic_ids[i], npmi)

    def _decay_and_reinforce(
        self, source_id: str, target_id: str, delta_w: float
    ):
        """
        衰减-强化更新公式：

        w_new = w_old * exp(-ln2 / t_half * Δt) + η * (1 - w_decayed / w_max) * Δw

        - w_old：上次存储的权重
        - Δt：距上次更新的时间间隔（天）
        - η：学习率
        - w_max：权重上限，(1 - w_decayed/w_max) 实现边际递减
        - Δw：本次新增信号强度
        """
        key = (source_id, target_id)
        now = datetime.now()

        if key in self.edges:
            edge = self.edges[key]
            # 计算时间衰减
            dt_days = (now - edge.last_updated).total_seconds() / 86400.0
            decay_factor = math.exp(-math.log(2) / config.HALF_LIFE_DAYS * dt_days)
            w_decayed = edge.weight * decay_factor

            # 强化（边际递减）
            eta = config.DECAY_REINFORCE_LR
            w_max = config.WEIGHT_MAX
            reinforcement = eta * (1.0 - w_decayed / w_max) * delta_w

            edge.weight = min(w_max, w_decayed + reinforcement)
            edge.last_updated = now
        else:
            # 新建边
            self._add_or_update_edge(source_id, target_id, npmi_mem=delta_w)

    def apply_time_decay(self):
        """
        惰性衰减：在查询或维护时对所有边按需应用时间衰减

        w_decayed = w * exp(-ln2 / t_half * Δt)
        """
        now = datetime.now()
        for edge in self.edges.values():
            dt_days = (now - edge.last_updated).total_seconds() / 86400.0
            if dt_days > 0:
                decay_factor = math.exp(
                    -math.log(2) / config.HALF_LIFE_DAYS * dt_days
                )
                edge.weight *= decay_factor
                edge.last_updated = now

    # ======================== 3.6 Personalized PageRank ========================

    def personalized_pagerank(
        self, seed_topic_ids: list[str], top_k: int = config.PPR_TOP_K
    ) -> list[tuple[str, float]]:
        """
        基于 Personalized PageRank 的跨主题扩展

        模拟海马体联想记忆的 spreading activation 过程（参考 HippoRAG）：
        从种子主题出发，沿关联路径逐步扩散，自然实现多跳推理

        PPR 迭代公式：
        p^{k+1} = α * M * p^k + (1-α) * s

        Args:
            seed_topic_ids: 种子主题 id 列表（query 直接命中的主题）
            top_k: 返回 top-K 个扩展主题

        Returns:
            [(topic_id, activation_score), ...] 按激活值降序排列
        """
        # 收集关联图中的所有节点
        node_set = set()
        for src, tgt in self.edges:
            node_set.add(src)
            node_set.add(tgt)
        # 加上种子节点（即使没有边）
        for sid in seed_topic_ids:
            node_set.add(sid)

        if not node_set:
            return []

        nodes = sorted(node_set)
        n = len(nodes)
        node_idx = {nid: i for i, nid in enumerate(nodes)}

        # 构建列归一化加权邻接矩阵 M
        # M_ij = w_decayed(t_j → t_i) / Σ_l w_decayed(t_j → t_l)
        M = np.zeros((n, n))
        now = datetime.now()
        for (src, tgt), edge in self.edges.items():
            if src in node_idx and tgt in node_idx:
                # 计算衰减后权重
                dt_days = (now - edge.last_updated).total_seconds() / 86400.0
                decay = math.exp(-math.log(2) / config.HALF_LIFE_DAYS * dt_days)
                w = edge.weight * decay
                M[node_idx[tgt]][node_idx[src]] = w  # 注意 M_ij 对应 j→i 的转移

        # 列归一化
        col_sums = M.sum(axis=0)
        col_sums[col_sums == 0] = 1.0  # 避免除零
        M = M / col_sums

        # 构建种子向量 s
        s = np.zeros(n)
        seed_count = 0
        for sid in seed_topic_ids:
            if sid in node_idx:
                s[node_idx[sid]] = 1.0
                seed_count += 1
        if seed_count > 0:
            s /= seed_count

        # PPR 幂迭代
        alpha = config.PPR_DAMPING
        p = s.copy()
        for _ in range(config.PPR_MAX_ITER):
            p_new = alpha * M @ p + (1 - alpha) * s
            if np.sum(np.abs(p_new - p)) < config.PPR_CONVERGENCE_EPSILON:
                break
            p = p_new

        # 提取结果：排除种子节点，按激活值降序
        seed_set = set(seed_topic_ids)
        results = []
        for i, nid in enumerate(nodes):
            if nid not in seed_set and p[i] > 0:
                results.append((nid, float(p[i])))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ======================== 完整构建流程 ========================

    def build(self):
        """
        完整的主题关联图构建流程：
        1. 记忆共现边
        2. LLM 因果边
        3. 剪枝
        """
        logger.info("开始构建主题关联图...")
        self.build_cooccurrence_edges()
        self.build_llm_causal_edges()
        self.prune_edges()
        logger.info(
            f"主题关联图构建完成: {len(self.edges)} 条边, "
            f"涉及 {len(set(s for s, _ in self.edges) | set(t for _, t in self.edges))} 个主题"
        )

    # ======================== 辅助方法 ========================

    def get_neighbors(self, topic_id: str) -> list[tuple[str, float, str]]:
        """获取一个主题的所有出边邻居: [(target_id, weight, edge_type), ...]"""
        result = []
        for (src, tgt), edge in self.edges.items():
            if src == topic_id:
                result.append((tgt, edge.weight, edge.edge_type))
        result.sort(key=lambda x: x[1], reverse=True)
        return result
