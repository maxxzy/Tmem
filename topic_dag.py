"""
主题层次 DAG 构建与维护模块

核心思路：
  1. 主题向量表征：融合标签嵌入、摘要嵌入和关键词 Jaccard 计算综合相似度
  2. HAC 多粒度聚类构建初始层次树
  3. 基于相似度 + LLM 确认添加多父边，从树扩展为 DAG
  4. 支持动态更新：新主题插入、主题漂移检测与分裂、相似主题合并
"""

import logging
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import config
from models import Topic, Memory
from embedding_service import EmbeddingService
from llm_service import LLMService

logger = logging.getLogger(__name__)

# 根节点的固定 id
ROOT_TOPIC_ID = "__ROOT__"


class TopicDAG:
    """
    主题层次 DAG（有向无环图）

    支持：
    - 从主题集合构建多粒度层次
    - 多父边的添加（DAG 而非严格树）
    - 新主题的动态插入
    - 主题漂移检测与分裂
    - 相似主题的合并
    """

    def __init__(
        self,
        topics: dict[str, Topic],
        memories: dict[str, Memory],
        embedding_service: EmbeddingService,
        llm_service: LLMService,
    ):
        self.topics = topics      # topic_id -> Topic
        self.memories = memories   # memory_id -> Memory
        self.emb = embedding_service
        self.llm = llm_service
        # 确保根节点存在
        self._ensure_root()

    def _ensure_root(self):
        """确保 DAG 的虚拟根节点存在"""
        if ROOT_TOPIC_ID not in self.topics:
            root = Topic(
                topic_id=ROOT_TOPIC_ID,
                label="ROOT",
                is_virtual=True,
                depth=0,
            )
            self.topics[ROOT_TOPIC_ID] = root

    # ======================== 主题综合相似度 ========================

    def compute_topic_similarity(self, t1: Topic, t2: Topic) -> float:
        """
        计算两个主题的综合相似度
        sim(t_i, t_j) = α1·cos(e_i, e_j) + α2·cos(e^s_i, e^s_j) + α3·Jaccard(K_i, K_j)
        """
        a1, a2, a3 = config.TOPIC_SIM_WEIGHTS
        sim = 0.0

        # 标签嵌入相似度
        if t1.label_embedding is not None and t2.label_embedding is not None:
            sim += a1 * EmbeddingService.cosine_similarity(
                t1.label_embedding, t2.label_embedding
            )

        # 摘要嵌入相似度
        if t1.summary_embedding is not None and t2.summary_embedding is not None:
            sim += a2 * EmbeddingService.cosine_similarity(
                t1.summary_embedding, t2.summary_embedding
            )
        else:
            # 摘要不可用时，将其权重转移给标签嵌入
            if t1.label_embedding is not None and t2.label_embedding is not None:
                sim += a2 * EmbeddingService.cosine_similarity(
                    t1.label_embedding, t2.label_embedding
                )

        # 关键词 Jaccard 相似度
        if t1.keywords or t2.keywords:
            intersection = t1.keywords & t2.keywords
            union = t1.keywords | t2.keywords
            jaccard = len(intersection) / len(union) if union else 0.0
            sim += a3 * jaccard

        return sim

    # ======================== 构建初始层次（HAC 聚类） ========================

    def build_initial_hierarchy(self):
        """
        步骤 2：使用层次凝聚聚类（HAC）构建初始多粒度主题层次

        以 1 - sim(t_i, t_j) 为距离度量，采用 average-linkage，
        在多个距离阈值处切割树状图得到从粗到细的聚类结果。
        每层的聚类簇形成虚拟主题节点，叶子节点为原始具体主题。
        """
        # 收集所有非虚拟、非根的叶子主题
        leaf_topics = [
            t for t in self.topics.values()
            if not t.is_virtual and t.topic_id != ROOT_TOPIC_ID
        ]

        if len(leaf_topics) < 2:
            # 主题数量不足，直接挂到根节点下
            for t in leaf_topics:
                t.parent_ids = {ROOT_TOPIC_ID}
                t.depth = 1
                self.topics[ROOT_TOPIC_ID].child_ids.add(t.topic_id)
            return

        n = len(leaf_topics)
        # 计算两两距离矩阵（距离 = 1 - 相似度）
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_topic_similarity(leaf_topics[i], leaf_topics[j])
                dist = 1.0 - sim
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist

        # 转为压缩格式并执行 HAC
        condensed_dist = squareform(dist_matrix)
        Z = linkage(condensed_dist, method="average")

        # 在多个距离阈值处切割，从粗到细
        thresholds = config.HAC_DISTANCE_THRESHOLDS  # [0.7, 0.5, 0.3]

        # 先清除旧的层次关系
        for t in leaf_topics:
            t.parent_ids = set()
            t.child_ids = set()
        # 清除 ROOT 的旧子节点引用
        self.topics[ROOT_TOPIC_ID].child_ids = set()
        # 移除旧的虚拟节点（保留 ROOT）
        virtual_ids = [
            tid for tid, t in self.topics.items()
            if t.is_virtual and tid != ROOT_TOPIC_ID
        ]
        for vid in virtual_ids:
            del self.topics[vid]

        # 逐层建立虚拟节点
        # previous_clusters: {cluster_label -> [topic_id, ...]}
        previous_clusters = {i: [leaf_topics[i].topic_id] for i in range(n)}

        for layer_idx, threshold in enumerate(thresholds):
            depth = len(thresholds) - layer_idx  # 粗粒度层深度小
            cluster_labels = fcluster(Z, t=threshold, criterion="distance")
            # 按聚类标签分组
            clusters = defaultdict(list)
            for i, cl in enumerate(cluster_labels):
                clusters[cl].append(i)

            # 为每个包含多个子节点的簇创建虚拟节点
            new_clusters = {}
            for cl, member_indices in clusters.items():
                if len(member_indices) == 1:
                    # 单成员簇不需要虚拟节点
                    idx = member_indices[0]
                    new_clusters[cl] = [leaf_topics[idx].topic_id]
                    continue

                # 收集该簇包含的子主题标签
                child_labels = [leaf_topics[i].label for i in member_indices]
                virtual_label = self.llm.name_cluster(child_labels)

                # 创建虚拟节点
                virtual_topic = Topic(
                    label=virtual_label,
                    label_embedding=self.emb.encode(virtual_label),
                    is_virtual=True,
                    depth=depth,
                )
                self.topics[virtual_topic.topic_id] = virtual_topic

                # 建立父子关系：虚拟节点 -> 叶子主题
                for idx in member_indices:
                    child_id = leaf_topics[idx].topic_id
                    virtual_topic.child_ids.add(child_id)
                    self.topics[child_id].parent_ids.add(virtual_topic.topic_id)

                new_clusters[cl] = [virtual_topic.topic_id]
                logger.info(
                    f"层 {depth}: 创建虚拟主题 '{virtual_label}' "
                    f"(子主题: {child_labels})"
                )

        # 将没有父节点的顶层节点挂到 ROOT 下
        root = self.topics[ROOT_TOPIC_ID]
        for tid, topic in self.topics.items():
            if tid == ROOT_TOPIC_ID:
                continue
            if not topic.parent_ids:
                topic.parent_ids.add(ROOT_TOPIC_ID)
                topic.depth = 1
                root.child_ids.add(tid)

        # 更新所有节点的深度
        self._update_depths()

    def _update_depths(self):
        """从根节点出发 BFS 更新所有节点的深度"""
        visited = set()
        queue = [(ROOT_TOPIC_ID, 0)]
        while queue:
            tid, depth = queue.pop(0)
            if tid in visited:
                continue
            visited.add(tid)
            if tid not in self.topics:
                continue
            self.topics[tid].depth = depth
            for cid in self.topics[tid].child_ids:
                if cid not in visited:
                    queue.append((cid, depth + 1))

    # ======================== 步骤 3：添加多父边（树 → DAG） ========================

    def add_multi_parent_edges(self):
        """
        在初始层次树基础上，识别需要多归属的主题，添加跨分支的父子边。

        方法 (a)：基于相似度的候选父节点发现
          对每个主题 t_i，在非当前父节点的上层节点中找 sim > beta 的候选
        方法 (b)：LLM 辅助判断 is-a/part-of 关系，只保留确认的边

        为控制 LLM 调用量，候选对按相似度降序排列，仅取前 MAX_MULTI_PARENT_LLM_CALLS 个询问 LLM。
        """
        leaf_topics = [
            t for t in self.topics.values()
            if not t.is_virtual and t.topic_id != ROOT_TOPIC_ID
        ]
        upper_topics = [
            t for t in self.topics.values()
            if t.is_virtual and t.topic_id != ROOT_TOPIC_ID
        ]

        # (a) 收集所有通过相似度阈值的候选对
        candidates = []
        for child in leaf_topics:
            for candidate_parent in upper_topics:
                if candidate_parent.topic_id in child.parent_ids:
                    continue
                if candidate_parent.depth >= child.depth:
                    continue
                sim = self.compute_topic_similarity(child, candidate_parent)
                if sim > config.MULTI_PARENT_THRESHOLD:
                    candidates.append((child, candidate_parent, sim))

        # 按相似度降序排列，只取前 N 个询问 LLM
        candidates.sort(key=lambda x: x[2], reverse=True)
        max_calls = config.MAX_MULTI_PARENT_LLM_CALLS
        logger.info(f"多父边候选对: {len(candidates)} 对，LLM 上限: {max_calls}")

        for child, candidate_parent, sim in candidates[:max_calls]:
            # (b) LLM 确认父子关系
            if self.llm.judge_parent_child(child.label, candidate_parent.label):
                child.parent_ids.add(candidate_parent.topic_id)
                candidate_parent.child_ids.add(child.topic_id)
                logger.info(
                    f"添加多父边: '{child.label}' → '{candidate_parent.label}' (sim={sim:.3f})"
                )

    # ======================== 步骤 5：动态更新 ========================

    def insert_new_topic(self, new_topic: Topic):
        """
        将新主题动态插入 DAG

        从根节点开始贪心下降，每层选择与新主题相似度最高的节点，
        直到相似度低于阈值或到达叶子，然后经 LLM 确认后插入。
        """
        self.topics[new_topic.topic_id] = new_topic

        # 贪心下降寻找最佳插入位置
        current_id = ROOT_TOPIC_ID
        candidate_parents = []

        while True:
            current = self.topics[current_id]
            if not current.child_ids:
                break

            # 在子节点中找相似度最高的
            best_sim = 0.0
            best_child_id = None
            for cid in current.child_ids:
                child = self.topics[cid]
                sim = self.compute_topic_similarity(new_topic, child)
                if sim > best_sim:
                    best_sim = sim
                    best_child_id = cid

            # 相似度不够，停在当前层
            if best_sim < config.MULTI_PARENT_THRESHOLD:
                break

            candidate_parents.append(current_id)
            current_id = best_child_id

        # 将最终停留位置作为候选父节点
        candidate_parents.append(current_id)

        # LLM 确认每个候选父节点
        confirmed_parents = set()
        for pid in candidate_parents:
            parent = self.topics[pid]
            if pid == ROOT_TOPIC_ID or self.llm.judge_parent_child(
                new_topic.label, parent.label
            ):
                confirmed_parents.add(pid)

        # 未确认任何父节点时，挂到根节点下
        if not confirmed_parents:
            confirmed_parents = {ROOT_TOPIC_ID}

        # 建立父子关系
        new_topic.parent_ids = confirmed_parents
        for pid in confirmed_parents:
            self.topics[pid].child_ids.add(new_topic.topic_id)

        self._update_depths()
        logger.info(
            f"新主题 '{new_topic.label}' 插入 DAG, "
            f"父节点: {[self.topics[p].label for p in confirmed_parents]}"
        )

    def detect_topic_drift(self, topic: Topic, new_keywords: dict[str, float]):
        """
        主题漂移检测：当新记忆的关键词分布与原有分布差异过大时触发分裂

        使用 Jensen-Shannon 散度度量分布差异：
        JSD(P_old || P_new) = 0.5 * KL(P_old||M) + 0.5 * KL(P_new||M)
        当 JSD > delta 时触发分裂
        """
        if not topic.keyword_distribution or not new_keywords:
            return

        # 构建统一词表
        all_keys = set(topic.keyword_distribution.keys()) | set(new_keywords.keys())
        p_old = np.array([topic.keyword_distribution.get(k, 1e-10) for k in all_keys])
        p_new = np.array([new_keywords.get(k, 1e-10) for k in all_keys])

        # 归一化
        p_old = p_old / p_old.sum()
        p_new = p_new / p_new.sum()

        # 计算 JSD
        m = (p_old + p_new) / 2
        kl_old = np.sum(p_old * np.log(p_old / m))
        kl_new = np.sum(p_new * np.log(p_new / m))
        jsd = 0.5 * kl_old + 0.5 * kl_new

        if jsd > config.TOPIC_DRIFT_JSD_THRESHOLD:
            logger.info(
                f"主题 '{topic.label}' 检测到漂移 (JSD={jsd:.3f})，触发分裂"
            )
            self._split_topic(topic)

    def _split_topic(self, topic: Topic):
        """
        将一个漂移的主题分裂为两个子主题
        由 LLM 重新命名分裂后的子主题
        """
        # 将记忆按时间分为前后两半
        mem_list = sorted(
            [self.memories[mid] for mid in topic.memory_ids if mid in self.memories],
            key=lambda m: m.created_at,
        )
        mid_point = len(mem_list) // 2
        if mid_point == 0:
            return

        early_mems = mem_list[:mid_point]
        late_mems = mem_list[mid_point:]

        # LLM 为两组记忆各生成新主题名
        early_contents = [m.content for m in early_mems[:10]]
        late_contents = [m.content for m in late_mems[:10]]
        early_label = self.llm.name_cluster(early_contents)
        late_label = self.llm.name_cluster(late_contents)

        # 创建两个子主题
        for label, mems in [(early_label, early_mems), (late_label, late_mems)]:
            sub = Topic(
                label=label,
                label_embedding=self.emb.encode(label),
                keywords={kw for m in mems for kw in m.keywords},
                memory_ids={m.memory_id for m in mems},
                parent_ids=topic.parent_ids.copy(),
            )
            self.topics[sub.topic_id] = sub
            # 父节点添加子引用
            for pid in sub.parent_ids:
                if pid in self.topics:
                    self.topics[pid].child_ids.add(sub.topic_id)

        # 从父节点移除原主题的引用
        for pid in topic.parent_ids:
            if pid in self.topics:
                self.topics[pid].child_ids.discard(topic.topic_id)

        del self.topics[topic.topic_id]
        self._update_depths()
        logger.info(f"主题 '{topic.label}' 已分裂为 '{early_label}' 和 '{late_label}'")

    def check_and_merge_topics(self):
        """
        定期检查：当两个兄弟主题相似度超过阈值且记忆数量都较少时，合并为一个主题
        """
        # 收集所有兄弟对
        siblings_checked = set()
        for tid, topic in list(self.topics.items()):
            for pid in topic.parent_ids:
                parent = self.topics.get(pid)
                if not parent:
                    continue
                for sibling_id in parent.child_ids:
                    if sibling_id == tid:
                        continue
                    pair = tuple(sorted([tid, sibling_id]))
                    if pair in siblings_checked:
                        continue
                    siblings_checked.add(pair)

                    sibling = self.topics.get(sibling_id)
                    if not sibling:
                        continue

                    # 检查合并条件
                    sim = self.compute_topic_similarity(topic, sibling)
                    if (
                        sim > config.TOPIC_MERGE_THRESHOLD
                        and len(topic.memory_ids) < config.TOPIC_MERGE_MIN_MEMORY_COUNT
                        and len(sibling.memory_ids) < config.TOPIC_MERGE_MIN_MEMORY_COUNT
                    ):
                        self._merge_topics(tid, sibling_id)

    def _merge_topics(self, tid_a: str, tid_b: str):
        """将两个主题合并为一个，保留 tid_a，删除 tid_b"""
        a = self.topics[tid_a]
        b = self.topics[tid_b]

        # 合并属性
        a.keywords |= b.keywords
        a.memory_ids |= b.memory_ids
        a.child_ids |= b.child_ids

        # 更新 b 的子节点的父引用
        for cid in b.child_ids:
            if cid in self.topics:
                self.topics[cid].parent_ids.discard(tid_b)
                self.topics[cid].parent_ids.add(tid_a)

        # 更新 b 的父节点的子引用
        for pid in b.parent_ids:
            if pid in self.topics:
                self.topics[pid].child_ids.discard(tid_b)

        # 更新记忆的主题引用
        for mid in b.memory_ids:
            if mid in self.memories:
                mem = self.memories[mid]
                if tid_b in mem.topic_ids:
                    mem.topic_ids.remove(tid_b)
                if tid_a not in mem.topic_ids:
                    mem.topic_ids.append(tid_a)

        del self.topics[tid_b]
        logger.info(f"主题合并: '{b.label}' → '{a.label}'")

    # ======================== 摘要更新 ========================

    def update_topic_summary(self, topic: Topic):
        """
        当主题下新增记忆达到阈值时，由 LLM 重新生成摘要并更新摘要嵌入
        参考 Generative Agents 的 reflection 机制
        """
        if topic.memory_count_since_summary < config.SUMMARY_UPDATE_INTERVAL:
            return

        mem_contents = [
            self.memories[mid].content
            for mid in topic.memory_ids
            if mid in self.memories
        ]
        if not mem_contents:
            return

        topic.summary = self.llm.generate_topic_summary(topic.label, mem_contents)
        topic.summary_embedding = self.emb.encode(topic.summary)
        topic.memory_count_since_summary = 0
        topic.updated_at = __import__("datetime").datetime.now()
        logger.info(f"更新主题 '{topic.label}' 的摘要")

    # ======================== 辅助查询方法 ========================

    def get_ancestors(self, topic_id: str) -> set[str]:
        """获取一个主题的所有祖先节点 id"""
        ancestors = set()
        queue = list(self.topics.get(topic_id, Topic()).parent_ids)
        while queue:
            pid = queue.pop(0)
            if pid not in ancestors:
                ancestors.add(pid)
                if pid in self.topics:
                    queue.extend(self.topics[pid].parent_ids)
        return ancestors

    def get_descendants(self, topic_id: str) -> set[str]:
        """获取一个主题的所有后代节点 id"""
        descendants = set()
        queue = list(self.topics.get(topic_id, Topic()).child_ids)
        while queue:
            cid = queue.pop(0)
            if cid not in descendants:
                descendants.add(cid)
                if cid in self.topics:
                    queue.extend(self.topics[cid].child_ids)
        return descendants

    def get_siblings(self, topic_id: str) -> set[str]:
        """获取一个主题的所有兄弟节点 id"""
        siblings = set()
        topic = self.topics.get(topic_id)
        if not topic:
            return siblings
        for pid in topic.parent_ids:
            parent = self.topics.get(pid)
            if parent:
                siblings |= parent.child_ids
        siblings.discard(topic_id)
        return siblings

    # ======================== 完整构建流程 ========================

    def build(self):
        """
        完整的 DAG 构建流程：
        1. HAC 聚类建立初始层次（步骤 2）
        2. 添加多父边扩展为 DAG（步骤 3）
        3. 更新需要更新的摘要
        """
        logger.info("开始构建主题 DAG...")
        self.build_initial_hierarchy()
        self.add_multi_parent_edges()
        # 更新摘要
        for topic in self.topics.values():
            if not topic.is_virtual and topic.topic_id != ROOT_TOPIC_ID:
                self.update_topic_summary(topic)
        logger.info(f"主题 DAG 构建完成，共 {len(self.topics)} 个节点")
