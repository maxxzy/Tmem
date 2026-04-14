"""
TMem 实验脚本

实验一：核心功能验证（构造数据）
实验二：LoCoMo P@5 / R@5 评测
实验三：消融实验（5 个变体）
实验四：因果查询跨主题召回（PPR 验证）
"""

import argparse
import copy
import json
import logging
import os
import time
import types
from collections import defaultdict
from datetime import datetime

import numpy as np

import config
from embedding_service import EmbeddingService
from locomo_loader import LoCoMoLoader
from models import DialogueTurn, Memory, RetrievalResult, TopicSegment
from tmem import TMem
from topic_extractor import TopicExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("experiments")


# ============================================================
#  共享工具函数
# ============================================================

CATEGORY_NAMES = {
    1: "多跳推理", 2: "时序问题", 3: "开放域",
    4: "单跳事实", 5: "对抗性",
}


def _log_per_category(label: str, result: dict):
    """打印 per_category 的 P@5 / R@5 明细"""
    per_cat = result.get("per_category", {})
    if not per_cat:
        return
    parts = []
    for cat in sorted(per_cat, key=lambda c: int(c)):
        cm = per_cat[cat]
        name = CATEGORY_NAMES.get(int(cat), f"Cat{cat}")
        parts.append(f"Cat{cat}({name}) P@5={cm['p5']:.4f} R@5={cm['r5']:.4f} n={cm['count']}")
    logger.info(f"  {label} per_category:")
    for p in parts:
        logger.info(f"    {p}")

def build_evidence_lookup(sample: dict) -> dict[str, str]:
    """
    构建 dia_id -> 原始对话文本 的映射表
    用于 evidence 的 embedding 匹配
    """
    lookup = {}
    conversation = sample.get("conversation", {})
    session_idx = 1
    while True:
        session_key = f"session_{session_idx}"
        if session_key not in conversation:
            break
        for turn in conversation[session_key]:
            dia_id = turn.get("dia_id", "")
            text = turn.get("text", "")
            if turn.get("blip_caption"):
                text += f" [shares image: {turn['blip_caption']}]"
            if dia_id:
                lookup[dia_id] = text
        session_idx += 1
    return lookup


def dense_retrieve(
    query: str,
    memories: dict[str, Memory],
    emb_service: EmbeddingService,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """
    纯 Dense 检索：对所有记忆做余弦相似度排序，无主题过滤
    """
    query_emb = emb_service.encode(query)
    scored = []
    for mem in memories.values():
        if mem.embedding is None:
            continue
        sim = float(np.dot(query_emb, mem.embedding))
        scored.append((mem, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        RetrievalResult(memory=mem, score=score, source_type="dense", matched_topics=[])
        for mem, score in scored[:top_k]
    ]


def match_evidence(
    retrieved: list[RetrievalResult],
    evidence_ids: list[str],
    evidence_lookup: dict[str, str],
    emb_service: EmbeddingService,
    sim_threshold: float = config.EVIDENCE_MATCH_SIM_THRESHOLD,
) -> dict[str, bool]:
    """
    对每个 evidence dia_id 判断是否被检索结果命中

    双重标准：
    1. dia_id 字符串匹配（主标准）
    2. embedding 相似度 > sim_threshold（备选标准）

    Returns:
        {dia_id: True/False, ...}
    """
    hits = {eid: False for eid in evidence_ids}

    for eid in evidence_ids:
        # 标准1：dia_id 精确匹配
        for r in retrieved:
            if eid in r.memory.source_dia_ids:
                hits[eid] = True
                break

        # 标准2：embedding 相似度匹配（仅在标准1未命中时尝试）
        if not hits[eid]:
            eid_text = evidence_lookup.get(eid, "")
            if eid_text:
                eid_emb = emb_service.encode(eid_text)
                for r in retrieved:
                    if r.memory.embedding is not None:
                        sim = float(np.dot(eid_emb, r.memory.embedding))
                        if sim >= sim_threshold:
                            hits[eid] = True
                            break
    return hits


def compute_precision_recall_at_k(
    retrieved: list[RetrievalResult],
    evidence_ids: list[str],
    evidence_lookup: dict[str, str],
    emb_service: EmbeddingService,
    k: int = 5,
) -> tuple[float, float]:
    """
    计算 P@k 和 R@k

    P@k = |匹配到至少一个 evidence 的 retrieved 记忆数| / k
    R@k = |被至少一条 retrieved 记忆匹配的 evidence 数| / |evidence 总数|
    """
    if not evidence_ids:
        return 0.0, 1.0  # 无 evidence 的 QA（如对抗性问题），recall 定义为 1

    top_k = retrieved[:k]
    hits = match_evidence(top_k, evidence_ids, evidence_lookup, emb_service)

    # Recall: 多少 evidence 被命中
    recall = sum(1 for v in hits.values() if v) / len(evidence_ids)

    # Precision: 多少 retrieved 至少匹配了一个 evidence
    relevant_count = 0
    for r in top_k:
        is_relevant = False
        for eid in evidence_ids:
            if eid in r.memory.source_dia_ids:
                is_relevant = True
                break
            eid_text = evidence_lookup.get(eid, "")
            if eid_text and r.memory.embedding is not None:
                eid_emb = emb_service.encode(eid_text)
                if float(np.dot(eid_emb, r.memory.embedding)) >= config.EVIDENCE_MATCH_SIM_THRESHOLD:
                    is_relevant = True
                    break
        if is_relevant:
            relevant_count += 1

    precision = relevant_count / k if k > 0 else 0.0
    return precision, recall


def ingest_sample(
    sample: dict,
    loader: LoCoMoLoader,
    use_neo4j: bool,
    use_qdrant: bool,
    max_sessions: int | None = None,
) -> TMem:
    """
    创建 TMem 实例，灌入一个样本的全部对话，构建索引
    """
    tmem = TMem(use_neo4j=use_neo4j, use_qdrant=use_qdrant)
    sessions = loader.get_conversation_turns(sample, max_sessions=max_sessions)

    total = 0
    for i, session_turns in enumerate(sessions):
        new_mems = tmem.add_locomo_session(session_turns)
        total += len(new_mems)
        logger.info(f"  Session {i + 1}: {len(session_turns)} turns -> {len(new_mems)} memories")

    logger.info(f"Memory extraction done: {total} memories, building index...")
    tmem.build_index()
    logger.info(f"Index built. Topics: {len([t for t in tmem.topics.values() if not t.is_virtual])}")
    return tmem


def evaluate_sample_retrieval(
    tmem: TMem,
    qas: list[dict],
    evidence_lookup: dict[str, str],
    top_k: int = 5,
    method: str = "tmem",
) -> dict:
    """
    在一个样本的全部 QA 上评测检索质量

    Args:
        method: "tmem" 使用主题路由检索, "dense" 使用纯 dense 检索
    """
    per_qa = []
    cat_metrics = defaultdict(lambda: {"p": [], "r": []})

    for qa in qas:
        question = qa["question"]
        evidence = qa.get("evidence", [])
        category = qa.get("category", 0)

        if method == "tmem":
            results = tmem.retrieve(question, top_k=top_k)
        else:
            results = dense_retrieve(question, tmem.memories, tmem.emb_service, top_k=top_k)

        p, r = compute_precision_recall_at_k(results, evidence, evidence_lookup, tmem.emb_service, k=top_k)

        per_qa.append({
            "question": question,
            "category": category,
            "p_at_k": round(p, 4),
            "r_at_k": round(r, 4),
            "num_evidence": len(evidence),
            "num_retrieved": len(results),
        })
        cat_metrics[category]["p"].append(p)
        cat_metrics[category]["r"].append(r)

    # 汇总
    all_p = [q["p_at_k"] for q in per_qa]
    all_r = [q["r_at_k"] for q in per_qa]
    per_category = {}
    for cat, vals in sorted(cat_metrics.items()):
        per_category[cat] = {
            "p5": round(np.mean(vals["p"]), 4),
            "r5": round(np.mean(vals["r"]), 4),
            "count": len(vals["p"]),
        }

    return {
        "overall_p5": round(np.mean(all_p), 4) if all_p else 0.0,
        "overall_r5": round(np.mean(all_r), 4) if all_r else 0.0,
        "per_category": per_category,
        "num_qas": len(per_qa),
        "per_qa": per_qa,
    }


def clone_tmem_state(source: TMem, use_neo4j: bool, use_qdrant: bool) -> TMem:
    """
    深拷贝 TMem 的 topics/memories 状态到新实例
    共享 embedding/LLM service（它们是无状态的）
    """
    new_tmem = TMem(use_neo4j=use_neo4j, use_qdrant=use_qdrant)
    new_tmem.topics = copy.deepcopy(source.topics)
    new_tmem.memories = copy.deepcopy(source.memories)
    new_tmem._extractor = None
    new_tmem._dag = None
    new_tmem._graph = None
    new_tmem._retriever = None
    return new_tmem


# ============================================================
#  实验一：核心功能验证（构造数据）
# ============================================================

def build_constructed_dialogue() -> list[DialogueTurn]:
    """构造包含 2 个主题的对话，用于验证主题过滤"""
    turns = [
        # Topic A: Job Seeking (7 turns)
        DialogueTurn(role="user", content="I've been thinking about switching jobs recently. The tech industry seems to have a lot of opportunities right now."),
        DialogueTurn(role="assistant", content="That's a big decision! Have you started looking at any specific companies or roles?"),
        DialogueTurn(role="user", content="Yeah, I've been preparing my resume and updating my LinkedIn profile. I also applied to a position at ByteDance last week."),
        DialogueTurn(role="assistant", content="ByteDance is a great company! How are you preparing for the interview?"),
        DialogueTurn(role="user", content="I've been practicing coding problems on LeetCode every day and reviewing system design concepts. I also prepared a portfolio of my past projects."),
        DialogueTurn(role="assistant", content="That sounds like thorough preparation. When is the interview scheduled?"),
        DialogueTurn(role="user", content="Next Thursday. I'm also planning to do a mock interview with my friend who works at Google to get some feedback on my answers."),

        # Topic B: Health (7 turns)
        DialogueTurn(role="user", content="By the way, I've been having trouble sleeping lately. I keep waking up at 3am and can't fall back asleep."),
        DialogueTurn(role="assistant", content="That sounds frustrating. Have you tried any changes to your sleep routine?"),
        DialogueTurn(role="user", content="I started taking melatonin and vitamin D supplements. I've also been preparing a calming bedtime routine with herbal tea and reading."),
        DialogueTurn(role="assistant", content="Those are good steps. Are you getting any exercise during the day?"),
        DialogueTurn(role="user", content="I've been running 5 kilometers every morning since last month. It helps with my energy during the day but hasn't fixed the sleep issue yet."),
        DialogueTurn(role="assistant", content="Running is great! Have you considered seeing a sleep specialist?"),
        DialogueTurn(role="user", content="Not yet, but I might if the insomnia continues. I've also cut caffeine after 2pm, which seems to help a little."),
    ]
    # 添加合成 dia_id
    for i, turn in enumerate(turns):
        turn.dia_id = f"SYNTH:{i + 1}"
    return turns


def run_experiment_1(use_neo4j: bool, use_qdrant: bool) -> dict:
    """实验一：核心功能验证"""
    logger.info("=" * 60)
    logger.info("Experiment 1: Core Function Verification")
    logger.info("=" * 60)

    trap_query = "What did I do to prepare for my last interview?"

    # 1. 构造对话并灌入 TMem
    turns = build_constructed_dialogue()
    tmem = TMem(use_neo4j=use_neo4j, use_qdrant=use_qdrant)
    tmem.add_dialogue(turns)
    tmem.build_index()

    # 2. 识别 Job Seeking 相关主题
    job_emb = tmem.emb_service.encode("job seeking interview preparation")
    job_topic_ids = set()
    for tid, topic in tmem.topics.items():
        if topic.is_virtual or topic.label_embedding is None:
            continue
        sim = float(np.dot(job_emb, topic.label_embedding))
        if sim > 0.4:  # 宽松阈值，把求职相关主题都纳入
            job_topic_ids.add(tid)
    logger.info(f"Job-related topics: {[tmem.topics[t].label for t in job_topic_ids]}")

    # 3. TMem 检索
    tmem_results = tmem.retrieve(trap_query, top_k=5)
    tmem_noise = []
    for r in tmem_results:
        if not any(tid in job_topic_ids for tid in r.memory.topic_ids):
            tmem_noise.append(r.memory.content)

    # 4. Dense 检索
    dense_results = dense_retrieve(trap_query, tmem.memories, tmem.emb_service, top_k=5)
    dense_noise = []
    for r in dense_results:
        if not any(tid in job_topic_ids for tid in r.memory.topic_ids):
            dense_noise.append(r.memory.content)

    # 5. 收集主题信息
    topics_info = []
    for tid, topic in tmem.topics.items():
        if topic.is_virtual:
            continue
        topics_info.append({
            "label": topic.label,
            "memory_count": len(topic.memory_ids),
            "keywords": list(topic.keywords)[:10],
        })

    result = {
        "trap_query": trap_query,
        "tmem": {
            "top5": [{"content": r.memory.content, "score": round(r.score, 4),
                       "topics": [tmem.topics.get(t, None) and tmem.topics[t].label for t in r.memory.topic_ids],
                       "source_type": r.source_type}
                     for r in tmem_results],
            "noise_count": len(tmem_noise),
            "noise_memories": tmem_noise,
        },
        "dense_baseline": {
            "top5": [{"content": r.memory.content, "score": round(r.score, 4),
                       "topics": [tmem.topics.get(t, None) and tmem.topics[t].label for t in r.memory.topic_ids]}
                     for r in dense_results],
            "noise_count": len(dense_noise),
            "noise_memories": dense_noise,
        },
        "topics_detected": topics_info,
        "total_memories": len(tmem.memories),
    }

    # 打印摘要
    logger.info(f"\n{'=' * 40}")
    logger.info(f"Trap query: {trap_query}")
    logger.info(f"TMem noise count: {len(tmem_noise)} / 5")
    logger.info(f"Dense noise count: {len(dense_noise)} / 5")
    logger.info(f"TMem top-5:")
    for r in tmem_results:
        topics = [tmem.topics[t].label for t in r.memory.topic_ids if t in tmem.topics]
        marker = " [NOISE]" if r.memory.content in tmem_noise else ""
        logger.info(f"  [{r.score:.3f}] {r.memory.content[:80]}... topics={topics}{marker}")
    logger.info(f"Dense top-5:")
    for r in dense_results:
        topics = [tmem.topics[t].label for t in r.memory.topic_ids if t in tmem.topics]
        marker = " [NOISE]" if r.memory.content in dense_noise else ""
        logger.info(f"  [{r.score:.3f}] {r.memory.content[:80]}... topics={topics}{marker}")

    tmem.close()
    return result


# ============================================================
#  实验二：LoCoMo P@5 / R@5
# ============================================================

def run_experiment_2(
    loader: LoCoMoLoader,
    use_neo4j: bool,
    use_qdrant: bool,
    top_k: int = 5,
    max_sessions: int | None = None,
    output_dir: str = "experiment_results",
) -> dict:
    """实验二：LoCoMo P@5 / R@5 评测"""
    logger.info("=" * 60)
    logger.info("Experiment 2: LoCoMo P@5 / R@5 Evaluation")
    logger.info("=" * 60)

    tmem_per_sample = {}
    dense_per_sample = {}
    exp2_path = os.path.join(output_dir, "exp2_locomo_evaluation.json")

    for sample in loader.samples:
        sid = sample["sample_id"]
        logger.info(f"\n--- Sample: {sid} ---")

        # 灌入数据
        tmem = ingest_sample(sample, loader, use_neo4j, use_qdrant, max_sessions)
        evidence_lookup = build_evidence_lookup(sample)
        qas = loader.get_qa_pairs(sample, max_sessions=max_sessions)
        logger.info(f"Evaluating {len(qas)} QA pairs...")

        # TMem 评测
        tmem_result = evaluate_sample_retrieval(tmem, qas, evidence_lookup, top_k, method="tmem")
        tmem_per_sample[sid] = tmem_result
        logger.info(f"  TMem   P@5={tmem_result['overall_p5']:.4f}  R@5={tmem_result['overall_r5']:.4f}")
        _log_per_category("TMem", tmem_result)

        # Dense baseline 评测
        dense_result = evaluate_sample_retrieval(tmem, qas, evidence_lookup, top_k, method="dense")
        dense_per_sample[sid] = dense_result
        logger.info(f"  Dense  P@5={dense_result['overall_p5']:.4f}  R@5={dense_result['overall_r5']:.4f}")
        _log_per_category("Dense", dense_result)

        tmem.close()

        # 增量保存
        save_json(_aggregate_exp2(tmem_per_sample, dense_per_sample), exp2_path)
        logger.info(f"  结果已保存至 {exp2_path}")

    return _aggregate_exp2(tmem_per_sample, dense_per_sample)


# ============================================================
#  实验三：消融实验
# ============================================================

def _fixed_window_segment(self, turns: list[DialogueTurn]) -> list[TopicSegment]:
    """固定窗口分段（stride=5），替代主题边界检测"""
    segments = []
    for i in range(0, len(turns), 5):
        chunk = turns[i:i + 5]
        if chunk:
            seg = TopicSegment(turns=chunk)
            segments.append(seg)
    logger.info(f"Fixed window segmentation: {len(turns)} turns -> {len(segments)} segments (stride=5)")
    return segments


def apply_multilabel_ablation(tmem: TMem) -> None:
    """将每条记忆的主题缩减为仅保留 1 个（与 memory embedding 最相似的主题）"""
    for mem in tmem.memories.values():
        if len(mem.topic_ids) <= 1:
            continue
        best_tid = None
        best_sim = -1.0
        for tid in mem.topic_ids:
            topic = tmem.topics.get(tid)
            if topic and topic.label_embedding is not None and mem.embedding is not None:
                sim = float(np.dot(mem.embedding, topic.label_embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_tid = tid
        if best_tid is None:
            best_tid = mem.topic_ids[0]
        # 从其他主题中移除该记忆
        for tid in mem.topic_ids:
            if tid != best_tid and tid in tmem.topics:
                tmem.topics[tid].memory_ids.discard(mem.memory_id)
        mem.topic_ids = [best_tid]


def _no_dag_adjust(self, scored_topics):
    """消融 DAG 调整：直接截断返回"""
    return scored_topics[:config.MAX_ROUTED_TOPICS]


def _no_cross_topic_expand(self, query, intra_results):
    """消融跨主题扩展：始终不做 PPR 扩展"""
    return False


def run_experiment_3(
    loader: LoCoMoLoader,
    use_neo4j: bool,
    use_qdrant: bool,
    top_k: int = 5,
    max_sessions: int | None = None,
    output_dir: str = "experiment_results",
) -> dict:
    """实验三：消融实验"""
    logger.info("=" * 60)
    logger.info("Experiment 3: Ablation Study")
    logger.info("=" * 60)

    variants = ["full", "-boundary", "-multilabel", "-dag", "-crosstopic"]
    variant_results = {v: {} for v in variants}
    exp3_path = os.path.join(output_dir, "exp3_ablation_study.json")

    for sample in loader.samples:
        sid = sample["sample_id"]
        logger.info(f"\n--- Sample: {sid} ---")

        evidence_lookup = build_evidence_lookup(sample)
        qas = loader.get_qa_pairs(sample, max_sessions=max_sessions)

        # === full: 完整 TMem ===
        logger.info(f"  [full] Ingesting...")
        tmem_full = ingest_sample(sample, loader, use_neo4j, use_qdrant, max_sessions)

        logger.info(f"  [full] Evaluating...")
        full_result = evaluate_sample_retrieval(tmem_full, qas, evidence_lookup, top_k, "tmem")
        variant_results["full"][sid] = full_result
        logger.info(f"  [full] P@5={full_result['overall_p5']:.4f}  R@5={full_result['overall_r5']:.4f}")

        # === -dag: 消融 DAG 调整 ===
        logger.info(f"  [-dag] Evaluating...")
        retriever = tmem_full.retriever  # 触发懒加载
        orig_dag_adjust = retriever.adjust_topics_by_dag
        retriever.adjust_topics_by_dag = types.MethodType(_no_dag_adjust, retriever)
        dag_result = evaluate_sample_retrieval(tmem_full, qas, evidence_lookup, top_k, "tmem")
        retriever.adjust_topics_by_dag = orig_dag_adjust  # 恢复
        variant_results["-dag"][sid] = dag_result
        logger.info(f"  [-dag] P@5={dag_result['overall_p5']:.4f}  R@5={dag_result['overall_r5']:.4f}")

        # === -crosstopic: 消融跨主题扩展 ===
        logger.info(f"  [-crosstopic] Evaluating...")
        orig_expand = retriever.should_cross_topic_expand
        retriever.should_cross_topic_expand = types.MethodType(_no_cross_topic_expand, retriever)
        cross_result = evaluate_sample_retrieval(tmem_full, qas, evidence_lookup, top_k, "tmem")
        retriever.should_cross_topic_expand = orig_expand  # 恢复
        variant_results["-crosstopic"][sid] = cross_result
        logger.info(f"  [-crosstopic] P@5={cross_result['overall_p5']:.4f}  R@5={cross_result['overall_r5']:.4f}")

        # === -multilabel: 每记忆仅保留 1 个主题 ===
        logger.info(f"  [-multilabel] Cloning and applying...")
        tmem_ml = clone_tmem_state(tmem_full, use_neo4j, use_qdrant)
        apply_multilabel_ablation(tmem_ml)
        tmem_ml.build_index()
        ml_result = evaluate_sample_retrieval(tmem_ml, qas, evidence_lookup, top_k, "tmem")
        variant_results["-multilabel"][sid] = ml_result
        logger.info(f"  [-multilabel] P@5={ml_result['overall_p5']:.4f}  R@5={ml_result['overall_r5']:.4f}")
        tmem_ml.close()

        # === -boundary: 固定窗口分段 ===
        logger.info(f"  [-boundary] Re-ingesting with fixed window...")
        orig_segment = TopicExtractor.segment_dialogue
        TopicExtractor.segment_dialogue = _fixed_window_segment
        try:
            tmem_bnd = ingest_sample(sample, loader, use_neo4j, use_qdrant, max_sessions)
            bnd_result = evaluate_sample_retrieval(tmem_bnd, qas, evidence_lookup, top_k, "tmem")
            variant_results["-boundary"][sid] = bnd_result
            logger.info(f"  [-boundary] P@5={bnd_result['overall_p5']:.4f}  R@5={bnd_result['overall_r5']:.4f}")
            tmem_bnd.close()
        finally:
            TopicExtractor.segment_dialogue = orig_segment

        tmem_full.close()

        # 增量保存
        save_json(_aggregate_exp3(variant_results, variants), exp3_path)
        logger.info(f"  结果已保存至 {exp3_path}")

    return _aggregate_exp3(variant_results, variants)


# ============================================================
#  实验四：因果查询跨主题召回
# ============================================================

def identify_cross_topic_qas(
    tmem: TMem,
    qas: list[dict],
    evidence_lookup: dict[str, str],
) -> list[dict]:
    """
    识别跨主题 QA：evidence 涉及 >= 2 个不同主题的 QA

    同时标注每个 QA 的"跨主题证据记忆 ID"（非种子主题下的证据记忆）
    """
    # 构建 dia_id -> memory_id 反向索引
    dia_to_mids = defaultdict(list)
    for mem in tmem.memories.values():
        for did in mem.source_dia_ids:
            dia_to_mids[did].append(mem.memory_id)

    cross_qas = []
    for qa in qas:
        evidence = qa.get("evidence", [])
        if not evidence:
            continue

        # 收集 evidence 对应的记忆及其主题
        evidence_mids = set()
        evidence_topics = set()
        for eid in evidence:
            for mid in dia_to_mids.get(eid, []):
                mem = tmem.memories.get(mid)
                if mem:
                    evidence_mids.add(mid)
                    evidence_topics.update(mem.topic_ids)

        # 也通过 embedding 匹配补充
        for eid in evidence:
            eid_text = evidence_lookup.get(eid, "")
            if not eid_text:
                continue
            eid_emb = tmem.emb_service.encode(eid_text)
            for mem in tmem.memories.values():
                if mem.memory_id in evidence_mids:
                    continue
                if mem.embedding is not None:
                    sim = float(np.dot(eid_emb, mem.embedding))
                    if sim >= config.EVIDENCE_MATCH_SIM_THRESHOLD:
                        evidence_mids.add(mem.memory_id)
                        evidence_topics.update(mem.topic_ids)

        # 判断是否跨主题（涉及 >= 2 个非虚拟主题）
        real_topics = {t for t in evidence_topics if t in tmem.topics and not tmem.topics[t].is_virtual}
        if len(real_topics) < 2:
            continue

        # 计算种子主题
        query_emb = tmem.emb_service.encode(qa["question"])
        scored = tmem.retriever.route_to_topics(qa["question"], query_emb)
        adjusted = tmem.retriever.adjust_topics_by_dag(scored)
        seed_tids = {tid for tid, _ in adjusted}

        # 跨主题证据 = evidence 记忆中属于非种子主题的
        cross_mids = set()
        for mid in evidence_mids:
            mem = tmem.memories.get(mid)
            if mem and not any(t in seed_tids for t in mem.topic_ids):
                cross_mids.add(mid)

        if cross_mids:
            cross_qas.append({
                **qa,
                "evidence_mids": evidence_mids,
                "evidence_topics": real_topics,
                "seed_topics": seed_tids,
                "cross_topic_mids": cross_mids,
            })

    return cross_qas


def compute_cross_topic_recall(
    retrieved: list[RetrievalResult],
    cross_topic_mids: set[str],
) -> float:
    """跨主题召回率 = 被检索到的跨主题证据记忆 / 全部跨主题证据记忆"""
    if not cross_topic_mids:
        return 1.0
    retrieved_mids = {r.memory.memory_id for r in retrieved}
    hit = len(cross_topic_mids & retrieved_mids)
    return hit / len(cross_topic_mids)


def run_experiment_4(
    loader: LoCoMoLoader,
    use_neo4j: bool,
    use_qdrant: bool,
    top_k: int = 5,
    max_sessions: int | None = None,
    output_dir: str = "experiment_results",
) -> dict:
    """实验四：跨主题召回对比（full vs -crosstopic）"""
    logger.info("=" * 60)
    logger.info("Experiment 4: Cross-Topic Recall (PPR Validation)")
    logger.info("=" * 60)

    full_recalls = []
    no_ppr_recalls = []
    total_cross_qas = 0
    exp4_path = os.path.join(output_dir, "exp4_cross_topic_recall.json")

    for sample in loader.samples:
        sid = sample["sample_id"]
        logger.info(f"\n--- Sample: {sid} ---")

        tmem = ingest_sample(sample, loader, use_neo4j, use_qdrant, max_sessions)
        evidence_lookup = build_evidence_lookup(sample)
        qas = loader.get_qa_pairs(sample, max_sessions=max_sessions)

        # 识别跨主题 QA
        cross_qas = identify_cross_topic_qas(tmem, qas, evidence_lookup)
        logger.info(f"  Found {len(cross_qas)} cross-topic QA pairs (out of {len(qas)})")
        total_cross_qas += len(cross_qas)

        if not cross_qas:
            tmem.close()
            continue

        retriever = tmem.retriever

        for cqa in cross_qas:
            question = cqa["question"]
            cross_mids = cqa["cross_topic_mids"]

            # full（有 PPR）
            full_results = tmem.retrieve(question, top_k=top_k)
            full_r = compute_cross_topic_recall(full_results, cross_mids)
            full_recalls.append(full_r)

            # -crosstopic（无 PPR）
            orig = retriever.should_cross_topic_expand
            retriever.should_cross_topic_expand = types.MethodType(_no_cross_topic_expand, retriever)
            no_ppr_results = tmem.retrieve(question, top_k=top_k)
            no_ppr_r = compute_cross_topic_recall(no_ppr_results, cross_mids)
            no_ppr_recalls.append(no_ppr_r)
            retriever.should_cross_topic_expand = orig

        tmem.close()

        # 增量保存
        save_json(_aggregate_exp4(full_recalls, no_ppr_recalls, total_cross_qas), exp4_path)
        logger.info(f"  结果已保存至 {exp4_path}")

    full_mean = round(np.mean(full_recalls), 4) if full_recalls else 0.0
    no_ppr_mean = round(np.mean(no_ppr_recalls), 4) if no_ppr_recalls else 0.0

    result = {
        "num_cross_topic_qas": total_cross_qas,
        "full": {"cross_topic_recall": full_mean},
        "-crosstopic": {"cross_topic_recall": no_ppr_mean},
        "ppr_contribution": round(full_mean - no_ppr_mean, 4),
    }

    logger.info(f"\nCross-topic QAs: {total_cross_qas}")
    logger.info(f"Full recall:      {full_mean:.4f}")
    logger.info(f"-crosstopic recall: {no_ppr_mean:.4f}")
    logger.info(f"PPR contribution: {result['ppr_contribution']:.4f}")

    return result


# ============================================================
#  实验 2/3/4 联合执行（共享样本提取，节省 LLM 调用）
# ============================================================

def _aggregate_exp2(exp2_tmem, exp2_dense):
    """汇总 exp2 结果，包含 per_category 聚合"""
    def agg(per_sample):
        all_p = [v["overall_p5"] for v in per_sample.values()]
        all_r = [v["overall_r5"] for v in per_sample.values()]
        if not all_p:
            return {"overall_p5": 0.0, "overall_r5": 0.0, "per_sample": {}, "per_category": {}}
        cat_p, cat_r, cat_n = {}, {}, {}
        for v in per_sample.values():
            for cat, cm in v.get("per_category", {}).items():
                cat_p.setdefault(cat, []).append(cm["p5"])
                cat_r.setdefault(cat, []).append(cm["r5"])
                cat_n[cat] = cat_n.get(cat, 0) + cm.get("count", 0)
        per_category = {
            cat: {"p5": round(np.mean(cat_p[cat]), 4), "r5": round(np.mean(cat_r[cat]), 4), "count": cat_n[cat]}
            for cat in cat_p
        }
        return {
            "overall_p5": round(np.mean(all_p), 4),
            "overall_r5": round(np.mean(all_r), 4),
            "per_sample": {k: {"p5": v["overall_p5"], "r5": v["overall_r5"]} for k, v in per_sample.items()},
            "per_category": per_category,
        }
    return {"tmem": agg(exp2_tmem), "dense_baseline": agg(exp2_dense)}


def _aggregate_exp3(exp3_variant_results, variants):
    """汇总 exp3 结果"""
    exp3_summary = {}
    for v in variants:
        per_s = exp3_variant_results[v]
        all_p = [r["overall_p5"] for r in per_s.values()]
        all_r = [r["overall_r5"] for r in per_s.values()]
        exp3_summary[v] = {
            "p5": round(np.mean(all_p), 4) if all_p else 0.0,
            "r5": round(np.mean(all_r), 4) if all_r else 0.0,
            "per_sample": {k: {"p5": r["overall_p5"], "r5": r["overall_r5"]} for k, r in per_s.items()},
        }
    return {"variants": exp3_summary}


def _aggregate_exp4(exp4_full_recalls, exp4_no_ppr_recalls, exp4_total_cross_qas):
    """汇总 exp4 结果"""
    full_mean = round(np.mean(exp4_full_recalls), 4) if exp4_full_recalls else 0.0
    no_ppr_mean = round(np.mean(exp4_no_ppr_recalls), 4) if exp4_no_ppr_recalls else 0.0
    return {
        "num_cross_topic_qas": exp4_total_cross_qas,
        "full": {"cross_topic_recall": full_mean},
        "-crosstopic": {"cross_topic_recall": no_ppr_mean},
        "ppr_contribution": round(full_mean - no_ppr_mean, 4),
    }


def run_combined_234(
    loader: LoCoMoLoader,
    use_neo4j: bool,
    use_qdrant: bool,
    top_k: int = 5,
    max_sessions: int | None = None,
    output_dir: str = "results",
) -> tuple[dict, dict, dict]:
    """
    联合执行实验 2/3/4，每个样本只做一次 full 提取

    每个样本完成 exp2 后立即增量保存，exp3/exp4 各阶段完成后也增量保存，
    确保后续实验崩溃时不丢失已完成实验的结果。

    Returns:
        (exp2_result, exp3_result, exp4_result)
    """
    logger.info("=" * 60)
    logger.info("Combined Experiments 2 + 3 + 4 (shared extraction)")
    logger.info("=" * 60)

    # 实验 2 结果
    exp2_tmem = {}
    exp2_dense = {}

    # 实验 3 结果
    variants = ["full", "-boundary", "-multilabel", "-dag", "-crosstopic"]
    exp3_variant_results = {v: {} for v in variants}

    # 实验 4 结果
    exp4_full_recalls = []
    exp4_no_ppr_recalls = []
    exp4_total_cross_qas = 0

    exp2_path = os.path.join(output_dir, "exp2_locomo_evaluation.json")
    exp3_path = os.path.join(output_dir, "exp3_ablation_study.json")
    exp4_path = os.path.join(output_dir, "exp4_cross_topic_recall.json")

    for sample in loader.samples:
        sid = sample["sample_id"]
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Sample: {sid}")
        logger.info(f"{'=' * 50}")

        evidence_lookup = build_evidence_lookup(sample)
        qas = loader.get_qa_pairs(sample, max_sessions=max_sessions)

        # ========== 1. Full extraction (共享) ==========
        logger.info(f"[full] Ingesting...")
        tmem_full = ingest_sample(sample, loader, use_neo4j, use_qdrant, max_sessions)

        # ========== 2. Exp2: TMem + Dense ==========
        logger.info(f"[Exp2] TMem evaluation...")
        tmem_result = evaluate_sample_retrieval(tmem_full, qas, evidence_lookup, top_k, "tmem")
        exp2_tmem[sid] = tmem_result
        logger.info(f"  TMem  P@5={tmem_result['overall_p5']:.4f}  R@5={tmem_result['overall_r5']:.4f}")
        _log_per_category("TMem", tmem_result)

        logger.info(f"[Exp2] Dense evaluation...")
        dense_result = evaluate_sample_retrieval(tmem_full, qas, evidence_lookup, top_k, "dense")
        exp2_dense[sid] = dense_result
        logger.info(f"  Dense P@5={dense_result['overall_p5']:.4f}  R@5={dense_result['overall_r5']:.4f}")
        _log_per_category("Dense", dense_result)

        # 增量保存 exp2
        save_json(_aggregate_exp2(exp2_tmem, exp2_dense), exp2_path)

        # ========== 3. Exp3: full (复用 exp2 结果) ==========
        exp3_variant_results["full"][sid] = tmem_result

        # ========== 4. Exp3: -dag ==========
        logger.info(f"[Exp3 -dag] Evaluating...")
        retriever = tmem_full.retriever
        orig_dag = retriever.adjust_topics_by_dag
        retriever.adjust_topics_by_dag = types.MethodType(_no_dag_adjust, retriever)
        dag_result = evaluate_sample_retrieval(tmem_full, qas, evidence_lookup, top_k, "tmem")
        retriever.adjust_topics_by_dag = orig_dag
        exp3_variant_results["-dag"][sid] = dag_result
        logger.info(f"  -dag  P@5={dag_result['overall_p5']:.4f}  R@5={dag_result['overall_r5']:.4f}")

        # ========== 5. Exp3: -crosstopic ==========
        logger.info(f"[Exp3 -crosstopic] Evaluating...")
        orig_expand = retriever.should_cross_topic_expand
        retriever.should_cross_topic_expand = types.MethodType(_no_cross_topic_expand, retriever)
        cross_result = evaluate_sample_retrieval(tmem_full, qas, evidence_lookup, top_k, "tmem")
        retriever.should_cross_topic_expand = orig_expand
        exp3_variant_results["-crosstopic"][sid] = cross_result
        logger.info(f"  -cross P@5={cross_result['overall_p5']:.4f}  R@5={cross_result['overall_r5']:.4f}")

        # ========== 6. Exp4: Cross-topic recall ==========
        logger.info(f"[Exp4] Identifying cross-topic QAs...")
        cross_qas = identify_cross_topic_qas(tmem_full, qas, evidence_lookup)
        logger.info(f"  Found {len(cross_qas)} cross-topic QA pairs")
        exp4_total_cross_qas += len(cross_qas)

        for cqa in cross_qas:
            cross_mids = cqa["cross_topic_mids"]

            # full
            full_r = tmem_full.retrieve(cqa["question"], top_k=top_k)
            exp4_full_recalls.append(compute_cross_topic_recall(full_r, cross_mids))

            # -crosstopic
            retriever.should_cross_topic_expand = types.MethodType(_no_cross_topic_expand, retriever)
            no_ppr_r = tmem_full.retrieve(cqa["question"], top_k=top_k)
            exp4_no_ppr_recalls.append(compute_cross_topic_recall(no_ppr_r, cross_mids))
            retriever.should_cross_topic_expand = orig_expand

        # 增量保存 exp4
        save_json(_aggregate_exp4(exp4_full_recalls, exp4_no_ppr_recalls, exp4_total_cross_qas), exp4_path)

        # ========== 7. Exp3: -multilabel ==========
        logger.info(f"[Exp3 -multilabel] Cloning and applying...")
        tmem_ml = clone_tmem_state(tmem_full, use_neo4j, use_qdrant)
        apply_multilabel_ablation(tmem_ml)
        tmem_ml.build_index()
        ml_result = evaluate_sample_retrieval(tmem_ml, qas, evidence_lookup, top_k, "tmem")
        exp3_variant_results["-multilabel"][sid] = ml_result
        logger.info(f"  -multi P@5={ml_result['overall_p5']:.4f}  R@5={ml_result['overall_r5']:.4f}")
        tmem_ml.close()

        # ========== 8. Exp3: -boundary ==========
        logger.info(f"[Exp3 -boundary] Re-ingesting with fixed window...")
        orig_segment = TopicExtractor.segment_dialogue
        TopicExtractor.segment_dialogue = _fixed_window_segment
        try:
            tmem_bnd = ingest_sample(sample, loader, use_neo4j, use_qdrant, max_sessions)
            bnd_result = evaluate_sample_retrieval(tmem_bnd, qas, evidence_lookup, top_k, "tmem")
            exp3_variant_results["-boundary"][sid] = bnd_result
            logger.info(f"  -bnd  P@5={bnd_result['overall_p5']:.4f}  R@5={bnd_result['overall_r5']:.4f}")
            tmem_bnd.close()
        finally:
            TopicExtractor.segment_dialogue = orig_segment

        # 增量保存 exp3
        save_json(_aggregate_exp3(exp3_variant_results, variants), exp3_path)

        # 清理
        tmem_full.close()

    # ========== 最终汇总 ==========
    exp2_result = _aggregate_exp2(exp2_tmem, exp2_dense)
    exp3_result = _aggregate_exp3(exp3_variant_results, variants)
    exp4_result = _aggregate_exp4(exp4_full_recalls, exp4_no_ppr_recalls, exp4_total_cross_qas)

    return exp2_result, exp3_result, exp4_result


# ============================================================
#  CLI 和主入口
# ============================================================

def save_json(data: dict, filepath: str):
    """保存结果为 JSON"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Results saved to: {filepath}")


def parse_args():
    parser = argparse.ArgumentParser(description="TMem Experiments")
    parser.add_argument(
        "--exp", type=str, default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Which experiment to run (default: all)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-K (default: 5)")
    parser.add_argument("--max-sessions", type=int, default=None, help="Max sessions per sample (for debugging)")
    parser.add_argument("--no-neo4j", action="store_true", help="Disable Neo4j")
    parser.add_argument("--no-qdrant", action="store_true", help="Disable Qdrant")
    parser.add_argument("--output-dir", type=str, default="experiment_results", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    use_neo4j = not args.no_neo4j
    use_qdrant = not args.no_qdrant
    exp = args.exp
    out_dir = args.output_dir

    summary = {"timestamp": datetime.now().isoformat(), "config": vars(args)}

    t_total = time.time()

    # 实验 1: 独立运行
    if exp in ("1", "all"):
        t0 = time.time()
        exp1_result = run_experiment_1(use_neo4j, use_qdrant)
        exp1_time = time.time() - t0
        save_json(exp1_result, os.path.join(out_dir, "exp1_core_verification.json"))
        summary["experiment_1"] = {
            "tmem_noise": exp1_result["tmem"]["noise_count"],
            "dense_noise": exp1_result["dense_baseline"]["noise_count"],
            "runtime_seconds": round(exp1_time, 1),
        }

    # 实验 2/3/4: 联合或独立
    need_locomo = exp in ("2", "3", "4", "all")
    if need_locomo:
        loader = LoCoMoLoader()
        loader.load()
        loader.print_dataset_stats()

    if exp == "all":
        t0 = time.time()
        exp2_result, exp3_result, exp4_result = run_combined_234(
            loader, use_neo4j, use_qdrant, args.top_k, args.max_sessions,
            output_dir=out_dir,
        )
        combined_time = time.time() - t0

        save_json(exp2_result, os.path.join(out_dir, "exp2_locomo_evaluation.json"))
        save_json(exp3_result, os.path.join(out_dir, "exp3_ablation_study.json"))
        save_json(exp4_result, os.path.join(out_dir, "exp4_cross_topic_recall.json"))

        summary["experiment_2"] = {
            "tmem": {
                "p5": exp2_result["tmem"]["overall_p5"],
                "r5": exp2_result["tmem"]["overall_r5"],
                "per_category": exp2_result["tmem"].get("per_category", {}),
            },
            "dense": {
                "p5": exp2_result["dense_baseline"]["overall_p5"],
                "r5": exp2_result["dense_baseline"]["overall_r5"],
                "per_category": exp2_result["dense_baseline"].get("per_category", {}),
            },
        }
        summary["experiment_3"] = {v: {"p5": d["p5"], "r5": d["r5"]} for v, d in exp3_result["variants"].items()}
        summary["experiment_4"] = exp4_result
        summary["runtime_234_seconds"] = round(combined_time, 1)

    elif exp == "2":
        t0 = time.time()
        exp2_result = run_experiment_2(loader, use_neo4j, use_qdrant, args.top_k, args.max_sessions, output_dir=out_dir)
        save_json(exp2_result, os.path.join(out_dir, "exp2_locomo_evaluation.json"))
        summary["experiment_2"] = {
            "tmem": {
                "p5": exp2_result["tmem"]["overall_p5"],
                "r5": exp2_result["tmem"]["overall_r5"],
                "per_category": exp2_result["tmem"].get("per_category", {}),
            },
            "dense": {
                "p5": exp2_result["dense_baseline"]["overall_p5"],
                "r5": exp2_result["dense_baseline"]["overall_r5"],
                "per_category": exp2_result["dense_baseline"].get("per_category", {}),
            },
            "runtime_seconds": round(time.time() - t0, 1),
        }

    elif exp == "3":
        t0 = time.time()
        exp3_result = run_experiment_3(loader, use_neo4j, use_qdrant, args.top_k, args.max_sessions, output_dir=out_dir)
        save_json(exp3_result, os.path.join(out_dir, "exp3_ablation_study.json"))
        summary["experiment_3"] = {v: {"p5": d["p5"], "r5": d["r5"]} for v, d in exp3_result["variants"].items()}
        summary["runtime_seconds"] = round(time.time() - t0, 1)

    elif exp == "4":
        t0 = time.time()
        exp4_result = run_experiment_4(loader, use_neo4j, use_qdrant, args.top_k, args.max_sessions, output_dir=out_dir)
        save_json(exp4_result, os.path.join(out_dir, "exp4_cross_topic_recall.json"))
        summary["experiment_4"] = exp4_result
        summary["runtime_seconds"] = round(time.time() - t0, 1)

    summary["total_runtime_seconds"] = round(time.time() - t_total, 1)
    save_json(summary, os.path.join(out_dir, "summary.json"))

    # 打印 summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
