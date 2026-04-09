"""
TMem 测试与评估脚本

基于 LoCoMo 数据集，测试 TMem 系统的完整流程：
  1. 加载 LoCoMo 对话数据
  2. 逐 session 添加对话并抽取记忆
  3. 构建主题索引（DAG + 关联图）
  4. 用 LoCoMo QA 对进行检索评测
  5. 调用 LLM 生成回答并计算评测指标（F1、Evidence Recall）

用法：
  python run_eval.py --sample conv-26 --top-k 5

评测指标：
  - F1 Score：回答与标准答案的 token 级 F1
  - Evidence Recall：检索到的记忆是否覆盖了标注 evidence（对话轮）
  - Category-wise 分解：按 QA 类别（1-5）分别统计
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(__file__))

import config
from tmem import TMem
from locomo_loader import LoCoMoLoader
from models import DialogueTurn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("eval")


# ======================== 评测指标 ========================

def normalize_answer(s: str) -> str:
    """LoCoMo 标准答案归一化"""
    import string
    s = s.replace(",", "")
    s = s.lower()
    # 去标点
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    # 去冠词
    tokens = s.split()
    tokens = [t for t in tokens if t not in {"a", "an", "the", "and"}]
    return " ".join(tokens).strip()


def f1_score(prediction: str, ground_truth: str) -> float:
    """计算 token 级 F1"""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    from collections import Counter
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def multi_hop_f1(prediction: str, ground_truth: str) -> float:
    """多跳问题的 F1：逐子答案计算再取均值"""
    preds = [p.strip() for p in prediction.split(",")]
    gts = [g.strip() for g in ground_truth.split(",")]
    return float(np.mean([
        max([f1_score(p, g) for p in preds]) for g in gts
    ]))


def compute_qa_score(prediction: str, answer: str, category: int) -> float:
    """根据 QA 类别计算得分"""
    if category in [2, 3, 4]:
        # 单跳、时序、推理 → 直接 F1
        if category == 3:
            answer = answer.split(";")[0].strip()
        return f1_score(prediction, answer)
    elif category == 1:
        # 多跳 → 逐子答案 F1
        return multi_hop_f1(prediction, answer)
    elif category == 5:
        # 对抗性 → 是否识别"无信息"
        pred_lower = prediction.lower()
        if "no information available" in pred_lower or "not mentioned" in pred_lower:
            return 1.0
        return 0.0
    return 0.0


def compute_evidence_recall(
    retrieved_dia_ids: list[str], evidence_dia_ids: list[str]
) -> float:
    """
    计算 evidence recall：检索到的记忆覆盖了多少标注的 evidence 对话轮

    对每条 evidence dia_id（如 "D1:3"），检查是否出现在检索到的记忆的 source_dia_ids 中
    """
    if not evidence_dia_ids:
        return 1.0  # 无 evidence 标注时视为完美
    hit = sum(1 for eid in evidence_dia_ids if eid in retrieved_dia_ids)
    return hit / len(evidence_dia_ids)


# ======================== LLM 回答生成 ========================

def generate_answer_with_context(
    llm_service,
    question: str,
    context_memories: list[str],
    category: int,
) -> str:
    """
    利用检索到的记忆作为上下文，让 LLM 回答问题

    对于 category 5（对抗性），额外提示模型判断信息是否充足
    """
    context = "\n".join(f"- {m}" for m in context_memories[:10])

    system_prompt = (
        "You are a helpful assistant. Answer the question based ONLY on the provided context. "
        "If the context does not contain enough information to answer the question, "
        'respond with "no information available". '
        "Keep your answer concise and factual."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    try:
        return llm_service._chat(system_prompt, user_prompt)
    except Exception as e:
        logger.warning(f"LLM 回答生成失败: {e}")
        return "no information available"


# ======================== 主测试流程 ========================

def run_evaluation(
    sample_ids: list[str] | None = None,
    top_k: int = 5,
    max_sessions: int | None = None,
    qa_categories: list[int] | None = None,
    use_neo4j: bool = True,
    use_qdrant: bool = True,
    output_file: str = "eval_results.json",
):
    """
    完整评测流程

    Args:
        sample_ids: 要评测的样本 id 列表（None 为全部）
        top_k: 检索 top-K 条记忆
        max_sessions: 每个样本最多处理多少个 session
        qa_categories: 要评测的 QA 类别
        use_neo4j: 是否使用 Neo4j
        use_qdrant: 是否使用 Qdrant
        output_file: 结果输出文件
    """
    # 加载数据集
    loader = LoCoMoLoader()
    loader.load()
    loader.print_dataset_stats()

    all_results = []

    samples = loader.samples
    if sample_ids:
        samples = [s for s in samples if s["sample_id"] in sample_ids]

    for sample in samples:
        sid = sample["sample_id"]
        logger.info(f"\n{'='*60}\n开始评测样本: {sid}\n{'='*60}")

        # 初始化 TMem（每个样本独立一套）
        tmem = TMem(
            llm_api_key=config.LLM_API_KEY,
            use_neo4j=use_neo4j,
            use_qdrant=use_qdrant,
        )

        # 加载对话 sessions
        sessions = loader.get_conversation_turns(sample, max_sessions=max_sessions)

        # 逐 session 添加对话并抽取记忆
        t0 = time.time()
        total_memories = 0
        for i, session_turns in enumerate(sessions):
            new_mems = tmem.add_locomo_session(session_turns)
            total_memories += len(new_mems)
            logger.info(f"  Session {i+1}: {len(session_turns)} turns → {len(new_mems)} memories")

        ingest_time = time.time() - t0
        logger.info(f"记忆抽取完成: {total_memories} 条记忆, 耗时 {ingest_time:.1f}s")

        # 构建索引
        t1 = time.time()
        tmem.build_index()
        index_time = time.time() - t1
        logger.info(f"索引构建完成, 耗时 {index_time:.1f}s")

        # 打印主题 DAG
        tree_str = tmem.get_topic_tree_str()
        logger.info(f"主题 DAG:\n{tree_str}")
        stats = tmem.get_stats()
        logger.info(f"系统统计: {stats}")

        # QA 评测
        qas = loader.get_qa_pairs(sample, categories=qa_categories)
        logger.info(f"开始 QA 评测: {len(qas)} 个问题")

        sample_results = {
            "sample_id": sid,
            "stats": stats,
            "ingest_time": ingest_time,
            "index_time": index_time,
            "qa_results": [],
        }

        cat_scores = defaultdict(list)
        cat_recalls = defaultdict(list)

        for qi, qa in enumerate(qas):
            question = qa["question"]
            answer = qa["answer"]
            evidence = qa["evidence"]
            category = qa["category"]

            # 检索
            results = tmem.retrieve(question, top_k=top_k)

            # 收集检索到的记忆内容和 dia_ids
            context_memories = [r.memory.content for r in results]
            retrieved_dia_ids = []
            for r in results:
                retrieved_dia_ids.extend(r.memory.source_dia_ids)

            # Evidence recall
            ev_recall = compute_evidence_recall(retrieved_dia_ids, evidence)

            # 生成回答
            prediction = generate_answer_with_context(
                tmem.llm_service, question, context_memories, category,
            )

            # 计算得分
            qa_score = compute_qa_score(prediction, answer, category)

            cat_scores[category].append(qa_score)
            cat_recalls[category].append(ev_recall)

            qa_result = {
                "question": question,
                "answer": answer,
                "prediction": prediction,
                "category": category,
                "f1": round(qa_score, 4),
                "evidence_recall": round(ev_recall, 4),
                "num_retrieved": len(results),
                "evidence": evidence,
                "retrieved_dia_ids": list(set(retrieved_dia_ids)),
            }
            sample_results["qa_results"].append(qa_result)

            if (qi + 1) % 20 == 0:
                logger.info(f"  进度: {qi+1}/{len(qas)}")

        # 样本级汇总
        sample_summary = {}
        for cat in sorted(cat_scores.keys()):
            scores = cat_scores[cat]
            recalls = cat_recalls[cat]
            sample_summary[f"cat_{cat}_f1"] = round(float(np.mean(scores)), 4) if scores else 0
            sample_summary[f"cat_{cat}_recall"] = round(float(np.mean(recalls)), 4) if recalls else 0
            sample_summary[f"cat_{cat}_count"] = len(scores)

        all_cat_scores = [s for ss in cat_scores.values() for s in ss]
        all_cat_recalls = [r for rr in cat_recalls.values() for r in rr]
        sample_summary["overall_f1"] = round(float(np.mean(all_cat_scores)), 4) if all_cat_scores else 0
        sample_summary["overall_recall"] = round(float(np.mean(all_cat_recalls)), 4) if all_cat_recalls else 0
        sample_results["summary"] = sample_summary

        all_results.append(sample_results)

        # 打印样本摘要
        logger.info(f"\n样本 {sid} 评测结果:")
        for key, val in sample_summary.items():
            logger.info(f"  {key}: {val}")

        # 清理
        tmem.close()

    # 全局汇总
    if len(all_results) > 1:
        global_summary = defaultdict(list)
        for sr in all_results:
            for key, val in sr["summary"].items():
                if key.endswith("_f1") or key.endswith("_recall"):
                    global_summary[key].append(val)

        logger.info(f"\n{'='*60}\n全局评测结果 ({len(all_results)} 个样本):\n{'='*60}")
        for key in sorted(global_summary.keys()):
            vals = global_summary[key]
            logger.info(f"  {key}: {round(float(np.mean(vals)), 4)}")

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n评测结果已保存到: {output_file}")

    return all_results


# ======================== 快速功能测试 ========================

def run_quick_test(use_neo4j: bool = False, use_qdrant: bool = False):
    """
    不依赖 LLM 和外部数据库的快速功能测试

    验证：数据加载 → 嵌入计算 → 主题分段 → 向量检索流程
    """
    logger.info("====== 快速功能测试 ======")

    # 1. 测试 LoCoMo 加载
    loader = LoCoMoLoader()
    samples = loader.load()
    assert len(samples) > 0, "LoCoMo 数据加载失败"
    loader.print_dataset_stats()

    sample = samples[0]
    sessions = loader.get_conversation_turns(sample, max_sessions=2)
    assert len(sessions) > 0, "Session 解析失败"

    qas = loader.get_qa_pairs(sample)
    assert len(qas) > 0, "QA 解析失败"
    logger.info(f"样本 {sample['sample_id']}: {len(sessions)} sessions, {len(qas)} QA pairs")

    # 2. 测试嵌入服务
    from embedding_service import EmbeddingService
    emb = EmbeddingService()
    vec = emb.encode("Hello world")
    assert vec.shape[0] == config.EMBEDDING_DIM, f"嵌入维度错误: {vec.shape}"
    logger.info(f"嵌入服务正常，维度: {vec.shape[0]}")

    # 3. 测试对话 turn 嵌入
    first_session = sessions[0]
    for turn in first_session[:3]:
        turn.embedding = emb.encode(turn.content)
    logger.info(f"Turn 嵌入计算正常")

    # 4. 测试主题分段（不调用 LLM，仅测试边界检测）
    from topic_extractor import TopicExtractor
    from llm_service import LLMService

    # 创建一个 mock LLM（不实际调用）
    class MockLLM:
        def generate_topic_labels(self, text):
            return [{"label": "test_topic", "keywords": ["test"]}]
        def extract_memories(self, text, labels):
            return [{"content": text[:100], "topics": labels, "keywords": ["test"], "importance": 0.5}]

    mock_llm = MockLLM()
    extractor = TopicExtractor(emb, mock_llm)

    # 对所有 turn 计算嵌入
    for turn in first_session:
        if turn.embedding is None:
            turn.embedding = emb.encode(turn.content)

    boundaries = extractor.detect_topic_boundaries(first_session)
    segments = extractor.segment_dialogue(first_session)
    logger.info(f"主题分段: {len(first_session)} turns → {len(segments)} segments (boundaries: {boundaries})")

    # 5. 测试 Qdrant（如果启用）
    if use_qdrant:
        from qdrant_service import QdrantService
        qdrant = QdrantService()
        test_vec = emb.encode("test memory content")
        qdrant.upsert_memory(
            memory_id="test-001",
            embedding=test_vec,
            payload={"content": "test", "topic_ids": ["topic-1"], "keywords": ["test"]},
        )
        hits = qdrant.search_memories(test_vec, top_k=1)
        assert len(hits) > 0, "Qdrant 检索失败"
        logger.info(f"Qdrant 检索正常: score={hits[0]['score']:.4f}")

    # 6. 测试 Neo4j（如果启用）
    if use_neo4j:
        from neo4j_service import Neo4jService
        neo4j = Neo4jService()
        neo4j.upsert_topic("test-topic-1", label="Test Topic", keywords=["test"])
        topic = neo4j.get_topic("test-topic-1")
        assert topic is not None, "Neo4j 查询失败"
        logger.info(f"Neo4j 操作正常: {topic['label']}")
        neo4j.delete_topic("test-topic-1")
        neo4j.close()

    # 7. 打印 QA 样例
    print("\nQA 样例:")
    for qa in qas[:3]:
        print(f"  [{qa['category']}] Q: {qa['question']}")
        print(f"       A: {qa['answer']}")
        print(f"       Evidence: {qa['evidence'][:3]}")
        print()

    logger.info("====== 快速功能测试通过 ======")


# ======================== 入口 ========================

def parse_args():
    parser = argparse.ArgumentParser(description="TMem 评测脚本（基于 LoCoMo 数据集）")
    parser.add_argument(
        "--mode", choices=["eval", "quick_test"], default="quick_test",
        help="运行模式: eval(完整评测), quick_test(快速功能测试)",
    )
    parser.add_argument(
        "--sample", type=str, default=None,
        help="指定评测的样本 ID（如 conv-26），不指定则评测全部",
    )
    parser.add_argument("--top-k", type=int, default=5, help="检索 top-K")
    parser.add_argument("--max-sessions", type=int, default=None, help="每样本最多处理的 session 数")
    parser.add_argument(
        "--categories", type=str, default=None,
        help="评测的 QA 类别，逗号分隔（如 1,2,4）",
    )
    parser.add_argument("--no-neo4j", action="store_true", help="不使用 Neo4j")
    parser.add_argument("--no-qdrant", action="store_true", help="不使用 Qdrant")
    parser.add_argument("--output", type=str, default="eval_results.json", help="结果文件路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "quick_test":
        run_quick_test(
            use_neo4j=not args.no_neo4j,
            use_qdrant=not args.no_qdrant,
        )
    elif args.mode == "eval":
        sample_ids = [args.sample] if args.sample else None
        categories = None
        if args.categories:
            categories = [int(c) for c in args.categories.split(",")]

        run_evaluation(
            sample_ids=sample_ids,
            top_k=args.top_k,
            max_sessions=args.max_sessions,
            qa_categories=categories,
            use_neo4j=not args.no_neo4j,
            use_qdrant=not args.no_qdrant,
            output_file=args.output,
        )
