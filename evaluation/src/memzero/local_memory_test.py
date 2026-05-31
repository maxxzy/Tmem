import os
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

# Use local Memory instead of API client
from mem0.memory.main import Memory

# Reuse prompts from evaluation
import sys
from pathlib import Path
# Ensure evaluation directory is importable for `prompts`
_eval_root = Path(__file__).resolve().parents[2]
if str(_eval_root) not in sys.path:
    sys.path.insert(0, str(_eval_root))
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
from mem0.configs.prompts import MACRO_TOPICS

os.environ["OPENAI_API_KEY"] = "sk-0mCxucY3RJ6nS8619rJITPRY6tARrIueT3s2HFLbsQgxHyI5"
os.environ["OPENAI_BASE_URL"] = "https://yansd666.top/v1"

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-5.2",
            "temperature": 0,
            "max_tokens": 1024,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0_topic_test",
            "host": "localhost",
            "port": 6333,
        }
    }
}

load_dotenv()


class LocalMem0:
    """
    Local evaluator that uses mem0.memory.main.Memory directly for add/search on LOCOMO dataset.
    """

    def __init__(
        self,
        output_path: str = "results.json",
        top_k: int = 10,
        filter_memories: bool = False,
        is_graph: bool = False,
        config_dict: Dict[str, Any] | None = None,
    ):
        # Construct Memory via from_config (align with test.py style)
        self.memory = Memory.from_config(config)
        self.top_k = top_k
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph

        self.openai_client = OpenAI()
        self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH if self.is_graph else ANSWER_PROMPT

    # ---------- Add phase ----------
    def delete_all_for_user(self, user_id: str):
        """Delete all memories for a given user_id using local Memory APIs."""
        try:
            # Prefer native delete_all if present
            self.memory.delete_all(user_id=user_id)
        except Exception:
            # Fallback: enumerate and delete
            items = self.memory.get_all(user_id=user_id, limit=10000)
            for it in items:
                try:
                    self.memory.delete(it["id"])  # id field expected
                except Exception:
                    pass

    def add_memory(self, user_id: str, messages: List[Dict[str, str]], metadata: Dict[str, Any]) -> None:
        """Add a batch of messages for a single user into local Memory.
        Uses infer=True to extract memories via LLM if configured in env.
        """
        # Local Memory.add signature supports messages and metadata; infer defaults to True
        self.memory.add(messages, user_id=user_id, metadata=metadata, infer=True)

    def add_memories_for_speaker(self, speaker_id: str, messages: List[Dict[str, str]], timestamp: str, desc: str):
        for i in tqdm(range(0, len(messages), 2), desc=desc):  # batch size fixed to 2 for parity
            batch = messages[i : i + 2]
            self.add_memory(speaker_id, batch, metadata={"timestamp": timestamp})

    def process_conversation_add(self, item: Dict[str, Any], idx: int):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]
        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # Cleanup old
        self.delete_all_for_user(speaker_a_user_id)
        self.delete_all_for_user(speaker_b_user_id)

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue
            date_time_key = key + "_date_time"
            timestamp = conversation.get(date_time_key, conversation.get("timestamp", ""))
            chats = conversation[key]

            messages_a: List[Dict[str, str]] = []
            messages_b_view: List[Dict[str, str]] = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages_a.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_b_view.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages_a.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_b_view.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            # Add memories for two users concurrently
            t_a = ThreadPoolExecutor(max_workers=1)
            t_b = ThreadPoolExecutor(max_workers=1)
            f1 = t_a.submit(self.add_memories_for_speaker, speaker_a_user_id, messages_a, timestamp, "Add Speaker A")
            f2 = t_b.submit(
                self.add_memories_for_speaker, speaker_b_user_id, messages_b_view, timestamp, "Add Speaker B"
            )
            f1.result()
            f2.result()

    # ---------- Search + Answer phase ----------
    def search_memory(self, user_id: str, query: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        start = time.time()
        filters = {"user_id": user_id} if self.filter_memories else None
        # Local Memory.search expected to return list of dicts with memory/metadata/score
        memories = self.memory.search(query, user_id=user_id, filters=filters, limit=self.top_k)
        end = time.time()

        if not self.is_graph:
            # Expect a dict with key "results" (same as test.py usage)
            items = memories.get("results", []) if isinstance(memories, dict) else []
            semantic_memories = [
                {
                    "memory": m.get("memory"),
                    "timestamp": (m.get("metadata") or {}).get("timestamp", ""),
                    "score": round(m.get("score", 0.0), 2),
                    "topics": (m.get("metadata") or {}).get("topics", []),
                }
                for m in items
            ]
            graph_memories = None
        else:
            # If graph enabled, local Memory may return structured graph; keeping compatible shape
            semantic_memories = [
                {
                    "memory": m.get("memory"),
                    "timestamp": (m.get("metadata") or {}).get("timestamp", ""),
                    "score": round(m.get("score", 0.0), 2),
                    "topics": (m.get("metadata") or {}).get("topics", []),
                }
                for m in (memories.get("results", []) if isinstance(memories, dict) else [])
            ]
            graph_memories = [
                {"source": r.get("source"), "relationship": r.get("relationship"), "target": r.get("target")}
                for r in (memories.get("relations", []) if isinstance(memories, dict) else [])
            ]
        return semantic_memories, graph_memories, end - start

    def answer_question(self, speaker_1_user_id: str, speaker_2_user_id: str, question: str) -> Dict[str, Any]:
        s1_mems, s1_graph, s1_time = self.search_memory(speaker_1_user_id, question)
        s2_mems, s2_graph, s2_time = self.search_memory(speaker_2_user_id, question)

        # Simple topic shape verification prints (macro-first + specifics)
        s1_topics_samples = [m.get("topics", []) for m in s1_mems[:5]]
        s2_topics_samples = [m.get("topics", []) for m in s2_mems[:5]]
        s1_macro_first_flags = [bool(t) and t[0] in MACRO_TOPICS for t in s1_topics_samples]
        s2_macro_first_flags = [bool(t) and t[0] in MACRO_TOPICS for t in s2_topics_samples]
        print(f"[TopicCheck] {speaker_1_user_id} samples: {s1_topics_samples} | macro-first: {s1_macro_first_flags}")
        print(f"[TopicCheck] {speaker_2_user_id} samples: {s2_topics_samples} | macro-first: {s2_macro_first_flags}")

        search_1 = [f"{item['timestamp']}: {item['memory']}" for item in s1_mems]
        search_2 = [f"{item['timestamp']}: {item['memory']}" for item in s2_mems]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1, indent=4),
            speaker_2_memories=json.dumps(search_2, indent=4),
            speaker_1_graph_memories=json.dumps(s1_graph, indent=4),
            speaker_2_graph_memories=json.dumps(s2_graph, indent=4),
            question=question,
        )

        t1 = time.time()
        # Fallback to Memory's LLM model if env MODEL is unset
        model_name = os.getenv("MODEL") or (
            getattr(self.memory, "config", None) and getattr(self.memory.config.llm, "config", {}).get("model")
        ) or "gpt-4o-mini"
        response = self.openai_client.chat.completions.create(
            model=model_name, messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
        )
        t2 = time.time()

        return {
            "response": response.choices[0].message.content,
            "speaker_1_memories": s1_mems,
            "speaker_2_memories": s2_mems,
            "num_speaker_1_memories": len(s1_mems),
            "num_speaker_2_memories": len(s2_mems),
            "speaker_1_memory_time": s1_time,
            "speaker_2_memory_time": s2_time,
            "speaker_1_graph_memories": s1_graph,
            "speaker_2_graph_memories": s2_graph,
            "response_time": t2 - t1,
        }

    def process_data_file(self, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Adding & Processing conversations"):
            # First add
            self.process_conversation_add(item, idx)

            # Then answer questions for this item
            qa_list = item.get("qa", [])
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            results_for_item = []
            for val in qa_list:
                question = str(val.get("question", ""))
                answer = str(val.get("answer", ""))
                category = val.get("category", -1)
                evidence = val.get("evidence", [])
                adversarial_answer = val.get("adversarial_answer", "")

                res = self.answer_question(speaker_a_user_id, speaker_b_user_id, question)
                results_for_item.append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "evidence": evidence,
                        "response": res["response"],
                        "adversarial_answer": adversarial_answer,
                        "speaker_1_memories": res["speaker_1_memories"],
                        "speaker_2_memories": res["speaker_2_memories"],
                        "num_speaker_1_memories": res["num_speaker_1_memories"],
                        "num_speaker_2_memories": res["num_speaker_2_memories"],
                        "speaker_1_memory_time": res["speaker_1_memory_time"],
                        "speaker_2_memory_time": res["speaker_2_memory_time"],
                        "speaker_1_graph_memories": res["speaker_1_graph_memories"],
                        "speaker_2_graph_memories": res["speaker_2_graph_memories"],
                        "response_time": res["response_time"],
                    }
                )

            self.results[f"conversation_{idx}"].extend(results_for_item)

            # Save after each conversation
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)

        # Final save
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Local Mem0 evaluator on LOCOMO dataset")
    parser.add_argument("--dataset", type=str, default="evaluation/dataset/locomo10.json", help="Path to LOCOMO dataset")
    parser.add_argument("--output", type=str, default="evaluation/results/local_mem0_results.json", help="Output JSON path")
    parser.add_argument("--top_k", type=int, default=30, help="Number of top memories to retrieve")
    parser.add_argument("--filter_memories", action="store_true", default=False, help="Filter by user_id in search")
    parser.add_argument("--is_graph", action="store_true", default=False, help="Use graph-based search")
    args = parser.parse_args()

    runner = LocalMem0(
        output_path=args.output,
        top_k=args.top_k,
        filter_memories=args.filter_memories,
        is_graph=args.is_graph,
    )
    runner.process_data_file(args.dataset)


if __name__ == "__main__":
    main()
