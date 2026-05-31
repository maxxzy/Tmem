import json
import os
import re
import sys
import time

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT_DIR / "evaluation"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

load_dotenv(EVAL_DIR / ".env")
load_dotenv()

from models import DialogueTurn
from prompts import ANSWER_PROMPT
from tmem import TMem


class TMemEvaluator:
    """
    TMem evaluator for the new evaluation framework.

    This implementation follows the new framework contract:
    - ingest a LOCOMO conversation
    - answer all QA items for that conversation
    - write one results JSON consumable by evals.py

    TMem is run with two in-process instances per conversation, one for each speaker's
    perspective, so the prompt contract matches the other memory techniques.
    """

    def __init__(
        self,
        output_path: str = "results/tmem_results.json",
        top_k: int = 30,
        use_full_architecture: bool = False,
    ):
        self.output_path = output_path
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.use_full_architecture = use_full_architecture
        self.results = defaultdict(list)
        self.answer_template = Template(ANSWER_PROMPT)
        self.openai_client = OpenAI()
        self.answer_model = os.getenv("MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"

    def _build_storage_namespace(self, conversation_index: int, speaker_suffix: str) -> str:
        return f"eval_locomo_conv_{conversation_index}_{speaker_suffix}"

    def _create_tmem_instance(self, storage_namespace: str | None = None) -> TMem:
        return TMem(
            use_neo4j=self.use_full_architecture,
            use_qdrant=self.use_full_architecture,
            storage_namespace=storage_namespace,
        )

    @staticmethod
    def _parse_datetime(dt_str: str, default: datetime) -> datetime:
        if not dt_str:
            return default

        normalized = re.sub(r"\b(am|pm)\b", lambda m: m.group(1).upper(), dt_str.strip())
        for fmt in (
            "%I:%M %p on %d %B, %Y",
            "%I:%M %p on %d %B %Y",
            "%I:%M %p on %B %d, %Y",
            "%H:%M on %d %B, %Y",
            "%H:%M on %d %B %Y",
        ):
            try:
                return datetime.strptime(normalized, fmt)
            except ValueError:
                continue
        return default

    @staticmethod
    def _iter_sessions(conversation: dict):
        session_keys = []
        for key in conversation:
            match = re.fullmatch(r"session_(\d+)", key)
            if match and isinstance(conversation[key], list):
                session_keys.append((int(match.group(1)), key))

        session_keys.sort()
        base_time = datetime(2023, 5, 1)

        for session_idx, session_key in session_keys:
            session_time = TMemEvaluator._parse_datetime(
                conversation.get(f"{session_key}_date_time", ""),
                base_time + timedelta(days=session_idx),
            )
            yield session_key, conversation[session_key], session_time

    @staticmethod
    def _build_perspective_turns(
        session: list[dict],
        session_key: str,
        session_time: datetime,
        perspective_speaker: str,
    ) -> list[DialogueTurn]:
        turns = []
        for chat in session:
            speaker = chat.get("speaker", "")
            text = chat.get("text", "")
            if chat.get("blip_caption"):
                text = f"{text} [shares image: {chat['blip_caption']}]"

            turns.append(
                DialogueTurn(
                    role="user" if speaker == perspective_speaker else "assistant",
                    content=f"{speaker}: {text}",
                    timestamp=session_time,
                    dia_id=chat.get("dia_id", ""),
                    session_id=session_key,
                )
            )
        return turns

    def _ingest_conversation(self, item: dict, conversation_index: int):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_tmem = self._create_tmem_instance(
            self._build_storage_namespace(conversation_index, "speaker_a") if self.use_full_architecture else None
        )
        speaker_b_tmem = self._create_tmem_instance(
            self._build_storage_namespace(conversation_index, "speaker_b") if self.use_full_architecture else None
        )

        if self.use_full_architecture:
            speaker_a_tmem.clear_all_data()
            speaker_b_tmem.clear_all_data()

        for session_key, session, session_time in self._iter_sessions(conversation):
            speaker_a_turns = self._build_perspective_turns(session, session_key, session_time, speaker_a)
            speaker_b_turns = self._build_perspective_turns(session, session_key, session_time, speaker_b)

            new_memories_a = speaker_a_tmem.add_locomo_session(speaker_a_turns)
            new_memories_b = speaker_b_tmem.add_locomo_session(speaker_b_turns)

            for memory in new_memories_a:
                memory.created_at = session_time
            for memory in new_memories_b:
                memory.created_at = session_time

        speaker_a_tmem.build_index()
        speaker_b_tmem.build_index()

        return speaker_a, speaker_b, speaker_a_tmem, speaker_b_tmem

    @staticmethod
    def _format_retrieved_memories(results: list, tmem_client: TMem) -> list[dict]:
        formatted = []
        for result in results:
            timestamp = ""
            if getattr(result.memory, "created_at", None) is not None:
                timestamp = result.memory.created_at.isoformat()

            formatted.append(
                {
                    "memory": result.memory.content,
                    "timestamp": timestamp,
                    "score": round(result.score, 2),
                    "topics": [
                        tmem_client.topics[tid].label
                        for tid in result.memory.topic_ids
                        if tid in tmem_client.topics
                    ],
                    "matched_topics": [
                        tmem_client.topics[tid].label
                        for tid in result.matched_topics
                        if tid in tmem_client.topics
                    ],
                    "source_type": result.source_type,
                }
            )
        return formatted

    def _search_memory(self, tmem_client: TMem, query: str) -> tuple[list[dict], float]:
        start_time = time.time()
        results = tmem_client.retrieve(query, top_k=self.top_k)
        end_time = time.time()
        return self._format_retrieved_memories(results, tmem_client), end_time - start_time

    @staticmethod
    def _render_memory_lines(memories: list[dict]) -> list[str]:
        rendered = []
        for item in memories:
            timestamp = item.get("timestamp", "")
            memory = item.get("memory", "")
            rendered.append(f"{timestamp}: {memory}" if timestamp else memory)
        return rendered

    def _answer_question(
        self,
        speaker_a_name: str,
        speaker_b_name: str,
        speaker_a_tmem: TMem,
        speaker_b_tmem: TMem,
        question: str,
    ) -> dict:
        speaker_1_memories, speaker_1_memory_time = self._search_memory(speaker_a_tmem, question)
        speaker_2_memories, speaker_2_memory_time = self._search_memory(speaker_b_tmem, question)

        answer_prompt = self.answer_template.render(
            speaker_1_user_id=speaker_a_name,
            speaker_2_user_id=speaker_b_name,
            speaker_1_memories=json.dumps(self._render_memory_lines(speaker_1_memories), indent=4),
            speaker_2_memories=json.dumps(self._render_memory_lines(speaker_2_memories), indent=4),
            question=question,
        )

        response_start = time.time()
        response = self.openai_client.chat.completions.create(
            model=self.answer_model,
            messages=[{"role": "system", "content": answer_prompt}],
            temperature=0.0,
        )
        response_end = time.time()

        return {
            "response": response.choices[0].message.content,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": None,
            "speaker_2_graph_memories": None,
            "response_time": response_end - response_start,
        }

    def process_data_file(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations with TMem"):
            speaker_a_name, speaker_b_name, speaker_a_tmem, speaker_b_tmem = self._ingest_conversation(item, idx)

            results_for_item = []
            for qa_item in tqdm(item.get("qa", []), total=len(item.get("qa", [])), desc=f"Questions {idx}", leave=False):
                question = str(qa_item.get("question", ""))
                answer = str(qa_item.get("answer", ""))
                category = qa_item.get("category", -1)
                evidence = qa_item.get("evidence", [])
                adversarial_answer = qa_item.get("adversarial_answer", "")

                result = self._answer_question(
                    speaker_a_name,
                    speaker_b_name,
                    speaker_a_tmem,
                    speaker_b_tmem,
                    question,
                )

                results_for_item.append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "evidence": evidence,
                        "response": result["response"],
                        "adversarial_answer": adversarial_answer,
                        "speaker_1_memories": result["speaker_1_memories"],
                        "speaker_2_memories": result["speaker_2_memories"],
                        "num_speaker_1_memories": result["num_speaker_1_memories"],
                        "num_speaker_2_memories": result["num_speaker_2_memories"],
                        "speaker_1_memory_time": result["speaker_1_memory_time"],
                        "speaker_2_memory_time": result["speaker_2_memory_time"],
                        "speaker_1_graph_memories": result["speaker_1_graph_memories"],
                        "speaker_2_graph_memories": result["speaker_2_graph_memories"],
                        "response_time": result["response_time"],
                    }
                )

            self.results[f"conversation_{idx}"] = results_for_item

            with open(self.output_path, "w", encoding="utf-8") as file:
                json.dump(self.results, file, indent=4, ensure_ascii=False)

            speaker_a_tmem.close()
            speaker_b_tmem.close()

        with open(self.output_path, "w", encoding="utf-8") as file:
            json.dump(self.results, file, indent=4, ensure_ascii=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TMem evaluator on LOCOMO dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(EVAL_DIR / "dataset" / "locomo10.json"),
        help="Path to LOCOMO dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(EVAL_DIR / "results" / "tmem_results.json"),
        help="Output JSON path",
    )
    parser.add_argument("--top_k", type=int, default=30, help="Number of top memories to retrieve")
    parser.add_argument(
        "--full_architecture",
        action="store_true",
        help="Enable full TMem architecture with Neo4j and Qdrant",
    )
    args = parser.parse_args()

    runner = TMemEvaluator(
        output_path=args.output,
        top_k=args.top_k,
        use_full_architecture=args.full_architecture,
    )
    runner.process_data_file(args.dataset)


if __name__ == "__main__":
    main()