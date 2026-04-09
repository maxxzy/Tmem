"""
LoCoMo 数据集加载器

将 LoCoMo 的多 session 对话数据转换为 TMem 系统所需的 DialogueTurn 格式，
并提供 QA 评测数据的加载接口。

LoCoMo 数据结构：
  sample: {
    sample_id, conversation: {session_1: [{speaker, dia_id, text}, ...], ...},
    qa: [{question, answer, evidence, category}, ...]
  }
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import config
from models import DialogueTurn

logger = logging.getLogger(__name__)


class LoCoMoLoader:
    """LoCoMo 数据集加载器"""

    def __init__(self, data_path: str = config.LOCOMO_DATA_PATH):
        self.data_path = data_path
        self.samples: list[dict] = []

    def load(self) -> list[dict]:
        """加载 LoCoMo 数据集"""
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        logger.info(f"加载 LoCoMo 数据集: {len(self.samples)} 个对话样本")
        return self.samples

    def get_sample(self, sample_id: str) -> dict | None:
        """根据 sample_id 获取单个样本"""
        for s in self.samples:
            if s["sample_id"] == sample_id:
                return s
        return None

    def get_conversation_turns(
        self, sample: dict, max_sessions: int | None = None
    ) -> list[list[DialogueTurn]]:
        """
        将一个样本的多 session 对话转为 DialogueTurn 列表（按 session 分组）

        Args:
            sample: LoCoMo 样本
            max_sessions: 最多取多少个 session（None 表示全部）

        Returns:
            [[session_1 的 turns], [session_2 的 turns], ...]
        """
        conv = sample.get("conversation", {})
        speaker_a = conv.get("speaker_a", "SpeakerA")
        speaker_b = conv.get("speaker_b", "SpeakerB")

        all_sessions = []
        session_idx = 1
        base_time = datetime(2023, 5, 1)

        while True:
            session_key = f"session_{session_idx}"
            if session_key not in conv:
                break
            if max_sessions is not None and session_idx > max_sessions:
                break

            # 获取 session 时间戳
            date_key = f"{session_key}_date_time"
            session_time_str = conv.get(date_key, "")
            session_time = self._parse_datetime(session_time_str, base_time + timedelta(days=session_idx))

            turns = []
            for turn_data in conv[session_key]:
                speaker = turn_data.get("speaker", "")
                # 使用 clean_text 或 text 字段
                text = turn_data.get("text", "")
                dia_id = turn_data.get("dia_id", "")

                # 处理图片内容：附加 BLIP caption
                extra = ""
                if "blip_caption" in turn_data and turn_data.get("blip_caption"):
                    extra = f" [shares image: {turn_data['blip_caption']}]"

                turn = DialogueTurn(
                    role=speaker,
                    content=text + extra,
                    timestamp=session_time,
                    dia_id=dia_id,
                    session_id=session_key,
                )
                turns.append(turn)

            if turns:
                all_sessions.append(turns)
            session_idx += 1

        logger.info(
            f"样本 {sample['sample_id']}: "
            f"{len(all_sessions)} sessions, "
            f"{sum(len(s) for s in all_sessions)} turns"
        )
        return all_sessions

    def get_qa_pairs(self, sample: dict, categories: list[int] | None = None) -> list[dict]:
        """
        获取样本的 QA 评测对

        Args:
            sample: LoCoMo 样本
            categories: 筛选的 QA 类别（1-5），None 表示全部

        Returns:
            [{"question": str, "answer": str, "evidence": [dia_id, ...], "category": int}, ...]
        """
        qas = sample.get("qa", [])
        if categories:
            qas = [q for q in qas if q.get("category") in categories]

        result = []
        for qa in qas:
            entry = {
                "question": qa.get("question", ""),
                "evidence": qa.get("evidence", []),
                "category": qa.get("category", 0),
            }
            # category 5（对抗性问题）使用 adversarial_answer
            if qa.get("category") == 5:
                entry["answer"] = qa.get("adversarial_answer", "no information available")
            else:
                entry["answer"] = str(qa.get("answer", ""))
            result.append(entry)

        return result

    def get_all_qa_pairs(self, categories: list[int] | None = None) -> list[dict]:
        """获取全部样本的所有 QA 对（附带 sample_id）"""
        all_qas = []
        for sample in self.samples:
            qas = self.get_qa_pairs(sample, categories)
            for qa in qas:
                qa["sample_id"] = sample["sample_id"]
            all_qas.extend(qas)
        return all_qas

    @staticmethod
    def _parse_datetime(dt_str: str, default: datetime) -> datetime:
        """
        解析 LoCoMo 的时间字符串（如 "1:56 pm on 8 May, 2023"）
        失败时返回默认值
        """
        if not dt_str:
            return default
        try:
            # 尝试常见格式
            for fmt in [
                "%I:%M %p on %d %B, %Y",
                "%I:%M %p on %B %d, %Y",
                "%H:%M on %d %B, %Y",
            ]:
                try:
                    return datetime.strptime(dt_str.strip(), fmt)
                except ValueError:
                    continue
        except Exception:
            pass
        return default

    def print_dataset_stats(self):
        """打印数据集统计信息"""
        print(f"{'='*60}")
        print(f"LoCoMo 数据集统计")
        print(f"{'='*60}")
        total_turns = 0
        total_qas = 0
        for sample in self.samples:
            conv = sample.get("conversation", {})
            n_sessions = sum(1 for k in conv if k.startswith("session_") and not k.endswith("date_time"))
            n_turns = sum(
                len(conv.get(f"session_{i}", []))
                for i in range(1, 100)
                if f"session_{i}" in conv
            )
            n_qas = len(sample.get("qa", []))
            total_turns += n_turns
            total_qas += n_qas
            cats = {}
            for qa in sample.get("qa", []):
                c = qa.get("category", 0)
                cats[c] = cats.get(c, 0) + 1
            print(
                f"  {sample['sample_id']}: "
                f"{n_sessions} sessions, {n_turns} turns, {n_qas} QAs "
                f"{dict(sorted(cats.items()))}"
            )
        print(f"{'='*60}")
        print(f"总计: {len(self.samples)} 对话, {total_turns} 轮次, {total_qas} QA")
        print()
