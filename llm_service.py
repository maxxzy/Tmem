"""
LLM 服务
封装对大语言模型的调用，提供主题标签生成、记忆抽取、关系判断等能力
使用 OpenAI 兼容 API
"""

import json
import logging
from openai import OpenAI

import config

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM 调用服务
    通过 OpenAI 兼容接口调用轻量 LLM 完成各类结构化生成任务
    """

    def __init__(
        self,
        model: str = config.LLM_MODEL,
        base_url: str = config.LLM_BASE_URL,
        api_key: str = config.LLM_API_KEY,
    ):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @staticmethod
    def _strip_think(text: str) -> str:
        """去除 Qwen3 等模型的思维链 <think>...</think> 块"""
        import re
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _extract_json_from_text(text: str) -> dict | list:
        """
        在文本中搜索最佳的 JSON 数组或对象。
        优先返回 dict 或包含 dict 的数组（结构化输出），
        跳过纯基本类型数组（如 ["keyword1", ...]，通常是 think 块中的片段）。
        """
        decoder = json.JSONDecoder()
        first_found = None
        for start_char in ("[", "{"):
            search_from = 0
            while search_from < len(text):
                pos = text.find(start_char, search_from)
                if pos == -1:
                    break
                try:
                    result, end_idx = decoder.raw_decode(text, pos)
                    # 优先返回 dict 或数组中包含 dict 的结构化结果
                    if isinstance(result, dict):
                        return result
                    if isinstance(result, list) and result and isinstance(result[0], dict):
                        return result
                    # 记住第一个有效 JSON 作为最终回退
                    if first_found is None:
                        first_found = result
                    search_from = end_idx
                except json.JSONDecodeError:
                    search_from = pos + 1
        if first_found is not None:
            return first_found
        raise json.JSONDecodeError("No valid JSON found in text", text[:200], 0)

    def _chat(self, system_prompt: str, user_prompt: str, extra_params: dict | None = None, *, return_raw: bool = False):
        """基础聊天调用，返回模型文本响应。return_raw=True 时返回 (stripped, raw) 元组。"""
        kwargs = dict(
            model=self.model,
            temperature=config.LLM_TEMPERATURE,
            seed=config.LLM_SEED,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if extra_params:
            kwargs.update(extra_params)
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content if response.choices else None
        raw = (content or "").strip()
        stripped = self._strip_think(raw)
        if return_raw:
            return stripped, raw
        return stripped

    def _chat_json(self, system_prompt: str, user_prompt: str, extra_params: dict | None = None) -> dict | list:
        """调用 LLM 并解析 JSON 响应，多策略容错提取"""
        stripped, raw = self._chat(
            system_prompt, user_prompt + " /no_think", extra_params,
            return_raw=True,
        )

        # 策略 1：从 think 剥离后的文本中解析 JSON（快速路径）
        if stripped:
            try:
                text = stripped
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                return json.loads(text.strip())
            except json.JSONDecodeError:
                pass

        # 策略 2：在完整原始响应（含 think 块内容）中搜索 JSON
        if raw:
            try:
                result = self._extract_json_from_text(raw)
                logger.info("从 think 块中成功提取 JSON")
                return result
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError(
            f"All JSON extraction strategies failed (raw_len={len(raw) if raw else 0})",
            raw[:200] if raw else "",
            0,
        )

    # ======================== 主题标签与关键词生成 ========================

    def generate_topic_labels(self, segment_text: str) -> list[dict]:
        """
        为一个主题段生成主题标签和关键词
        返回格式: [{"label": "Work Stress", "keywords": ["overtime", "deadline"]}, ...]
        """
        system_prompt = (
            "You are a topic analysis assistant. Your ONLY output must be a valid JSON array, "
            "with NO extra text, NO explanation, NO markdown fences.\n\n"
            "Task: Identify 1-3 topics from a dialogue segment.\n"
            "Each element must be a JSON object with exactly these fields:\n"
            '  - "label" (string, REQUIRED): a concise topic label, 2-6 words\n'
            '  - "keywords" (array of strings, REQUIRED): 3-5 representative keywords\n\n'
            "IMPORTANT: Do NOT return a plain string array like [\"topic1\", \"topic2\"]. "
            "Each element MUST be a JSON object with \"label\" and \"keywords\" fields.\n\n"
            "Example output:\n"
            '[{"label": "Work Stress", "keywords": ["overtime", "deadline", "pressure"]}, '
            '{"label": "Family Vacation", "keywords": ["trip", "beach", "summer"]}]'
        )
        user_prompt = (
            f"Dialogue segment:\n\n{segment_text}\n\n"
            "Output the JSON array now:"
        )
        try:
            result = self._chat_json(system_prompt, user_prompt)
            items = result if isinstance(result, list) else [result]
            # 规范化：确保每个元素都是 dict
            normalized = []
            for item in items:
                if isinstance(item, dict) and item.get("label"):
                    normalized.append(item)
                elif isinstance(item, str) and item.strip():
                    normalized.append({"label": item.strip(), "keywords": []})
                # 其他类型（int, None, list 等）直接跳过
            return normalized if normalized else [{"label": "Unknown Topic", "keywords": []}]
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to generate topic labels: {e}")
            return [{"label": "Unknown Topic", "keywords": []}]

    # ======================== 结构化记忆抽取 ========================

    def extract_memories(
        self, segment_text: str, topic_labels: list[str]
    ) -> list[dict]:
        """
        从主题段中抽取结构化记忆
        每条记忆必须从给定的主题标签集合中选取涉及的主题
        返回格式: [{"content": "...", "topics": ["topic1"], "keywords": [...], "importance": 0.7}, ...]
        """
        topics_str = ", ".join(f'"{t}"' for t in topic_labels)
        system_prompt = (
            "You are a memory extraction assistant. Your ONLY output must be a valid JSON array, "
            "with NO extra text, NO explanation, NO markdown fences.\n\n"
            "Task: Extract ALL key factual memories from a dialogue. "
            "Include personal facts, events, preferences, plans, opinions, and relationships. "
            "Each memory must be a complete, self-contained declarative statement. "
            "Extract as many distinct memories as possible — do not merge multiple facts into one.\n\n"
            "Each element must be a JSON object with exactly these fields:\n"
            '  - "content" (string, REQUIRED): the memory statement\n'
            '  - "topics" (array of strings): select from the provided topic labels only\n'
            '  - "keywords" (array of strings): 2-5 keywords for this memory\n'
            '  - "importance" (number): 0.0 to 1.0\n\n'
            "Example output:\n"
            '[{"content": "Alice plans to visit Paris in July", "topics": ["Travel Plans"], '
            '"keywords": ["Paris", "July", "travel"], "importance": 0.7}, '
            '{"content": "Bob prefers Italian food", "topics": ["Food Preferences"], '
            '"keywords": ["Italian", "food", "preference"], "importance": 0.5}]'
        )
        user_prompt = (
            f"Dialogue:\n{segment_text}\n\n"
            f"Available topic labels: [{topics_str}]\n\n"
            "Output the JSON array now:"
        )
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                result = self._chat_json(system_prompt, user_prompt)
                memories = result if isinstance(result, list) else [result]
                # 验证结构：至少有一条包含 content 的 dict 才算成功
                if any(isinstance(m, dict) and m.get("content") for m in memories):
                    return memories
                logger.warning(
                    f"记忆抽取第 {attempt} 次返回格式异常（非 dict 数组），"
                    f"样例: {memories[0] if memories else 'empty'}"
                )
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"记忆抽取第 {attempt} 次失败: {e}")
        logger.warning(f"记忆抽取 {max_attempts} 次均失败，返回空列表")
        return []

    # ======================== 主题摘要生成 ========================

    def generate_topic_summary(
        self, topic_label: str, memory_contents: list[str]
    ) -> str:
        """
        为主题生成 1-3 句概括性摘要
        参考 Generative Agents 的 reflection 机制
        """
        memories_text = "\n".join(f"- {m}" for m in memory_contents[:20])
        system_prompt = "You are a summarization assistant. Summarize all the key points under a topic in 1-3 sentences."
        user_prompt = (
            f'Topic: "{topic_label}"\n\nRelated memories:\n{memories_text}\n\n'
            "Generate a concise summary of this topic (1-3 sentences):"
        )
        try:
            return self._chat(system_prompt, user_prompt)
        except Exception as e:
            logger.warning(f"Failed to generate topic summary for '{topic_label}': {e}")
            return topic_label

    # ======================== 虚拟节点命名 ========================

    def name_cluster(self, child_labels: list[str]) -> str:
        """为一个聚类簇（包含多个子主题）生成父主题名称"""
        labels_str = ", ".join(f'"{l}"' for l in child_labels)
        system_prompt = "You are a taxonomy expert. Generate a higher-level umbrella topic name for a group of related sub-topics."
        user_prompt = (
            f"Sub-topic list: [{labels_str}]\n\n"
            "Provide a concise parent topic name (2-4 words):"
        )
        try:
            return self._chat(system_prompt, user_prompt).strip('"').strip("'")
        except Exception as e:
            logger.warning(f"Failed to name cluster {child_labels}: {e}")
            return " & ".join(child_labels[:2])

    # ======================== DAG 父子关系判断 ========================

    def judge_parent_child(self, child_label: str, parent_label: str) -> bool:
        """
        判断 child 是否可以被视为 parent 的子主题
        利用 LLM 的语义理解判断 is-a / part-of 关系
        参考 LLMs4OL 的思路
        """
        system_prompt = (
            "You are an ontology relation judge. Determine whether one topic can be considered "
            "a sub-topic of another (is-a or part-of relationship). Answer only yes or no."
        )
        user_prompt = f'Is "{child_label}" a sub-topic of "{parent_label}"? Answer yes or no.'
        try:
            answer = self._chat(system_prompt, user_prompt).lower()
            return "yes" in answer
        except Exception as e:
            logger.warning(f"Failed to judge parent-child '{child_label}' -> '{parent_label}': {e}")
            return False

    # ======================== 因果/语义关系判断 ========================

    def judge_association(
        self, topic_a: str, topic_b: str
    ) -> dict | None:
        """
        判断两个主题之间是否存在非层次关联关系
        返回: {"related": True, "type": "causal", "score": 0.7, "direction": "a->b"}
               或 None（无关联）
        """
        system_prompt = (
            "You are a relationship analysis assistant. Your ONLY output must be a valid JSON object, "
            "with NO extra text, NO explanation, NO markdown fences.\n\n"
            "Task: Determine whether two topics are associated.\n"
            "Association types: causal, conditional, complementary.\n\n"
            "Output must be a JSON object with exactly these fields:\n"
            '  - "related" (boolean): true if associated, false otherwise\n'
            '  - "type" (string): "causal", "conditional", or "complementary"\n'
            '  - "score" (number): 0.0 to 1.0, strength of association\n'
            '  - "direction" (string): "a->b", "b->a", or "both"\n\n'
            "Example output:\n"
            '{"related": true, "type": "causal", "score": 0.7, "direction": "a->b"}'
        )
        user_prompt = (
            f'Topic A: "{topic_a}"\nTopic B: "{topic_b}"\n\n'
            "Output the JSON object now:"
        )
        try:
            result = self._chat_json(system_prompt, user_prompt)
            if not isinstance(result, dict):
                return None
            if result.get("related"):
                # 确保 score 为数值
                score = result.get("score", 0.0)
                if not isinstance(score, (int, float)):
                    try:
                        result["score"] = float(score)
                    except (ValueError, TypeError):
                        result["score"] = 0.5
                return result
            return None
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to judge topic association: {e}")
            return None
