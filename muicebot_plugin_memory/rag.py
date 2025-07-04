from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import openai
from async_lru import alru_cache
from muicebot.models import Message
from muicebot.muice import Muice
from muicebot.templates import (
    generate_prompt_from_template as get_original_system_prompt,
)
from nonebot import logger
from nonebot_plugin_localstore import get_plugin_data_dir
from numpy import ndarray
from sqlalchemy.ext.asyncio import AsyncSession

from .config import config
from .database.models import ConversationSummary, MemoryMetric, UserProfile
from .database.repositories import (
    MemoryRepository,
    SummaryRepository,
    UserProfileRepository,
)
from .utils import (
    chat_with_model,
    format_response_to_int,
    generate_prompt_from_template,
    process_message,
)


class RAGSystem:
    _instance = None
    _initialized: bool

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.client = openai.AsyncOpenAI(
            api_key=config.memory_rag_api_key, base_url=config.memory_rag_base_url
        )

        # Lufy 记忆重要性模型权重
        self.wA = 2.76290708
        """唤醒权重"""
        self.wP = -0.2801391
        """困惑度权重"""
        self.wL = 0.44776699
        """LLM 估计的重要性"""
        self.wR1 = 1.02800192
        """最相关记忆计数"""
        self.wR2 = -0.01241566
        """第二最相关记忆计数"""

        # 初始化缓存目录
        if config.memory_embedding_cache_enabled:
            self.cache_dir = get_plugin_data_dir() / "embedding"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        self._initialized = True

    @staticmethod
    def get_instance() -> "RAGSystem":
        return RAGSystem()

    def _get_embedding_cache_path(self, text: str) -> Optional[Path]:
        """
        获取嵌入缓存文件路径

        :param text: 查询文本
        """
        if not self.cache_dir:
            return None

        # 根据文本和模型名称生成缓存键
        content = f"{config.memory_rag_embedding_model}:{text}"
        cache_key = hashlib.md5(content.encode("utf-8")).hexdigest()

        return self.cache_dir / cache_key

    def _load_embedding_from_cache(self, text: str) -> Optional[ndarray]:
        """
        从缓存文件中加载嵌入向量

        :param text: 查询文本
        """
        if not config.memory_embedding_cache_enabled:
            return None

        try:
            cache_path = self._get_embedding_cache_path(text)
            if not cache_path:
                return None

            meta_path = cache_path.with_suffix(".json")
            npy_path = cache_path.with_suffix(".npy")

            if not (meta_path.exists() and npy_path.exists()):
                return None

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if (
                isinstance(meta, dict)
                and "model" in meta
                and meta["model"] == config.memory_rag_embedding_model
                and "text_hash" in meta
                and meta["text_hash"]
                == hashlib.sha256(text.encode("utf-8")).hexdigest()
            ):
                embedding = np.load(npy_path)
                logger.debug(f"从缓存加载嵌入向量: {text[:50]}...")
                return embedding
            return None

        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            return None

    def _save_to_cache(self, text: str, embedding: ndarray) -> None:
        """
        将嵌入向量保存到缓存文件
        """
        import json

        if not config.memory_embedding_cache_enabled or not self.cache_dir:
            return

        try:
            cache_path = self._get_embedding_cache_path(text)
            if not cache_path:
                return

            meta_path = cache_path.with_suffix(".json")
            npy_path = cache_path.with_suffix(".npy")

            meta_data = {
                "model": config.memory_rag_embedding_model,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f)
            np.save(npy_path, embedding)

            logger.debug(f"嵌入向量已缓存: {text[:50]}...")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    @alru_cache(maxsize=1024)
    async def _get_embedding(self, text: str) -> ndarray:
        """
        调用 OpenAI API 兼容端口获取字符串的嵌入向量，支持离线缓存

        :param text: 要查询的字符串
        """
        logger.debug(f"正在查询文本嵌入向量: {text[:50]}...")

        # 首先尝试从磁盘缓存加载
        cached_embedding = self._load_embedding_from_cache(text)
        if cached_embedding is not None:
            return cached_embedding

        # 缓存未命中，调用 API
        start_time = perf_counter()
        try:
            response = await self.client.embeddings.create(
                model=config.memory_rag_embedding_model, input=[text]
            )
            embedding = np.array(response.data[0].embedding)

            # 保存到磁盘缓存
            self._save_to_cache(text, embedding)

            end_time = perf_counter()
            logger.debug(f"已完成查询，用时: {end_time - start_time}s")
            return embedding

        except Exception as e:
            logger.error(f"获取嵌入向量失败: {e}")
            raise

    def _cosine_similarity(self, vec1: ndarray, vec2: ndarray) -> float:
        """
        计算两个变量间的余弦相似度
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _emotion_arousal(self, text: str) -> float:
        """
        调用外部模型计算情绪唤醒度
        """
        from .roberta.inference_large import emotion_prediction

        logger.debug(f"计算文本 “{text}” 的情绪唤醒度...")
        start_time = perf_counter()

        emotion = emotion_prediction(text).cpu().numpy()
        arousal = float(emotion[1])

        end_time = perf_counter()
        logger.debug(f"已完成查询，用时: {end_time - start_time}s")
        return arousal

    def lufy_importance(self, memory_entry: MemoryMetric) -> float:
        """
        使用 Lufy 模型计算记忆重要性
        (参见 https://arxiv.org/html/2409.12524v1#S3)

        注意: 由于困惑度对模型的测量并不有效，且需要额外加载模型，计算成本比较高昂，故 perplexity 将始终设置为 0
        """
        arousal = memory_entry.arousal
        perplexity = 0
        llm_estimated_importance = memory_entry.llm_importance
        most_relevant_memory_count = memory_entry.r1_count
        most_relevant_memory_count2 = memory_entry.r2_count

        score = (
            arousal * self.wA
            + perplexity * self.wP
            + llm_estimated_importance * self.wL
            + most_relevant_memory_count * self.wR1
            + most_relevant_memory_count2 * self.wR2
        )
        session_interval = memory_entry.session_interval  # Δt

        return math.pow(math.e, -(session_interval / score))

    async def retrieval_memory(
        self,
        session: AsyncSession,
        user_message: str,
        userid: str,
        max_retrieval_items: int = 5,
        min_cos_sim: float = config.memory_cosine_similarity,
    ) -> list[tuple[MemoryMetric, float]]:
        """
        使用 Lufy 检索记忆
        (参见 https://arxiv.org/html/2409.12524v1#S4)
        """
        memory_items: list[tuple[MemoryMetric, float]] = []

        memory_metrics = await MemoryRepository.get_by_user_id(session, userid)

        for memory_item in memory_metrics:
            memory_item.session_interval += 1

            vec1 = await self._get_embedding(user_message)
            vec2 = await self._get_embedding(memory_item.content)
            cos_sim = self._cosine_similarity(vec1, vec2)

            logger.debug(f"Cos.Sim({memory_item.content}) = {cos_sim}")

            if cos_sim < min_cos_sim:
                continue

            if config.memory_lufy_importance_coefficient:
                importance = self.lufy_importance(memory_item)
            else:
                importance = 0

            memory_item.session_interval = 0
            memory_item.total_retrieval_count += 1

            score = cos_sim + config.memory_lufy_importance_coefficient * importance
            memory_items.append((memory_item, score))

        memory_items.sort(key=lambda x: x[1], reverse=True)

        if not memory_items:
            return []

        most_relevant_memory = memory_items[0][0]
        most_relevant_memory.r1_count += 1
        logger.debug(f"Top-1 Relevent Memory: {most_relevant_memory.content}")

        if len(memory_items) > 1:
            most_relevant_memory2 = memory_items[1][0]
            most_relevant_memory2.r2_count += 1
            logger.debug(f"Top-2 Relevent Memory: {most_relevant_memory2.content}")

        return memory_items[:max_retrieval_items]

    async def retrieval_memory_and_generate_prompt(
        self, session: AsyncSession, user_message: str, userid: str
    ) -> str:
        """
        检索用户的记忆并生成提示词
        """
        memory_metrics = await self.retrieval_memory(session, user_message, userid)

        user_profile = await UserProfileRepository.get_by_userid(session, userid)
        key_summary = user_profile.key_memory if user_profile else "暂无"
        user_perferences: list[str] = (
            json.loads(user_profile.preferences)
            if user_profile and user_profile.preferences
            else []
        )

        logger.debug(
            f"len(memory_metrics) = {len(memory_metrics)}; key_summary = {key_summary}"
        )

        if not memory_metrics:
            logger.warning("未找到该用户最合适的记忆")
            most_relevant_memory = "暂无"
        else:
            most_relevant_memory = memory_metrics[0][0].content

        if not any((key_summary, most_relevant_memory, user_perferences)):
            logger.warning("未找到有关此用户的更多信息")
            return ""

        return generate_prompt_from_template(
            "conversation.jinja2",
            key_summary=key_summary,
            related_memory=most_relevant_memory,
            user_perferences=";".join(user_perferences),
        )

    async def generate_memory(
        self, session: AsyncSession, message: Message
    ) -> MemoryMetric:
        """
        新建一条输入-输出对记忆并初始化心理数值
        """
        arousal = (
            self._emotion_arousal(message.message)
            if config.memory_lufy_enable_arousal
            else 0
        )

        key_memory = (
            await UserProfileRepository.get_by_userid(session, message.userid) or "暂无"
        )
        prompt = generate_prompt_from_template(
            "importance.jinja2", key_summary=key_memory, content=message.message
        )
        llm_estimated_importance = format_response_to_int(
            await chat_with_model(prompt, "你是一个有用的助手")
        )

        logger.debug(
            f"{message.message}: arousal = {arousal}; key_memory = {key_memory}; LEI = {llm_estimated_importance}"
        )

        new_memory_item = MemoryMetric(
            userid=message.userid,
            content=message.message,
            arousal=arousal,
            llm_importance=llm_estimated_importance,
            r1_count=0,
            r2_count=0,
            session_interval=0,
            total_retrieval_count=1,
        )

        return new_memory_item

    async def save_memories(self, session: AsyncSession, conversations: list[Message]):
        """
        保存当前对话中比较重要的记忆
        """
        logger.debug(f"Total {len(conversations)} conversations.")
        memory_scores: list[tuple[MemoryMetric, float]] = []

        for conversation in conversations:
            new_memory = await self.generate_memory(session, conversation)
            importance = self.lufy_importance(new_memory)

            memory_scores.append((new_memory, importance))

        memory_scores.sort(key=lambda x: x[1], reverse=True)

        # retain top-10%
        save_memories_len = int(len(memory_scores) * config.memory_retain_proportion)
        save_memories = [item[0] for item in memory_scores[: save_memories_len + 1]]

        logger.debug(f"{len(save_memories)} conversations will be saved.")

        for save_memory in save_memories:
            await MemoryRepository.add(session, save_memory, flush=False)

        await session.flush()

    async def summary_conversations(
        self, session: AsyncSession, conversations: list[Message]
    ):
        """
        总结对话
        """
        assert len(conversations), "必须拥有至少一轮对话才可以保存!"

        logger.debug("正在总结结束的对话...")
        userid = conversations[0].userid

        system_prompt_template = Muice.get_instance().template
        if system_prompt_template:
            original_system_prompt = get_original_system_prompt(
                system_prompt_template, userid
            )
        else:
            original_system_prompt = "暂无"

        history = [
            (conversation.message, conversation.respond)
            for conversation in conversations
        ]

        logger.debug("正调用 LLM 获取对话总结...")
        prompt = generate_prompt_from_template(
            "conversation_summary.jinja2",
            original_system_prompt=original_system_prompt,
            history=history,
        )
        summary = process_message(await chat_with_model(prompt, "你是一个有用的AI助手"))
        logger.debug(f"总结结果: {summary}")

        await SummaryRepository.add(
            session, ConversationSummary(content=summary, userid=userid)
        )

    async def key_summary(self, session: AsyncSession, userid: str):
        """
        更新关键总结
        """
        logger.debug(f"正在进行 {userid} 的关键总结...")

        user_profile = await UserProfileRepository.get_by_userid(session, userid)
        recent_conversation_summary = (
            await SummaryRepository.query_recent_summary(session, userid, limit=1)
        )[0]
        system = generate_prompt_from_template("key_summary.jinja2")
        prompt = f"关键摘要: {user_profile.key_memory if user_profile else '暂无'}\n\n"
        f"最近一次对话的摘要: {recent_conversation_summary.content}"
        key_memory = process_message(await chat_with_model(prompt, system))
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"key_memory: {key_memory}")

        if user_profile:
            user_profile.key_memory = key_memory
        else:
            await UserProfileRepository.add(
                session,
                UserProfile(
                    user_id=userid,
                    key_memory=key_memory,
                ),
            )

    async def mannual_record_memory(
        self, session: AsyncSession, userid: str, content: str, importance_score: int
    ):
        """
        保存 AI 的手动记忆内容
        """
        if not config.memory_lufy_importance_coefficient:
            arousal = 0.0
        else:
            arousal = self._emotion_arousal(content)

        new_memory_item = MemoryMetric(
            content=content,
            userid=userid,
            arousal=arousal,
            llm_importance=importance_score,
            r1_count=0,
            r2_count=0,
        )

        await MemoryRepository.add(session, new_memory_item)

    async def mannual_record_user_preferences(
        self, session: AsyncSession, userid: str, content: str
    ):
        """
        保存 AI 对用户的回答偏好
        """
        user_profile = await UserProfileRepository.get_by_userid(session, userid)
        if user_profile:
            user_perferences: list[str] = json.loads(user_profile.preferences)
            user_perferences.append(content)
            user_profile.preferences = json.dumps(user_perferences, ensure_ascii=False)
        else:
            await UserProfileRepository.add(
                session,
                UserProfile(
                    user_id=userid,
                    key_memory="暂无",
                    preferences=json.dumps([content], ensure_ascii=False),
                ),
            )
