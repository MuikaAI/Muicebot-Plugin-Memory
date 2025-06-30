from time import perf_counter
from typing import Optional

from muicebot.llm import ModelRequest
from muicebot.models import Message
from muicebot.plugin import PluginMetadata
from muicebot.plugin.func_call import on_function_call
from muicebot.plugin.hook import on_before_completion, on_finish_chat
from nonebot import get_driver, logger
from nonebot.adapters import Event
from nonebot_plugin_orm import get_session
from pydantic import BaseModel, Field

from .config import Config, config
from .rag import RAGSystem
from .scheduler import Scheduler

__plugin_meta__ = PluginMetadata(
    name="Muicebot 记忆插件",
    description="基于 LUFY 的 Muicebot RAG 记忆插件",
    usage="直接使用",
    config=Config,
)

driver = get_driver()


@driver.on_startup
def check_if_load_model():
    if not config.memory_lufy_enable_arousal:
        return
    logger.info("Loading Roberta Model...")
    from .roberta.inference_large import load_model

    load_model()


rag_system = RAGSystem()
_scheduler = Scheduler()


class RecordMemoryParams(BaseModel):
    memory: str = Field(description="记忆内容")
    importance_score: int = Field(
        description="重要性分数(0-10范围内的整数，越高的分数表示记忆越重要)"
    )


class RecordPreferenceParam(BaseModel):
    preference: str = Field(description="记忆内容，不多于50字", max_length=50)


class QueryMemoryParams(BaseModel):
    memory: str = Field(
        description="要查询的记忆内容（尽量以用户视角的陈述句作为查询内容）"
    )
    max_query_items: Optional[int] = Field(description="(可选)最大查询数量", default=5)


@on_function_call(
    description="手动添加对用户的记忆（日常回答相关）", params=RecordMemoryParams
)
async def record_memory(memory: str, importance_score: int, event: Event) -> str:
    userid = event.get_user_id()
    logger.info(f"AI 请求手动保存记忆: {memory}({userid})")

    session = get_session()
    async with session.begin():
        await rag_system.mannual_record_memory(
            session, userid, memory, importance_score
        )

    logger.info("已成功执行了请求")
    return "成功"


@on_function_call(
    description="记录用户的关键记忆（包含用户回答偏好，用户个人信息等），这一部分的记忆会被附加到系统提示中",
    params=RecordPreferenceParam,
)
async def record_user_perferences(preference: str, event: Event) -> str:
    userid = event.get_user_id()
    logger.info(f"AI 请求手动保存用户偏好: {preference}({userid})")

    session = get_session()
    async with session.begin():
        await rag_system.mannual_record_user_preferences(session, userid, preference)

    logger.info("已成功执行了请求")
    return "成功"


@on_function_call(
    description="以 RAG 形式手动查询记忆，返回由记忆内容和余弦相关性分数组成的元组列表",
    params=QueryMemoryParams,
)
async def query_memory(
    event: Event, memory: str, max_query_items: Optional[int] = 5
) -> str:
    logger.info(f"AI 请求手动查询记忆: {memory}")

    session = get_session()
    async with session.begin():
        memorys = await rag_system.retrieval_memory(
            session,
            memory,
            event.get_user_id(),
            max_retrieval_items=max_query_items or 5,
            min_cos_sim=0,
        )

    results: list[tuple[str, float]] = []

    for memory_metric, score in memorys:
        results.append((memory_metric.content, score))

    return str(results)


@on_before_completion()
async def retrieval_memory(request: ModelRequest, event: Event):
    session = get_session()
    start_time = perf_counter()
    logger.info("开始检索记忆")

    async with session.begin():
        system = await rag_system.retrieval_memory_and_generate_prompt(
            session, request.prompt, event.get_user_id()
        )

    end_time = perf_counter()
    logger.info(f"记忆检索完成⭐用时{end_time - start_time}s")

    if not request.system:
        request.system = system
    else:
        request.system += system


@on_finish_chat()
async def establish_session_end_timer(message: Message, event: Event):
    _scheduler.add_summary_job(event.get_user_id(), message)
