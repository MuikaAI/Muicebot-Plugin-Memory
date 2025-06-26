from time import perf_counter

from muicebot.llm import ModelRequest
from muicebot.models import Message
from muicebot.plugin.func_call import on_function_call
from muicebot.plugin.func_call.parameter import Integer, String
from muicebot.plugin.hook import on_before_completion, on_finish_chat
from nonebot import get_driver, logger
from nonebot.adapters import Event
from nonebot_plugin_orm import get_session

from .config import config
from .rag import RAGSystem
from .scheduler import Scheduler

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


@on_function_call(description="手动添加对用户的记忆").params(
    memory=String(description="记忆内容", required=True),
    importance_score=Integer(
        description="重要性分数(0-10范围内的整数，越高的分数表示记忆越重要)",
        required=True,
    ),
)
async def record_memory(memory: str, importance_score: int, event: Event) -> str:
    logger.info(f"AI 请求手动保存记忆: {memory}")

    session = get_session()
    async with session.begin():
        await rag_system.mannual_record_memory(
            session, event.get_user_id(), memory, importance_score
        )

    logger.info("已成功执行了请求")
    return "成功"


@on_before_completion()
async def retrieval_memory(request: ModelRequest, event: Event):
    session = get_session()
    start_time = perf_counter()
    logger.info("开始检索记忆")

    async with session.begin():
        system = await rag_system.retrieval_memory(
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
