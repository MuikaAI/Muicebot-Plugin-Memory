from datetime import datetime, timedelta
from time import perf_counter
from typing import Callable, Literal

from apscheduler.job import Job
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from muicebot.models import Message
from nonebot import logger
from nonebot_plugin_apscheduler import scheduler
from nonebot_plugin_orm import get_scoped_session

from .config import config
from .rag import RAGSystem


class Scheduler:
    def __init__(self) -> None:
        self.scheduler = scheduler
        self.jobs: list[Job] = []
        self._user_message_pool: dict[str, list[Message]] = {}

    def add_job(
        self,
        func: Callable,
        job_id: str,
        trigger_type: Literal["cron", "interval", "date"],
        **trigger_args,
    ):
        if trigger_type == "cron":
            trigger = CronTrigger(**trigger_args)

        elif trigger_type == "interval":
            trigger = IntervalTrigger(**trigger_args)

        elif trigger_type == "date":
            trigger = DateTrigger(**trigger_args)

        job = self.scheduler.add_job(func, trigger, id=job_id, replace_existing=True)

        self.jobs.append(job)

    def add_summary_job(self, userid: str, message: Message):
        self._user_message_pool.setdefault(userid, []).append(message)

        async def run_summary():
            if len(self._user_message_pool[userid]) < config.memory_session_min_epoch:
                logger.debug("会话总长度小于阈值，已跳过总结")
                del self._user_message_pool[userid]
                return

            logger.info(f"开始总结用户{userid}的记忆...")
            start_time = perf_counter()
            session = get_scoped_session()
            rag_system = RAGSystem.get_instance()

            logger.info(f"[{userid}] 总结对话 1/3")
            await rag_system.summary_conversations(
                session, self._user_message_pool[userid]
            )
            logger.info(f"[{userid}] 提取关键记忆 2/3")
            await rag_system.save_memories(session, self._user_message_pool[userid])
            logger.info(f"[{userid}] 更新关键记忆 3/3")
            await rag_system.key_summary(session, userid)

            end_time = perf_counter()
            logger.success(
                f"用户{userid}的记忆已总结完成⭐(用时{ end_time - start_time}s)"
            )
            del self._user_message_pool[userid]

        run_time = datetime.now() + timedelta(minutes=config.memory_session_expire_time)
        self.add_job(run_summary, userid, "date", run_date=run_time)
