from typing import Optional, Sequence, Union

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_scoped_session

from .models import ConversationSummary, MemoryMetric, UserProfile

DatabaseSession = Union[async_scoped_session, AsyncSession]


class MemoryRepository:
    @staticmethod
    async def add(session: DatabaseSession, memory: MemoryMetric, flush: bool = True):
        session.add(memory)
        return await session.flush() if flush else None

    @staticmethod
    async def get_by_user_id(
        session: DatabaseSession, userid: str, limit: Optional[int] = 100
    ) -> Sequence[MemoryMetric]:
        results = await session.execute(
            select(MemoryMetric)
            .where(
                MemoryMetric.userid == userid
                and MemoryMetric.retained == True  # noqa: E712
            )
            .limit(limit)
        )
        return results.scalars().all()

    @staticmethod
    async def mark_unretained(session: DatabaseSession, memory_ids: list[int]):
        await session.execute(
            update(MemoryMetric)
            .where(MemoryMetric.id.in_(memory_ids))
            .values(retained=False)
        )


class UserProfileRepository:
    @staticmethod
    async def add(session: DatabaseSession, memory: UserProfile):
        session.add(memory)
        await session.flush()

    @staticmethod
    async def get_by_userid(
        session: DatabaseSession, userid: str
    ) -> Optional[UserProfile]:
        result = await session.execute(
            select(UserProfile).where(UserProfile.user_id == userid)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_user_profiles(
        session: DatabaseSession, limit: Optional[int] = 100
    ) -> Sequence[UserProfile]:
        result = await session.execute(
            select(UserProfile).order_by(UserProfile.updated_at.desc()).limit(limit)
        )
        return result.scalars().all()


class SummaryRepository:
    @staticmethod
    async def add(session: DatabaseSession, summary: ConversationSummary):
        session.add(summary)
        await session.flush()

    @staticmethod
    async def query_recent_summary(
        session: DatabaseSession, userid: str, limit: int = 3
    ) -> Sequence[ConversationSummary]:
        result = await session.execute(
            select(ConversationSummary)
            .where(ConversationSummary.userid == userid)
            .order_by(ConversationSummary.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
