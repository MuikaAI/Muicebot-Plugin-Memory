from datetime import datetime

from nonebot_plugin_orm import Model
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column


class UserProfile(Model):
    user_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="用户ID")
    # name: Mapped[str] = mapped_column(String(32), comment="用户昵称")
    key_memory: Mapped[str] = mapped_column(Text, comment="关键记忆")
    preferences: Mapped[str] = mapped_column(
        Text, comment="用户偏好（列表映射为字符串）", default="[]"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


# class GroupProfile(Model):
#     group_id = Column(String(64), primary_key=True, comment="群组ID")
#     # name = Column(String(32), comment="群组名称")
#     key_memory = Column(Text, comment="关键记忆")


class MemoryMetric(Model):
    """记忆库(输入-输出对)"""

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    userid: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(String, comment="记忆内容(即用户输入)")

    # Psychological Metrics
    arousal: Mapped[float] = mapped_column(Float, default=0.0, comment="唤醒程度")
    llm_importance: Mapped[float] = mapped_column(
        Float, default=0.0, comment="LLM评估的重要性"
    )
    r1_count: Mapped[int] = mapped_column(
        Integer, default=0, comment="第一相关记忆计数"
    )
    r2_count: Mapped[int] = mapped_column(
        Integer, default=0, comment="第二相关记忆计数"
    )
    session_interval: Mapped[int] = mapped_column(
        Integer, default=0, comment="至最后一次检索以来经过的会话数Δt"
    )
    total_retrieval_count: Mapped[int] = mapped_column(
        Integer, default=1, comment="总共被检索的次数S"
    )

    # 是否在记忆库中保留
    retained: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ConversationSummary(Model):
    """对话总结"""

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(String, comment="笔记内容")
    userid: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
