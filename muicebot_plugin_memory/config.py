from typing import Optional

from nonebot import get_plugin_config
from pydantic import BaseModel


class Config(BaseModel):
    memory_rag_api_key: str
    """访问嵌入模型所需的 API Key"""
    memory_rag_embedding_model: str = "text-embedding-v4"
    """嵌入模型名称"""
    memory_rag_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    """OpenAI 兼容端口的嵌入模型 base_url"""

    memory_session_expire_time: int = 60
    """会话过期时间（分钟）"""
    memory_session_min_epoch: int = 5
    """会话中的最小对话轮数"""
    memory_summary_model: Optional[str] = None
    """总结模型配置(记忆总结、重要性估计)"""

    memory_cosine_similarity: float = 0.8
    """余弦相似度阈值，低于此值的将不被回忆"""
    memory_retain_proportion: float = 0.1
    """记忆保留比例"""
    memory_lufy_importance_coefficient: float = 0.1
    """LUFY 模型 importance 权重(a), 当将其值设为 0 时表示不启用 LUFY 模型，改为仅余弦相似度模式"""
    memory_lufy_enable_arousal: bool = False
    """启用 LUFY 唤醒度计算"""


config = get_plugin_config(Config)
