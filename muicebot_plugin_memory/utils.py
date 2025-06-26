import re
from functools import cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
from muicebot.llm import ModelCompletions, ModelRequest
from muicebot.muice import Muice
from nonebot import logger

SEARCH_PATH = [Path(__file__).parent / "templates"]


@cache
def _get_template(template_name: str) -> Template:
    env = Environment(loader=FileSystemLoader(SEARCH_PATH))

    if not template_name.endswith((".j2", ".jinja2")):
        template_name += ".jinja2"
    try:
        template = env.get_template(template_name)
    except TemplateNotFound:
        logger.error(f"模板文件 {template_name} 未找到!")
        raise

    return template


def generate_prompt_from_template(template_name: str, **kwargs) -> str:
    """
    获取提示词
    （使用Jinja2模板引擎的目的是为了后续可能的扩展）
    """
    template = _get_template(template_name)

    prompt = template.render(kwargs)

    return prompt


async def chat_with_model(
    prompt: str,
    system: str,
) -> str:
    """
    与 llm 交互

    :raise RuntimeError: LLM 尚未运行
    """

    model = Muice.get_instance().model
    if not (model and model.is_running):
        raise RuntimeError("LLM 尚未运行！")

    model_request = ModelRequest(prompt, system=system)
    response_usage = -1
    logger.debug(f"向 LLM 发送请求: {model_request}")

    response = await model.ask(model_request, stream=model.config.stream)

    if isinstance(response, ModelCompletions):
        response_text = response.text
        response_usage = response.usage
    else:
        response_chunks: list[str] = []
        async for chunk in response:
            response_chunks.append(chunk.chunk)
            response_usage = chunk.usage or chunk.usage
        response_text = "".join(response_chunks)

    logger.debug(f"LLM 请求已完成，用量: {response_usage}")

    return response_text


def format_response_to_int(response_text: str) -> int:
    """
    将模型输出转换为 int 格式
    """
    response_text = process_message(response_text)
    try:
        response_int = int(response_text)
    except ValueError:
        logger.warning(f"尝试提取模型回复时出现错误，尝试提取数字: {response_text}")
        match = re.search(r"\d+", response_text)
        if match:
            response_int = int(match.group())
        else:
            response_int = 0
        logger.warning(f"提取结果: {response_int}")

    return response_int


def process_message(message: str) -> str:
    """
    提取思考结果
    """
    if not message.startswith("<think>"):
        return message

    thoughts_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    result = thoughts_pattern.sub("", message).strip()

    return result
