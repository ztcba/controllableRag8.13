# 创建用于实例化 ChatOpenAI 模型的函数
# 将模型实例化逻辑集中起来，可以轻松更换模型或统一模型参数（如 temperature）
# 开始使用之前创建的 settings.py 来获取配置
# src/rag_pipeline/components/llms.py

from langchain_openai import ChatOpenAI
from src.rag_pipeline.settings import settings

def get_chat_model() -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance configured with settings.
    """
    return ChatOpenAI(
        temperature=settings.default_temperature,
        model_name=settings.default_model,
        max_tokens=settings.default_max_tokens,
        openai_api_key=settings.openai_api_key
    )

def get_planner_model() -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance configured with settings.
    """
    return ChatOpenAI(
        temperature=settings.default_temperature,
        model_name=settings.default_model,
        max_tokens=settings.default_max_tokens,
        openai_api_key=settings.openai_api_key
    )