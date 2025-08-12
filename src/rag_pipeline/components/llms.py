# 创建用于实例化 ChatOpenAI 模型的函数
# 将模型实例化逻辑集中起来，可以轻松更换模型或统一模型参数（如 temperature）
# 开始使用之前创建的 settings.py 来获取配置
# src/rag_pipeline/components/llms.py

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI as CommunityChatOpenAI
from src.rag_pipeline.settings import settings

def get_chat_model():
    """
    Returns a ChatOpenAI instance configured with settings.
    Supports multiple LLM providers based on settings.
    """
    if settings.llm_provider.lower() == "xiaocaseai":
        return CommunityChatOpenAI(
            temperature=settings.default_temperature,
            model_name=settings.default_model,
            max_tokens=settings.default_max_tokens,
            openai_api_key=settings.openai_api_key or "sk-WxkpgKYFSkkJkoUEfhlXhXsbl3QJ7fiAfG9POH2eXMQytiUp",  # xiaocaseai不需要真实的API密钥
            openai_api_base=settings.llm_base_url
        )

    else:
        # Default to OpenAI if provider is not specified or not supported
        return ChatOpenAI(
            temperature=settings.default_temperature,
            model_name=settings.default_model,
            max_tokens=settings.default_max_tokens,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.llm_base_url if hasattr(settings, 'llm_base_url') else None
        )

def get_planner_model():
    """
    Returns a ChatOpenAI instance configured with settings.
    Supports multiple LLM providers based on settings.
    """
    # Use the same logic as get_chat_model for consistency
    if settings.llm_provider.lower() == "xiaocaseai":
        return CommunityChatOpenAI(
            temperature=settings.default_temperature,
            model_name=settings.default_model,
            max_tokens=settings.default_max_tokens,
            openai_api_key=settings.openai_api_key or "sk-WxkpgKYFSkkJkoUEfhlXhXsbl3QJ7fiAfG9POH2eXMQytiUp",  # xiaocaseai不需要真实的API密钥
            openai_api_base=settings.llm_base_url
        )

    else:
        # Default to OpenAI if provider is not specified or not supported
        return ChatOpenAI(
            temperature=settings.default_temperature,
            model_name=settings.default_model,
            max_tokens=settings.default_max_tokens,
            openai_api_key=settings.openai_api_key
        )
