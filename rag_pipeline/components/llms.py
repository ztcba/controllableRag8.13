# 创建用于实例化 ChatOpenAI 模型的函数
# 将模型实例化逻辑集中起来，可以轻松更换模型或统一模型参数（如 temperature）
# 开始使用之前创建的 settings.py 来获取配置
# src/rag_pipeline/components/llms.py

from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI as CommunityChatOpenAI
from rag_pipeline.settings import settings

def get_chat_model() -> ChatOpenAI:
    """
    Creates and returns a ChatOpenAI instance configured from settings.
    
    This function correctly handles pointing to any OpenAI-compatible API 
    (like xiaocaseai) by using the base_url parameter, while retaining 
    all advanced features like structured output.
    """
    
    # 3. 始终使用 langchain_openai.ChatOpenAI。
    #    通过 api_key 和 base_url 参数，将其指向你的第三方服务。
    #    这样就能解决 with_structured_output 的 NotImplementedError。
    return ChatOpenAI(
        temperature=settings.default_temperature,
        model=settings.chat_model, # 注意：参数名是 model 而不是 model_name
        max_tokens=settings.default_max_tokens,
        api_key=settings.openai_api_key, # Pydantic 会从 .env 加载
        base_url=settings.llm_base_url    # Pydantic 会从 .env 加载
    )

def get_planner_model() -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance configured with settings.
    Supports multiple LLM providers based on settings.
    """
    # Use the same logic as get_chat_model for consistency
    return ChatOpenAI(
        temperature=settings.default_temperature,
        model=settings.planner_model, # 注意：参数名是 model 而不是 model_name
        max_tokens=settings.default_max_tokens,
        api_key=settings.openai_api_key, # Pydantic 会从 .env 加载
        base_url=settings.llm_base_url    # Pydantic 会从 .env 加载
    )

def get_embedding_model() -> ChatOpenAI:
    """
    Returns an OpenAIEmbeddings instance configured with settings.
    Used for creating embeddings for vector stores.
    """
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        base_url=settings.llm_base_url  # Pydantic 会从 .env 加载
    )