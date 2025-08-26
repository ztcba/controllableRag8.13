from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from rag_pipeline.settings import settings

def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.default_temperature,
        max_tokens=settings.default_max_tokens,
        openai_api_key=settings.api_key,
        openai_api_base=settings.llm_base_url
    )

def get_planner_model() -> ChatOpenAI:
    return ChatOpenAI(
        temperature=settings.default_temperature,
        model=settings.planner_model,
        max_tokens=settings.default_max_tokens,
        api_key=settings.api_key,
        base_url=settings.llm_base_url
    )

def get_embedding_model() -> OpenAIEmbeddings:
    import os
    os.environ['OPENAI_API_KEY'] = settings.api_key
    
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.api_key,
        openai_api_base=settings.llm_base_url
    )