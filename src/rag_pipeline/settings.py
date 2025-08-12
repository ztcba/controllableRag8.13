# src/rag_pipeline/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Manages all application settings.
    Settings are loaded from environment variables or a .env file.
    """
    # 1. Environment and API Keys
    # The model_config tells pydantic to load variables from a .env file
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    openai_api_key: str | None = Field(None, alias='OPENAI_API_KEY')
    # groq_api_key: str | None = Field(None, alias='GROQ_API_KEY') # Uncomment if you use it
    pydevd_warn_evaluation_timeout: int = 100000

    # 2. Model Configuration
    # We can define different models and easily switch between them
    default_model: str = "gpt-3.5-turbo"
    # large_model: str = "llama3-70b-8192" # Example for Groq
    
    # LLM Provider Configuration
    llm_provider: str = "xiaocaseai"  # Can be "openai", "xiaocaseai", or other providers
    llm_base_url: str = "https://api.xiaocaseai.com/v1"  # Base URL for the LLM API

    # 3. Vector Store Paths
    # Centralizing paths makes them easy to change
    # pydantic会先在.env 文件中查找这些变量，如果没有找到，则使用默认值绝对路径创建类似/rag_pipeline/vector_stores/chunks_vector_store
    chunks_vector_store_path: str = "chunks_vector_store"
    chapter_summaries_vector_store_path: str = "chapter_summaries_vector_store"
    book_quotes_vectorstore_path: str = "book_quotes_vectorstore"
    
    # 4. Other LLM parameters
    default_temperature: float = 0.0
    default_max_tokens: int = 2000


# Create a single, importable instance of the settings
settings = Settings()

# We can also set other environment variables if needed
import os
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = str(settings.pydevd_warn_evaluation_timeout)
# The OpenAI key is automatically handled by the LangChain library
# if the environment variable is set. We don't need to manually set it in os.environ again.