from pydantic import BaseSettings

class Settings(BaseSettings):
    api_key: str
    chat_model: str = "gpt-3.5-turbo"
    planner_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-ada-002"
    default_temperature: float = 0.7
    default_max_tokens: int = 150
    llm_base_url: str = "https://api.openai.com/v1"

    class Config:
        env_file = ".env"  # Load environment variables from .env file

settings = Settings()