# rag_pipeline/settings.py
import os
from pathlib import Path  # 导入 pathlib
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# 1. 定义项目的根目录
# Path(__file__) -> 获取当前文件(settings.py)的绝对路径
# .parent -> 获取父目录 (rag_pipeline/)
# .parent -> 再次获取父目录，即项目根目录 (controllable-rag/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 2. 构建 .env 文件的绝对路径
# 这样无论从哪里执行脚本，路径都是固定且正确的
ENV_FILE_PATH = PROJECT_ROOT / '.env'


class Settings(BaseSettings):
    """
    Manages all application settings.
    Settings are loaded from environment variables or a .env file.
    """
    model_config = SettingsConfigDict(
        # 3. 在这里使用我们计算好的绝对路径
        env_file=ENV_FILE_PATH, 
        env_file_encoding='utf-8', 
        extra='ignore'
    )

    # 1. Environment and API Keys
    openai_api_key: str | None = Field(None, alias='OPENAI_API_KEY')
    xiaocaseai_api_key: str | None = Field(None, alias='XIAOCASEAI_API_KEY')
    pydevd_warn_evaluation_timeout: int = 100000
    
    @property
    def api_key(self) -> str:
        """根据LLM提供商返回对应的API密钥"""
        if self.llm_provider.lower() == "xiaocaseai":
            return self.xiaocaseai_api_key or ""
        else:
            return self.openai_api_key or ""

    # 2. Model Configuration
    # 这些字段是必需的，必须在 .env 文件中提供
    # default_model: str = Field(..., alias='DEFAULT_LLM_MODEL')
    chat_model: str = Field(..., alias='CHAT_MODEL')
    planner_model: str = Field(..., alias='PLANNER_MODEL')
    embedding_model: str = Field(..., alias='EMBEDDING_MODEL')  # 默认embedding模型
    
    # LLM Provider Configuration
    llm_provider: str = Field(..., alias='LLM_PROVIDER')
    llm_base_url: str = Field(..., alias='LLM_BASE_URL')

    # 3. Vector Store Paths (使用Path对象，更加健壮)
    # 我们让它从字符串加载，然后通过property转换为相对于项目根目录的绝对路径
    chunks_vector_store_path_str: str = Field("chunks_vector_store", alias='CHUNKS_VECTOR_STORE_PATH')
    chapter_summaries_vector_store_path_str: str = Field("chapter_summaries_vector_store", alias='CHAPTER_SUMMARIES_VECTOR_STORE_PATH')
    book_quotes_vectorstore_path_str: str = Field("book_quotes_vectorstore", alias='BOOK_QUOTES_VECTORSTORE_PATH')
    
    # 4. Other LLM parameters
    default_temperature: float = 0.0
    default_max_tokens: int = 50000
    
    # 5. 动态计算向量存储的绝对路径 - 新设计：只有一个向量数据库
    @property
    def vector_store_path(self) -> Path:
        return PROJECT_ROOT / self.vector_store_path_str
    
    # === 原来的三个向量数据库路径属性（已注释） ===
    # @property
    # def chunks_vector_store_path(self) -> Path:
    #     return PROJECT_ROOT / self.chunks_vector_store_path_str

    # @property
    # def chapter_summaries_vector_store_path(self) -> Path:
    #     return PROJECT_ROOT / self.chapter_summaries_vector_store_path_str

    # @property
    # def book_quotes_vectorstore_path(self) -> Path:
    #     return PROJECT_ROOT / self.book_quotes_vectorstore_path_str


# Create a single, importable instance of the settings
settings = Settings()

# 删除或注释掉你的临时检验代码
# print(f"default_model: {settings.default_model}")
# ...

# 其他设置
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = str(settings.pydevd_warn_evaluation_timeout)
# The OpenAI key is automatically handled by the LangChain library
# if the environment variable is set. We don't need to manually set it in os.environ again.

# Now you can use the settings in your project like this:
# from rag_pipeline.settings import settings
# my_vector_store_path = settings.vector_store_path
# print(my_vector_store_path) 
# # Output would be C:\GitProject\controllable-rag\vector_stores\funding_guide_vectorstore (Path object)