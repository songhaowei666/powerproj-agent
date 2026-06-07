"""全局配置管理，基于 pydantic-settings 读取 .env 文件。"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置，字段名与 .env 中的变量名保持一致。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI 兼容接口
    openai_api_key: str = ""
    openai_api_base: str = ""
    chat_model: str = "gpt-4o-mini"

    # Embedding
    embedding_model: str = "text-embedding-3-large"

    # 可选第三方服务
    gemini_api_key: str = ""
    jina_api_key: str = ""
    mineru_api_key: str = ""


# 单例，启动时加载一次
settings = Settings()
