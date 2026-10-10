"""全局配置"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- 应用 ---
    APP_NAME: str = "法律检索增强问答系统"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # --- DashScope / 模型 ---
    DASHSCOPE_API_KEY: str = ""
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL: str = "qwen-turbo"
    LLM_TEMPERATURE: float = 0.3
    LLM_NUM_CTX: int = 8192
    LLM_TIMEOUT: int = 60
    EMBEDDING_MODEL: str = "text-embedding-v3"

    # --- DashScope 云端重排序 ---
    # 重排序接口使用百炼业务空间域名，与 OpenAI 兼容的 Chat/Embedding 地址不同。
    DASHSCOPE_WORKSPACE_ID: str = ""
    RERANKER_BASE_URL: str = ""
    RERANKER_MODEL: str = "qwen3.7-text-rerank"
    RERANKER_TIMEOUT: int = 30
    RERANKER_MAX_RETRIES: int = 2
    RERANKER_CANDIDATE_K: int = 20
    RERANKER_DOCUMENT_MAX_CHARS: int = 1200
    RERANKER_INSTRUCT: str = (
        "Given a Chinese legal question, rank passages by whether they provide "
        "accurate legal grounds for answering it."
    )

    # --- ChromaDB ---
    CHROMA_PERSIST_DIR: str = str(Path(__file__).resolve().parent.parent / "chroma_db")
    LAWS_COLLECTION: str = "laws"
    CASES_COLLECTION: str = "cases"

    # --- MySQL 原文库（唯一事实来源）---
    MYSQL_URL: str = "mysql+pymysql://root:password@127.0.0.1:3306/lawrag?charset=utf8mb4"
    MYSQL_ECHO: bool = False
    CHUNKER_VERSION: str = "parent-child-v1"
    CHILD_CHUNK_SIZE: int = 320
    CHILD_CHUNK_OVERLAP: int = 64
    CHILD_RERANK_TOP_K: int = 15
    PARENT_TOP_K: int = 5
    PARENT_HIT_BONUS: float = 0.02

    # --- 检索 ---
    RETRIEVAL_TOP_K: int = 10
    BM25_WEIGHT: float = 0.5
    VECTOR_WEIGHT: float = 0.5
    RERANK_TOP_K: int = 5

    # --- 数据目录 ---
    DATA_DIR: str = str(Path(__file__).resolve().parent.parent / "data")
    LAWS_DIR: str = str(Path(__file__).resolve().parent.parent / "data" / "laws")
    CASES_DIR: str = str(Path(__file__).resolve().parent.parent / "data" / "cases")
    REPORTS_DIR: str = str(Path(__file__).resolve().parent.parent / "data" / "reports")
    CHAT_RECORDS_DIR: str = str(Path(__file__).resolve().parent.parent / "data" / "chat_records")

    # --- 高级 RAG 管线 ---
    CONTEXTUAL_CHUNKING: bool = True
    KG_DATA_PATH: str = str(
        Path(__file__).resolve().parent.parent / "data" / "laws" / "CrimeKG" / "犯罪知识图谱.txt"
    )
    HYDE_TEMPERATURE: float = 0.7
    SELF_REFLECT_MAX_ITER: int = 1
    DEFAULT_QUERY_TRANSFORM: str = "none"
    DEFAULT_RERANK: str = "simple"
    DEFAULT_GENERATION: str = "standard"

    # --- 上下文工程与短期记忆 ---
    CONTEXT_MAX_TOKENS: int = 4000
    MEMORY_TTL_SECONDS: int = 86400
    MEMORY_MAX_TURNS: int = 6

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
