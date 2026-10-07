"""Embedding 初始化 — 单例缓存"""

from langchain_openai import OpenAIEmbeddings
from app.config import settings

_embed_cache: dict[str, OpenAIEmbeddings] = {}


def get_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    _model = model or settings.EMBEDDING_MODEL
    if _model not in _embed_cache:
        if not settings.DASHSCOPE_API_KEY:
            raise RuntimeError("未配置 DASHSCOPE_API_KEY")
        _embed_cache[_model] = OpenAIEmbeddings(
            model=_model,
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_BASE_URL,
            chunk_size=10,
            check_embedding_ctx_length=False,
            max_retries=2,
        )
    return _embed_cache[_model]
