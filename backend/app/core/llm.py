"""LLM 初始化 — 实例缓存，避免在单 GPU 环境下重复创建连接"""

from langchain_openai import ChatOpenAI
from app.config import settings

# 按 (model, temperature) 缓存 LLM 实例，本地部署场景下
# 同一参数组合复用同一连接，减少 Ollama 端的会话开销
_llm_cache: dict[tuple[str, float], ChatOpenAI] = {}


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    _model = model or settings.LLM_MODEL
    _temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
    cache_key = (_model, _temp)

    if cache_key not in _llm_cache:
        if not settings.DASHSCOPE_API_KEY:
            raise RuntimeError("未配置 DASHSCOPE_API_KEY")
        _llm_cache[cache_key] = ChatOpenAI(
            model=_model,
            temperature=_temp,
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
            max_retries=2,
        )
    return _llm_cache[cache_key]


def clear_llm_cache():
    """清空缓存（模型切换或测试时使用）"""
    _llm_cache.clear()
