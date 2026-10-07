"""DashScope 云端专用 Reranker。

该模块只在管线显式选择 ``rerank_strategy=cloud`` 时发起请求。
服务启动、健康检查和默认 ``simple`` 策略均不会调用此接口。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from langchain_core.documents import Document

from app.config import settings


class CloudRerankError(RuntimeError):
    """云端重排序请求或响应无效。"""


@dataclass
class CloudRerankResult:
    ranked_documents: list[tuple[Document, float]]
    total_tokens: int = 0
    request_id: str | None = None
    model: str = ""


def get_reranker_url() -> str:
    """返回百炼文本排序接口地址，不发起网络请求。"""
    if settings.RERANKER_BASE_URL:
        return settings.RERANKER_BASE_URL.rstrip("/")
    if not settings.DASHSCOPE_WORKSPACE_ID:
        raise CloudRerankError(
            "未配置 DASHSCOPE_WORKSPACE_ID 或 RERANKER_BASE_URL，无法调用云端重排序"
        )
    return (
        f"https://{settings.DASHSCOPE_WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com"
        "/api/v1/services/rerank/text-rerank/text-rerank"
    )


def _parse_response(
    data: dict,
    documents: list[Document],
    top_n: int,
) -> CloudRerankResult:
    """解析 qwen3.7-text-rerank 响应并映射回原始 Document。"""
    result_container = data.get("output", data)
    raw_results = result_container.get("results")
    if not isinstance(raw_results, list):
        message = data.get("message") or "响应中缺少 results"
        raise CloudRerankError(f"云端重排序响应无效: {message}")

    ranked: list[tuple[Document, float]] = []
    seen: set[int] = set()
    for item in raw_results:
        try:
            index = int(item["index"])
            score = float(item["relevance_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CloudRerankError("云端重排序结果字段格式错误") from exc
        if not 0 <= index < len(documents) or index in seen:
            continue
        seen.add(index)
        ranked.append((documents[index], max(0.0, min(score, 1.0))))

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    usage = data.get("usage") or result_container.get("usage") or {}
    return CloudRerankResult(
        ranked_documents=ranked[:top_n],
        total_tokens=int(usage.get("total_tokens", 0) or 0),
        request_id=data.get("request_id") or data.get("id"),
        model=str(data.get("model") or settings.RERANKER_MODEL),
    )


async def cloud_rerank(
    query: str,
    documents: list[Document],
    top_k: int = 5,
    *,
    client: httpx.AsyncClient | None = None,
) -> CloudRerankResult:
    """调用百炼专用文本排序模型。

    ``client`` 参数用于测试时注入 ``httpx.MockTransport``，从而验证完整
    请求与解析逻辑而不访问真实 API。
    """
    if not documents:
        return CloudRerankResult([], model=settings.RERANKER_MODEL)
    if not settings.DASHSCOPE_API_KEY:
        raise CloudRerankError("未配置 DASHSCOPE_API_KEY，无法调用云端重排序")

    candidates = documents[: settings.RERANKER_CANDIDATE_K]
    top_n = min(max(top_k, 1), len(candidates))
    payload = {
        "model": settings.RERANKER_MODEL,
        "input": {
            "query": query,
            "documents": [
                doc.page_content[: settings.RERANKER_DOCUMENT_MAX_CHARS]
                for doc in candidates
            ],
        },
        "parameters": {
            "top_n": top_n,
            "instruct": settings.RERANKER_INSTRUCT,
        },
    }
    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    owned_client = client is None
    http_client = client or httpx.AsyncClient(timeout=settings.RERANKER_TIMEOUT)
    last_error: Exception | None = None
    try:
        for attempt in range(settings.RERANKER_MAX_RETRIES + 1):
            try:
                response = await http_client.post(
                    get_reranker_url(),
                    headers=headers,
                    json=payload,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                if response.status_code >= 400:
                    detail = response.text[:300]
                    raise CloudRerankError(
                        f"云端重排序请求失败 ({response.status_code}): {detail}"
                    )
                return _parse_response(response.json(), candidates, top_n)
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt >= settings.RERANKER_MAX_RETRIES:
                    break
                await asyncio.sleep(0.25 * (2**attempt))
    finally:
        if owned_client:
            await http_client.aclose()

    raise CloudRerankError(f"云端重排序暂不可用: {last_error}")
