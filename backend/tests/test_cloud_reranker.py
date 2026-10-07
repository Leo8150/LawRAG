"""云端 Reranker 测试：全部使用 MockTransport，不访问真实 API。"""

import asyncio
import json

import httpx
from langchain_core.documents import Document


def test_cloud_rerank_request_and_response(monkeypatch):
    from app.config import settings
    from app.services.cloud_reranker import cloud_rerank

    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(settings, "RERANKER_BASE_URL", "https://example.test/rerank")
    monkeypatch.setattr(settings, "RERANKER_MODEL", "qwen3.7-text-rerank")
    monkeypatch.setattr(settings, "RERANKER_MAX_RETRIES", 0)

    docs = [
        Document(page_content="无关文本", metadata={"id": "a"}),
        Document(page_content="劳动合同法规定用人单位应及时支付工资", metadata={"id": "b"}),
        Document(page_content="民法典合同条文", metadata={"id": "c"}),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3.7-text-rerank"
        assert payload["input"]["query"] == "公司拖欠工资怎么办"
        assert len(payload["input"]["documents"]) == 3
        assert payload["parameters"]["top_n"] == 2
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.96},
                        {"index": 2, "relevance_score": 0.41},
                    ]
                },
                "usage": {"total_tokens": 123},
                "request_id": "mock-request",
            },
        )

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await cloud_rerank(
                "公司拖欠工资怎么办", docs, top_k=2, client=client
            )

    result = asyncio.run(run())
    assert [doc.metadata["id"] for doc, _ in result.ranked_documents] == ["b", "c"]
    assert [score for _, score in result.ranked_documents] == [0.96, 0.41]
    assert result.total_tokens == 123
    assert result.request_id == "mock-request"


def test_cloud_rerank_missing_config(monkeypatch):
    from app.config import settings
    from app.services.cloud_reranker import CloudRerankError, get_reranker_url

    monkeypatch.setattr(settings, "RERANKER_BASE_URL", "")
    monkeypatch.setattr(settings, "DASHSCOPE_WORKSPACE_ID", "")

    try:
        get_reranker_url()
    except CloudRerankError as exc:
        assert "DASHSCOPE_WORKSPACE_ID" in str(exc)
    else:
        raise AssertionError("缺少云端重排配置时应报错")


def test_pipeline_cloud_rerank_falls_back_without_api(monkeypatch):
    from app.services import cloud_reranker
    from app.services.cloud_reranker import CloudRerankError
    from app.services.pipeline import PipelineConfig, RAGPipeline, RerankStrategy

    async def fail_without_network(*_args, **_kwargs):
        raise CloudRerankError("mock unavailable")

    monkeypatch.setattr(cloud_reranker, "cloud_rerank", fail_without_network)
    pipeline = RAGPipeline(PipelineConfig(rerank_strategy=RerankStrategy.CLOUD, top_k=2))
    docs = [
        Document(page_content="天气信息", metadata={"id": "a"}),
        Document(page_content="劳动合同工资支付规定", metadata={"id": "b"}),
        Document(page_content="公司拖欠工资可以投诉", metadata={"id": "c"}),
    ]

    ranked = asyncio.run(pipeline._rerank("公司拖欠工资", docs))
    assert len(ranked) == 2
    assert pipeline.metrics["rerank_fallback"] is True
    assert pipeline.metrics["rerank_api_calls"] == 1
    assert all(doc.metadata["rerank_strategy"] == "simple_fallback" for doc in ranked)
