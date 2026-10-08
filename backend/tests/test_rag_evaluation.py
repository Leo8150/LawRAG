"""RAG 核心评测指标的离线测试。"""

from app.models.schemas import FaithfulnessScore, QualityMetrics, RetrievalMetrics, SourceDocument
from app.services import quality_service
from app.services.perf_service import calculate_p95


def test_recall_at_5_and_mrr_at_10(monkeypatch):
    monkeypatch.setattr(
        quality_service,
        "_eval_dataset",
        {
            "测试问题": {
                "question": "测试问题",
                "relevant_documents": [
                    {"law_name": "中华人民共和国刑法"},
                    {"law_name": "中华人民共和国民法典"},
                ],
            }
        },
    )
    sources = [
        SourceDocument(content="无关内容", metadata={"law_name": "中华人民共和国行政诉讼法"}),
        SourceDocument(content="相关内容", metadata={"law_name": "中华人民共和国刑法"}),
        SourceDocument(content="无关内容", metadata={"law_name": "中华人民共和国公司法"}),
        SourceDocument(content="无关内容", metadata={"law_name": "中华人民共和国劳动合同法"}),
        SourceDocument(content="无关内容", metadata={"law_name": "中华人民共和国著作权法"}),
        SourceDocument(content="相关内容", metadata={"law_name": "中华人民共和国民法典"}),
    ]

    result = quality_service.evaluate_retrieval("测试问题", sources)

    assert result is not None
    assert result.recall_at_5 == 0.5
    assert result.mrr_at_10 == 0.5
    assert result.first_relevant_rank == 2


def test_unlabeled_query_skips_retrieval_metrics(monkeypatch):
    monkeypatch.setattr(quality_service, "_eval_dataset", {})
    assert quality_service.evaluate_retrieval("未标注问题", []) is None


def test_rag_metric_aggregation():
    metrics = [
        QualityMetrics(
            query="q1",
            retrieval=RetrievalMetrics(recall_at_5=1.0, mrr_at_10=0.5),
            faithfulness=FaithfulnessScore(score=9),
        ),
        QualityMetrics(
            query="q2",
            retrieval=RetrievalMetrics(recall_at_5=0.5, mrr_at_10=1.0),
            faithfulness=FaithfulnessScore(score=7),
        ),
    ]

    result = quality_service.aggregate_quality(metrics)

    assert result.recall_at_5 == 0.75
    assert result.mrr_at_10 == 0.75
    assert result.avg_faithfulness == 8.0
    assert result.retrieval_evaluated_count == 2


def test_p95_latency_uses_nearest_rank():
    assert calculate_p95([]) == 0.0
    assert calculate_p95(list(range(1, 101))) == 95
    assert calculate_p95([100, 200]) == 200
