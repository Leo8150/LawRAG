"""RAG 评测服务。

核心指标：Recall@5、MRR@10、P95 Latency 和 Faithfulness。
P95 Latency 由性能基准服务根据整批请求延迟计算。
"""

import json
import re
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.llm import get_llm
from app.models.schemas import (
    FaithfulnessScore,
    QualityAggregated,
    QualityMetrics,
    RetrievalMetrics,
    SourceDocument,
)


_eval_dataset: dict[str, dict[str, Any]] | None = None


def _load_eval_dataset() -> dict[str, dict[str, Any]]:
    """按问题加载带相关文档标注的 RAG 评测集。"""
    global _eval_dataset
    if _eval_dataset is not None:
        return _eval_dataset

    dataset_path = Path(settings.DATA_DIR) / "rag_eval_dataset.json"
    if not dataset_path.exists():
        _eval_dataset = {}
        return _eval_dataset

    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("rag_eval_dataset.json 必须是 JSON 数组")

    _eval_dataset = {
        record["question"].strip(): record
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("question"), str)
        and record["question"].strip()
    }
    return _eval_dataset


def _normalize(value: Any) -> str:
    """统一法律名称、条号和文件名的比较格式。"""
    return re.sub(r"[\s《》〈〉第条（）()·._-]", "", str(value or "")).lower()


def _value_matches(actual: Any, expected: Any) -> bool:
    actual_norm = _normalize(actual)
    expected_norm = _normalize(expected)
    if not actual_norm or not expected_norm:
        return False
    return actual_norm == expected_norm or expected_norm in actual_norm or actual_norm in expected_norm


def _source_matches_relevant(source: SourceDocument, relevant: dict[str, Any]) -> bool:
    """判断一个召回结果是否符合一条标准相关文档标注。"""
    metadata = source.metadata or {}
    content_contains = relevant.get("content_contains")
    if content_contains and _normalize(content_contains) not in _normalize(source.content):
        return False

    criteria = {
        key: value
        for key, value in relevant.items()
        if key != "content_contains" and value not in (None, "")
    }
    if not criteria:
        return bool(content_contains)
    return all(_value_matches(metadata.get(key), value) for key, value in criteria.items())


def evaluate_retrieval(
    query: str,
    sources: list[SourceDocument],
) -> RetrievalMetrics | None:
    """根据人工标注计算单条查询的 Recall@5 和 MRR@10。"""
    record = _load_eval_dataset().get(query.strip())
    relevant_documents = record.get("relevant_documents", []) if record else []
    if not relevant_documents:
        return None

    matched_relevant: set[int] = set()
    for source in sources[:5]:
        for index, relevant in enumerate(relevant_documents):
            if _source_matches_relevant(source, relevant):
                matched_relevant.add(index)

    first_relevant_rank = None
    for rank, source in enumerate(sources[:10], start=1):
        if any(_source_matches_relevant(source, item) for item in relevant_documents):
            first_relevant_rank = rank
            break

    return RetrievalMetrics(
        recall_at_5=round(len(matched_relevant) / len(relevant_documents), 4),
        mrr_at_10=round(1 / first_relevant_rank, 4) if first_relevant_rank else 0.0,
        relevant_document_count=len(relevant_documents),
        retrieved_relevant_at_5=len(matched_relevant),
        first_relevant_rank=first_relevant_rank,
    )


_FAITHFULNESS_PROMPT = """请判断以下回答是否忠实于提供的参考来源。

用户问题：{query}

参考来源：
{sources}

生成的回答：
{answer}

请从以下两个方面评估：
1. 回答中的信息是否都能在参考来源中找到依据
2. 回答是否存在编造或添加参考来源中没有的法律条文

请按以下格式返回（只返回两行，不要其他内容）：
评分：[0-10的整数]
说明：[一句话解释]"""


async def evaluate_faithfulness(
    query: str,
    answer: str,
    sources: list[SourceDocument],
) -> FaithfulnessScore | None:
    """使用 LLM Judge 评估忠实度；仅在显式开启质量评测时调用。"""
    if not sources or not answer:
        return None

    llm = get_llm(temperature=0)
    sources_text = "\n".join(
        f"[来源{i + 1}] {src.content[:400]}" for i, src in enumerate(sources[:5])
    )
    prompt = _FAITHFULNESS_PROMPT.format(
        query=query,
        sources=sources_text,
        answer=answer[:1000],
    )

    try:
        resp = await llm.ainvoke(prompt)
        score = None
        explanation = ""
        for line in resp.content.strip().split("\n"):
            if "评分" in line:
                digits = "".join(c for c in line if c.isdigit() or c == ".")
                if digits:
                    score = min(max(float(digits), 0), 10)
            if "说明" in line:
                explanation = line.split("：", 1)[-1].strip() if "：" in line else line.split(":", 1)[-1].strip()
        if score is None:
            return None
        return FaithfulnessScore(score=round(score, 1), explanation=explanation)
    except Exception:
        return None


async def evaluate_single_query(
    query: str,
    answer: str,
    sources: list[SourceDocument],
) -> QualityMetrics:
    """计算一条查询的检索指标和回答忠实度。"""
    retrieval = evaluate_retrieval(query, sources)
    faithfulness = await evaluate_faithfulness(query, answer, sources)
    return QualityMetrics(query=query, retrieval=retrieval, faithfulness=faithfulness)


def aggregate_quality(quality_list: list[QualityMetrics]) -> QualityAggregated:
    """汇总 Recall@5、MRR@10 和 Faithfulness。"""
    if not quality_list:
        return QualityAggregated()

    retrieval_items = [item.retrieval for item in quality_list if item.retrieval is not None]
    faithfulness_items = [item.faithfulness for item in quality_list if item.faithfulness is not None]

    def _average(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return QualityAggregated(
        recall_at_5=_average([item.recall_at_5 for item in retrieval_items]),
        mrr_at_10=_average([item.mrr_at_10 for item in retrieval_items]),
        avg_faithfulness=_average([item.score for item in faithfulness_items]),
        evaluated_count=len(quality_list),
        retrieval_evaluated_count=len(retrieval_items),
        faithfulness_evaluated_count=len(faithfulness_items),
    )
