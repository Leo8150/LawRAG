"""RAG 核心指标与性能基准测试服务。"""

import math
import time
import psutil
from app.services.rag_service import rag_query
from app.config import settings
from app.models.schemas import (
    SystemInfo,
    BenchmarkResult,
    BenchmarkResultV2,
    SourceDocument,
)


LEGAL_TEST_QUERIES = [
    "民法典中关于合同成立的规定是什么？",
    "故意杀人罪的量刑标准是什么？",
    "劳动合同法中关于经济补偿金的规定",
    "交通事故损害赔偿的法律依据",
    "知识产权侵权的认定标准",
    "公司法中股东代表诉讼的条件",
    "行政诉讼的受案范围",
    "最高法关于民间借贷利率的指导案例",
]


def calculate_p95(latencies: list[float]) -> float:
    """使用最近秩法计算 P95 延迟。"""
    if not latencies:
        return 0.0
    ordered = sorted(latencies)
    return ordered[max(math.ceil(len(ordered) * 0.95) - 1, 0)]


def get_system_info() -> SystemInfo:
    """获取当前系统资源状态"""
    mem = psutil.virtual_memory()
    return SystemInfo(
        cpu_percent=psutil.cpu_percent(interval=0.5),
        memory_percent=mem.percent,
        memory_used_gb=round(mem.used / (1024**3), 2),
        memory_total_gb=round(mem.total / (1024**3), 2),
    )


async def get_model_status() -> dict:
    """返回云端模型配置状态，不额外消耗 API 额度。"""
    return {
        "provider": "dashscope",
        "status": "configured" if settings.DASHSCOPE_API_KEY else "missing_api_key",
        "base_url": settings.DASHSCOPE_BASE_URL,
    }


async def run_benchmark(
    queries: list[str] | None = None,
    use_rerank: bool = True,
    evaluate_quality: bool = False,
) -> BenchmarkResultV2:
    """运行基准测试，汇总 Recall@5、MRR@10、P95 Latency 和 Faithfulness。"""
    test_queries = queries or LEGAL_TEST_QUERIES[:4]
    sys_info = get_system_info()

    details = []
    total_retrieval = 0.0
    total_generation = 0.0
    total_latency = 0.0
    latencies = []

    # RAG 评测结果
    quality_details = []

    for q in test_queries:
        t0 = time.time()
        try:
            # MRR@10 需要保留前 10 个检索结果；Recall@5 从其中前 5 个计算。
            result = await rag_query(
                question=q,
                use_rerank=use_rerank,
                use_query_rewrite=False,
                top_k=10,
            )
            latency = (time.time() - t0) * 1000
            latencies.append(latency)
            details.append({
                "query": q,
                "latency_ms": round(latency, 1),
                "retrieval_ms": result.metrics.retrieval_ms,
                "generation_ms": result.metrics.generation_ms,
                "sources_count": len(result.sources),
                "answer_length": len(result.answer),
            })
            total_retrieval += result.metrics.retrieval_ms
            total_generation += result.metrics.generation_ms
            total_latency += latency

            # 检索与忠实度评测
            if evaluate_quality:
                from app.services.quality_service import evaluate_single_query
                qm = await evaluate_single_query(
                    query=q,
                    answer=result.answer,
                    sources=result.sources,
                )
                quality_details.append(qm)

        except Exception as e:
            latency = (time.time() - t0) * 1000
            details.append({
                "query": q,
                "error": str(e),
                "latency_ms": round(latency, 1),
            })
            total_latency += latency
            latencies.append(latency)

    n = len(test_queries)

    # 汇总 RAG 指标
    quality_aggregated = None
    if evaluate_quality and quality_details:
        from app.services.quality_service import aggregate_quality
        quality_aggregated = aggregate_quality(quality_details)

    p95_latency = calculate_p95(latencies)

    return BenchmarkResultV2(
        system_info=sys_info,
        total_queries=n,
        avg_latency_ms=round(total_latency / n, 1) if n else 0,
        p95_latency_ms=round(p95_latency, 1),
        avg_retrieval_ms=round(total_retrieval / n, 1) if n else 0,
        avg_generation_ms=round(total_generation / n, 1) if n else 0,
        queries_per_second=round(n / (total_latency / 1000), 2) if total_latency else 0,
        details=details,
        quality=quality_aggregated,
        quality_details=quality_details,
    )
