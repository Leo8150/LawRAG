"""运行 LawRAG 四项核心评测并保存 JSON 报告。

指标：Recall@5、MRR@10、P95 Latency、Faithfulness。
注意：执行本脚本会运行完整问答链路并调用 Faithfulness Judge。
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.perf_service import run_benchmark


def load_eval_queries() -> list[str]:
    dataset_path = Path(settings.DATA_DIR) / "rag_eval_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [record["question"] for record in records if record.get("question")]


async def main() -> None:
    queries = load_eval_queries()
    result = await run_benchmark(
        queries=queries,
        use_rerank=True,
        evaluate_quality=True,
    )

    quality = result.quality
    print("=" * 52)
    print(f"Recall@5:       {quality.recall_at_5:.4f}" if quality else "Recall@5:       N/A")
    print(f"MRR@10:         {quality.mrr_at_10:.4f}" if quality else "MRR@10:         N/A")
    print(f"P95 Latency:    {result.p95_latency_ms:.1f} ms")
    print(
        f"Faithfulness:   {quality.avg_faithfulness:.2f} / 10"
        if quality else "Faithfulness:   N/A"
    )
    print(f"评测问题数:      {result.total_queries}")

    output_path = Path(settings.REPORTS_DIR) / "rag_eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    asyncio.run(main())
