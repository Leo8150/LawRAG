"""本地轻量重排序服务。

`simple_rerank` 使用 jieba 分词后的 Jaccard 相似度与法律元数据加权，
既可作为零云调用方案，也可作为云端 Reranker 的降级路径。
"""

import jieba
from langchain_core.documents import Document


def simple_rerank(
    query: str,
    documents: list[Document],
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """基于关键词重叠与元数据加分执行轻量重排序。"""
    query_tokens = set(jieba.cut(query))

    scored: list[tuple[Document, float]] = []
    for doc in documents:
        doc_tokens = set(jieba.cut(doc.page_content))

        intersection = query_tokens & doc_tokens
        union = query_tokens | doc_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        metadata = doc.metadata
        bonus = 0.0
        law_name = metadata.get("law_name", "")
        if law_name and law_name in query:
            bonus += 0.2
        article_number = metadata.get("article_number", "")
        if article_number and article_number in query:
            bonus += 0.3
        guiding_number = metadata.get("guiding_number", "")
        if guiding_number and guiding_number in query:
            bonus += 0.2

        scored.append((doc, jaccard + bonus))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
