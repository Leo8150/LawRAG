"""Child-level hybrid retrieval with authoritative MySQL hydration."""

import jieba
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi
from app.core.vectorstore import get_vectorstore
from app.config import settings
from app.db.repository import ParentChildRepository, get_parent_child_repository


class BM25ChineseRetriever(BaseRetriever):
    """基于 jieba 分词的 BM25 中文检索器"""

    documents: list[Document] = []
    tokenized_corpus: list[list[str]] = []
    bm25: BM25Okapi | None = None
    k: int = 10

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, documents: list[Document], k: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.documents = documents
        self.k = k
        # jieba 分词构建语料库
        self.tokenized_corpus = [
            list(jieba.cut(doc.page_content)) for doc in documents
        ]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if not self.bm25 or not self.documents:
            return []
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        # 获取 top-k 索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            : self.k
        ]
        return [self.documents[i] for i in top_indices if scores[i] > 0]


class HybridRetriever(BaseRetriever):
    """Retrieve child IDs, fuse ranks, then batch-hydrate text from MySQL."""

    bm25_retriever: BM25ChineseRetriever | None = None
    collection_names: list[str] = ["laws", "cases"]
    bm25_weight: float = settings.BM25_WEIGHT
    vector_weight: float = settings.VECTOR_WEIGHT
    k: int = settings.RETRIEVAL_TOP_K
    all_documents: list[Document] = []
    repository: object | None = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        collection_names: list[str] | None = None,
        k: int | None = None,
        repository: ParentChildRepository | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if collection_names:
            self.collection_names = collection_names
        if k is not None:
            self.k = k
        self.repository = repository or get_parent_child_repository()
        self._load_bm25_corpus()

    def _load_bm25_corpus(self):
        """Load authoritative child text from MySQL for the sparse index."""
        doc_types = []
        if settings.LAWS_COLLECTION in self.collection_names:
            doc_types.append("law")
        if settings.CASES_COLLECTION in self.collection_names:
            doc_types.append("case")
        all_docs = self.repository.list_children(doc_types)  # type: ignore[union-attr]
        self.all_documents = all_docs
        if all_docs:
            self.bm25_retriever = BM25ChineseRetriever(documents=all_docs, k=self.k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """RRF by child_chunk_id, followed by one MySQL batch lookup."""
        return self._search(query, query)

    def _search(self, bm25_query: str, vector_query: str) -> list[Document]:
        results_map: dict[str, float] = {}
        rrf_k = 60  # RRF 常数

        # BM25 检索
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.invoke(bm25_query)
            for rank, doc in enumerate(bm25_results):
                key = str(doc.metadata.get("child_chunk_id", ""))
                if not key:
                    continue
                score = self.bm25_weight / (rrf_k + rank + 1)
                results_map[key] = results_map.get(key, 0.0) + score

        # 向量检索
        for name in self.collection_names:
            try:
                vec_results = get_vectorstore(name).similarity_search(vector_query, k=self.k)
                for rank, doc in enumerate(vec_results):
                    key = str(doc.metadata.get("child_chunk_id", ""))
                    if not key:
                        continue
                    score = self.vector_weight / (rrf_k + rank + 1)
                    results_map[key] = results_map.get(key, 0.0) + score
            except Exception:
                continue

        ranked_ids = [
            child_id for child_id, _ in
            sorted(results_map.items(), key=lambda item: item[1], reverse=True)[: self.k]
        ]
        hydrated = self.repository.fetch_children(ranked_ids)  # type: ignore[union-attr]
        for doc in hydrated:
            child_id = str(doc.metadata.get("child_chunk_id", ""))
            doc.metadata["rrf_score"] = round(results_map.get(child_id, 0.0), 8)
        return hydrated

    def search_with_split_queries(
        self, bm25_query: str, vector_query: str
    ) -> list[Document]:
        """分离式检索：BM25 使用一个查询，向量检索使用另一个查询

        用于 HyDE 场景：bm25_query 为原始问题，vector_query 为假设文档。
        """
        return self._search(bm25_query, vector_query)


def get_hybrid_retriever(
    collection_names: list[str] | None = None,
    k: int | None = None,
    repository: ParentChildRepository | None = None,
) -> HybridRetriever:
    """获取混合检索器"""
    names = collection_names or ["laws", "cases"]
    return HybridRetriever(collection_names=names, k=k, repository=repository)
