"""ChromaDB 向量数据库"""

from langchain_community.vectorstores import Chroma
from app.core.embeddings import get_embeddings
from app.config import settings

# 缓存已加载的向量库实例
_store_cache: dict[str, Chroma] = {}


def get_vectorstore(collection_name: str) -> Chroma:
    """获取（或创建）指定 collection 的 Chroma 向量库"""
    if collection_name in _store_cache:
        return _store_cache[collection_name]

    store = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    _store_cache[collection_name] = store
    return store


def reset_store_cache():
    """清空向量库缓存（在重建索引后调用）"""
    _store_cache.clear()


def add_child_documents(collection_name: str, documents: list, child_ids: list[str]) -> None:
    """Index child chunks with child_chunk_id as the vector record ID."""
    if len(documents) != len(child_ids):
        raise ValueError("documents 与 child_ids 数量不一致")
    if documents:
        get_vectorstore(collection_name).add_documents(documents, ids=child_ids)


def delete_child_vectors(collection_name: str, child_ids: list[str]) -> None:
    if child_ids:
        get_vectorstore(collection_name).delete(ids=child_ids)
