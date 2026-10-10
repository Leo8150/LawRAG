"""知识库管理服务"""

import os
import json
from pathlib import Path
from langchain_core.documents import Document
from app.core.vectorstore import delete_child_vectors, reset_store_cache
from app.db.repository import get_parent_child_repository
from app.services.ingestion_service import persist_and_index, prepare_document
from app.models.schemas import KnowledgeFileInfo, KnowledgeStats
from app.config import settings


# ===================== 文件读取 =====================

def _read_text_file(filepath: str) -> str:
    """读取文本文件"""
    encodings = ["utf-8", "gbk", "gb2312", "utf-16"]
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法解码文件: {filepath}")


def _detect_doc_type(filepath: str) -> str:
    """根据路径判断文档类型"""
    path_lower = filepath.lower()
    if "case" in path_lower or "案例" in path_lower:
        return "case"
    return "law"


def _collect_files(dir_path: str, extensions: set[str]) -> list[str]:
    """递归收集目录下指定后缀的文件"""
    results = []
    if not os.path.isdir(dir_path):
        return results
    for root, _dirs, files in os.walk(dir_path):
        for fname in sorted(files):
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in extensions:
                results.append(os.path.join(root, fname))
    return results


def _count_files(dir_path: str) -> int:
    """递归统计目录下的文件数"""
    count = 0
    if not os.path.isdir(dir_path):
        return 0
    for root, _dirs, files in os.walk(dir_path):
        count += sum(1 for f in files if not f.startswith("."))
    return count


# ===================== 知识库统计 =====================

def get_kb_stats() -> KnowledgeStats:
    """获取知识库统计信息"""
    stats = KnowledgeStats()
    try:
        counts = get_parent_child_repository().child_counts_by_doc_type()
        stats.laws_chunks = counts.get("law", 0)
        stats.cases_chunks = counts.get("case", 0)
        stats.total_chunks = stats.laws_chunks + stats.cases_chunks
        if stats.laws_chunks:
            stats.collections.append(settings.LAWS_COLLECTION)
        if stats.cases_chunks:
            stats.collections.append(settings.CASES_COLLECTION)
    except Exception:
        pass

    stats.total_files = _count_files(settings.LAWS_DIR) + _count_files(settings.CASES_DIR)
    return stats


# ===================== 上传文档 =====================

async def upload_document(
    filename: str,
    content: bytes,
    doc_type: str | None = None,
) -> KnowledgeFileInfo:
    """上传并索引一个文档"""
    # 确定类型和目标目录
    if doc_type is None:
        doc_type = _detect_doc_type(filename)
    target_dir = settings.CASES_DIR if doc_type == "case" else settings.LAWS_DIR
    os.makedirs(target_dir, exist_ok=True)

    # 保存文件
    filepath = os.path.join(target_dir, filename)
    with open(filepath, "wb") as f:
        f.write(content)

    # MySQL 保存原文与父子块，ChromaDB 只索引 Child。
    text = _read_text_file(filepath)
    tree = prepare_document(
        text,
        doc_type=doc_type,
        source_file=filename,
        source_path=filepath,
        dataset_name="upload",
    )
    collection_name = settings.CASES_COLLECTION if doc_type == "case" else settings.LAWS_COLLECTION
    repository = get_parent_child_repository()
    repository.create_schema()
    persist_and_index(tree, collection_name=collection_name, repository=repository)

    return KnowledgeFileInfo(
        filename=filename,
        doc_type=doc_type,
        size_bytes=len(content),
        chunk_count=len(tree.child_ids),
    )


# ===================== 列出知识库文件 =====================

def list_documents() -> list[KnowledgeFileInfo]:
    """列出所有已上传的文档（递归子目录）"""
    files: list[KnowledgeFileInfo] = []
    for dir_path, dtype, exts in [
        (settings.LAWS_DIR, "law", {".txt", ".md"}),
        (settings.CASES_DIR, "case", {".txt", ".md", ".json"}),
    ]:
        for fpath in _collect_files(dir_path, exts):
            fname = os.path.relpath(fpath, dir_path)
            files.append(KnowledgeFileInfo(
                filename=fname,
                doc_type=dtype,
                size_bytes=os.path.getsize(fpath),
            ))
    return files


# ===================== 删除文档 =====================

def delete_document(filename: str) -> bool:
    """Delete source file, its MySQL tree and all derived child vectors."""
    repository = get_parent_child_repository()
    for doc_id, doc_type in repository.document_ids_by_source_file(os.path.basename(filename)):
        child_ids = repository.child_ids_for_document(doc_id)
        collection = settings.CASES_COLLECTION if doc_type == "case" else settings.LAWS_COLLECTION
        delete_child_vectors(collection, child_ids)
        repository.delete_document(doc_id)

    removed = False
    for dir_path in [settings.LAWS_DIR, settings.CASES_DIR]:
        fpath = os.path.join(dir_path, filename)
        if os.path.isfile(fpath):
            os.remove(fpath)
            removed = True
    return removed


# ===================== 重建索引 =====================

async def rebuild_index() -> KnowledgeStats:
    """Rebuild MySQL parent-child data and derived Child vector indexes."""
    import chromadb
    from scripts.import_data import import_cases_jsonl, import_laws

    # 清除现有数据
    reset_store_cache()
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    for name in [settings.LAWS_COLLECTION, settings.CASES_COLLECTION]:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    repository = get_parent_child_repository()
    repository.create_schema()
    repository.reset_all()
    reset_store_cache()
    import_laws(repository, settings.LAWS_DIR)
    import_cases_jsonl(repository, settings.CASES_DIR, max_cases=5000)
    reset_store_cache()
    return get_kb_stats()
