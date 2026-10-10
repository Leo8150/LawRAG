"""Transactional MySQL ingestion followed by idempotent child vector indexing."""

from __future__ import annotations

from app.core.vectorstore import add_child_documents
from app.db.repository import ParentChildRepository, get_parent_child_repository
from app.utils.parent_child_chunker import ParentChildTree, build_parent_child_tree


def prepare_document(
    text: str,
    *,
    doc_type: str,
    source_file: str,
    source_path: str = "",
    dataset_name: str = "",
    source_record_id: str = "",
) -> ParentChildTree:
    return build_parent_child_tree(
        text,
        doc_type=doc_type,
        source_file=source_file,
        source_path=source_path,
        dataset_name=dataset_name,
        source_record_id=source_record_id,
    )


def persist_and_index(
    tree: ParentChildTree,
    *,
    collection_name: str,
    repository: ParentChildRepository | None = None,
) -> ParentChildTree:
    """Persist source data first, then build the derived vector index."""
    repository = repository or get_parent_child_repository()
    repository.save_tree(tree.document)
    try:
        add_child_documents(collection_name, tree.vector_documents, tree.child_ids)
        repository.update_status(tree.document.doc_id, "indexed")
    except Exception:
        repository.update_status(tree.document.doc_id, "failed")
        raise
    return tree
