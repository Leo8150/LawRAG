"""Offline tests for MySQL-style parent-child storage and retrieval."""

from langchain_core.documents import Document


def test_differentiated_chunk_profiles_match_data_types():
    from app.utils.parent_child_chunker import get_chunk_profile

    law = get_chunk_profile("law")
    case = get_chunk_profile("case")
    crime = get_chunk_profile("crime")

    assert (law.parent_size, law.child_size, law.child_overlap) == (1024, 320, 64)
    assert (case.parent_size, case.child_size, case.child_overlap) == (2048, 512, 96)
    assert (crime.parent_size, crime.child_size, crime.child_overlap) == (768, 256, 48)


def test_law_uses_article_parents_and_small_children():
    from app.utils.parent_child_chunker import build_parent_child_tree

    long_clause = "当事人应当依法履行义务，" * 100
    text = f"""中华人民共和国示例法
第一章 总则
第一条 {long_clause}
第二条 本条保持为独立法条。"""
    tree = build_parent_child_tree(text, doc_type="law", source_file="示例法.txt")

    assert {parent.article_number for parent in tree.document.parents} >= {"一", "二"}
    assert all(len(parent.parent_text) <= 1024 for parent in tree.document.parents)
    children = [child for parent in tree.document.parents for child in parent.children]
    assert all(len(child.child_text) <= 320 for child in children)
    assert all(child.child_metadata["child_chunk_overlap"] == 64 for child in children)
    assert "法律名称:" in tree.vector_documents[0].page_content


def test_case_uses_semantic_section_parents_and_larger_children():
    from app.utils.parent_child_chunker import build_parent_child_tree

    text = """指导案例99号

基本案情
被告人实施了相关行为。案发后主动到案。双方对事实没有争议。

裁判理由
法院结合证据、行为方式和法律规定进行认定。

裁判结果
法院依法作出判决。"""
    tree = build_parent_child_tree(text, doc_type="case", source_file="指导案例99号.txt")

    section_types = {parent.section_type for parent in tree.document.parents}
    assert {"基本案情", "裁判理由", "裁判结果"}.issubset(section_types)
    children = [child for parent in tree.document.parents for child in parent.children]
    assert all(child.child_metadata["child_chunk_size"] == 512 for child in children)
    assert all(child.child_metadata["child_chunk_overlap"] == 96 for child in children)


def test_crime_kg_uses_field_parents_and_context_headers():
    from app.utils.parent_child_chunker import build_parent_child_tree

    text = """【盗窃罪】（侵犯财产罪）

概念与定义：
盗窃罪是以非法占有为目的秘密窃取财物的行为。

犯罪构成特征：
本罪应结合主体、主观方面、客体和客观方面进行判断。

量刑处罚：
根据盗窃数额和犯罪情节依法处罚。

【诈骗罪】（侵犯财产罪）

概念与定义：
诈骗罪是以非法占有为目的骗取财物的行为。"""
    tree = build_parent_child_tree(text, doc_type="crime", source_file="犯罪知识图谱.txt")

    parent_metadata = [parent.chunk_metadata for parent in tree.document.parents]
    assert {item["crime_name"] for item in parent_metadata} == {"盗窃罪", "诈骗罪"}
    assert "概念与定义" in {item["knowledge_type"] for item in parent_metadata}
    children = [child for parent in tree.document.parents for child in parent.children]
    assert all(len(child.child_text) <= 256 for child in children)
    assert all(child.child_metadata["child_chunk_overlap"] == 48 for child in children)
    assert any("罪名: 盗窃罪" in item.page_content for item in tree.vector_documents)


def test_parent_child_ids_are_stable_and_linked():
    from app.utils.parent_child_chunker import build_parent_child_tree

    text = """中华人民共和国刑法

第二百六十四条 盗窃公私财物，数额较大的，依法追究刑事责任。
入户盗窃、多次盗窃、携带凶器盗窃的，依照本条规定处罚。
"""
    first = build_parent_child_tree(text, doc_type="law", source_file="刑法.txt")
    second = build_parent_child_tree(text, doc_type="law", source_file="刑法.txt")

    assert first.document.doc_id == second.document.doc_id
    assert first.child_ids == second.child_ids
    assert first.document.parents
    assert first.document.parents[0].children
    child = first.document.parents[0].children[0]
    assert child.parent_chunk_id == first.document.parents[0].parent_chunk_id
    assert child.child_chunk_id in first.child_ids
    assert first.vector_documents[0].metadata["child_chunk_id"] == child.child_chunk_id


def test_repository_hydrates_children_then_parents_from_sqlite():
    from app.db.repository import ParentChildRepository
    from app.utils.parent_child_chunker import build_parent_child_tree

    repository = ParentChildRepository("sqlite+pysqlite:///:memory:", create_schema=True)
    tree = build_parent_child_tree(
        "中华人民共和国劳动合同法\n第一条 为了完善劳动合同制度，保护劳动者合法权益。",
        doc_type="law",
        source_file="劳动合同法.txt",
    )
    repository.save_tree(tree.document)

    children = repository.fetch_children(tree.child_ids)
    assert len(children) == len(tree.child_ids)
    assert children[0].metadata["child_chunk_id"] == tree.child_ids[0]
    assert "劳动合同" in children[0].page_content

    parent_id = children[0].metadata["parent_chunk_id"]
    parents = repository.fetch_parents([parent_id])
    assert len(parents) == 1
    assert parents[0].metadata["doc_id"] == tree.document.doc_id
    assert parents[0].metadata["source_file"] == "劳动合同法.txt"

    source = repository.get_document_source(tree.document.doc_id)
    assert source is not None
    assert source["raw_content"].startswith("中华人民共和国劳动合同法")


def test_hybrid_retriever_fuses_child_ids_and_hydrates_mysql(monkeypatch):
    from app.core import retriever as retriever_module

    child_a = Document(
        page_content="MySQL中的盗窃子块正文",
        metadata={"child_chunk_id": "child-a", "parent_chunk_id": "parent-a"},
    )
    child_b = Document(
        page_content="MySQL中的合同子块正文",
        metadata={"child_chunk_id": "child-b", "parent_chunk_id": "parent-b"},
    )

    class FakeRepository:
        def list_children(self, _doc_types):
            return [child_a, child_b]

        def fetch_children(self, child_ids):
            mapping = {"child-a": child_a, "child-b": child_b}
            return [mapping[item] for item in child_ids if item in mapping]

    class FakeVectorStore:
        def similarity_search(self, _query, k):
            return [Document(page_content="非权威副本", metadata={"child_chunk_id": "child-a"})]

    monkeypatch.setattr(retriever_module, "get_vectorstore", lambda _name: FakeVectorStore())
    retriever = retriever_module.HybridRetriever(
        collection_names=["laws"],
        k=2,
        repository=FakeRepository(),
    )
    results = retriever.invoke("盗窃")

    assert results
    assert results[0].metadata["child_chunk_id"] == "child-a"
    assert results[0].page_content == "MySQL中的盗窃子块正文"


def test_parent_aggregation_uses_child_scores_without_parent_reranker():
    from app.services.pipeline import PipelineConfig, RAGPipeline

    class FakeRepository:
        def fetch_parents(self, parent_ids):
            return [
                Document(page_content=f"父块 {parent_id}", metadata={"parent_chunk_id": parent_id})
                for parent_id in parent_ids
            ]

    pipeline = RAGPipeline(PipelineConfig(top_k=2), repository=FakeRepository())
    children = [
        Document(page_content="A1", metadata={"child_chunk_id": "a1", "parent_chunk_id": "A", "rerank_score": 0.91}),
        Document(page_content="A2", metadata={"child_chunk_id": "a2", "parent_chunk_id": "A", "rerank_score": 0.88}),
        Document(page_content="B1", metadata={"child_chunk_id": "b1", "parent_chunk_id": "B", "rerank_score": 0.90}),
    ]

    parents = pipeline._aggregate_and_hydrate_parents(children)
    assert [item.metadata["parent_chunk_id"] for item in parents] == ["A", "B"]
    assert parents[0].metadata["parent_score"] > parents[1].metadata["parent_score"]
    assert parents[0].metadata["matched_child_count"] == 2
