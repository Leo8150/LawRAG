"""法律文本专用分块器

针对中国法律条文和指导案例的结构化特征，定制分块策略：
1. 法律条文：以"条"为基本单位，保持条文完整性
2. 指导案例：按"裁判要旨/基本案情/裁判理由/裁判结果"等结构切分
3. 超长内容兜底：按款/项进行二级切分
4. 可选的上下文标头注入，提升 Embedding 质量
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import settings


# ===================== 法律条文分块器 =====================

class LegalArticleSplitter(RecursiveCharacterTextSplitter):
    """法律条文分块器 — 以"条"为基本单位"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, **kwargs):
        # 法律条文层级分隔符（优先级从高到低）
        separators = [
            r"\n第[一二三四五六七八九十百千]+编",     # 编
            r"\n第[一二三四五六七八九十百千]+章",     # 章
            r"\n第[一二三四五六七八九十百千]+节",     # 节
            r"\n第[一二三四五六七八九十\d]+条",       # 条（核心切分点）
            r"\n[一二三四五六七八九十]+、",            # 款
            r"\n（[一二三四五六七八九十]+）",          # 项
            r"\n\(\d+\)",                              # 数字项
            r"\n\d+\.",                                # 数字编号
            r"\n",                                     # 换行
        ]
        super().__init__(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=True,
            keep_separator=True,
            **kwargs,
        )


class LegalCaseSplitter(RecursiveCharacterTextSplitter):
    """指导案例分块器 — 按案例结构切分"""

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128, **kwargs):
        # 指导案例的结构分隔符
        separators = [
            r"\n(?:裁判要旨|裁判要点)",
            r"\n(?:基本案情|案情简介|案件事实)",
            r"\n(?:裁判结果|判决结果)",
            r"\n(?:裁判理由|裁判说理|法院认为)",
            r"\n(?:关键词|相关法条)",
            r"\n(?:指导案例\d+号)",
            r"\n(?:一|二|三|四|五|六|七|八|九|十)、",
            r"\n\d+\.",
            r"\n",
        ]
        super().__init__(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=True,
            keep_separator=True,
            **kwargs,
        )


# ===================== 元数据提取 =====================

def extract_law_metadata(text: str, filename: str = "") -> dict:
    """从法律条文文本中提取元数据"""
    metadata = {
        "doc_type": "law",
        "source_file": filename,
        "law_name": "",
        "effective_date": "",
    }

    # 提取法律名称（通常在文件开头或标题行）
    name_patterns = [
        r"(中华人民共和国\S+?(?:法典|法|条例|规定|办法|决定|解释))",
        r"(《[^》]+》)",
        r"^(.+(?:法典|法|条例|规定|办法|决定|解释))",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text[:500])
        if match:
            metadata["law_name"] = match.group(1).strip("《》")
            break

    if not metadata["law_name"] and filename:
        metadata["law_name"] = re.sub(r"\.\w+$", "", filename)

    # 提取生效日期
    date_patterns = [
        r"(\d{4}年\d{1,2}月\d{1,2}日)(?:起)?施行",
        r"自(\d{4}年\d{1,2}月\d{1,2}日)起施行",
        r"(\d{4}-\d{2}-\d{2})",
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text[:2000])
        if match:
            metadata["effective_date"] = match.group(1)
            break

    return metadata


def extract_case_metadata(text: str, filename: str = "") -> dict:
    """从指导案例文本中提取元数据"""
    metadata = {
        "doc_type": "case",
        "source_file": filename,
        "case_number": "",
        "case_title": "",
        "guiding_number": "",
        "keywords": "",
    }

    # 提取指导案例编号
    guide_match = re.search(r"指导案例(\d+)号", text[:500])
    if guide_match:
        metadata["guiding_number"] = f"指导案例{guide_match.group(1)}号"

    # 提取案号
    case_match = re.search(r"[（(]\d{4}[）)][^\s]*\d+号", text[:1000])
    if case_match:
        metadata["case_number"] = case_match.group(0)

    # 提取关键词（用分号拼接为字符串，ChromaDB 不支持 list 值）
    kw_match = re.search(r"关键词[：:]\s*(.+?)(?:\n|$)", text[:1000])
    if kw_match:
        keywords = re.split(r"[；;,，\s]+", kw_match.group(1).strip())
        kw_list = [kw.strip() for kw in keywords if kw.strip()]
        metadata["keywords"] = "；".join(kw_list)

    if not metadata["case_title"] and filename:
        metadata["case_title"] = re.sub(r"\.\w+$", "", filename)

    return metadata


def extract_crime_metadata(text: str, filename: str = "") -> dict:
    """Extract the first CrimeKG entry metadata for document-level display."""
    match = re.search(r"(?:^|\n)(?:=+)?【([^】]+)】(?:（([^）]+)）)?", text)
    return {
        "doc_type": "crime",
        "source_file": filename,
        "crime_name": match.group(1).strip() if match else "",
        "crime_category": match.group(2).strip() if match and match.group(2) else "",
    }


def enrich_chunk_metadata(chunk: Document, parent_metadata: dict, chunk_index: int) -> Document:
    """为分块添加详细的元数据"""
    text = chunk.page_content

    # 合并父文档元数据
    enriched = {**parent_metadata, **chunk.metadata}
    enriched["chunk_index"] = chunk_index

    # 提取当前 chunk 所属的章节
    chapter_match = re.search(r"第[一二三四五六七八九十百千]+章\s*(.+?)(?:\n|$)", text)
    if chapter_match:
        enriched["chapter"] = chapter_match.group(0).strip()

    # 提取条号
    article_match = re.search(r"第([一二三四五六七八九十百千\d]+)条", text)
    if article_match:
        enriched["article_number"] = article_match.group(1)

    # 提取案例段落类型
    section_types = {
        "裁判要旨": "裁判要旨",
        "裁判要点": "裁判要旨",
        "基本案情": "基本案情",
        "案情简介": "基本案情",
        "裁判结果": "裁判结果",
        "裁判理由": "裁判理由",
        "法院认为": "裁判理由",
    }
    for keyword, section_type in section_types.items():
        if keyword in text[:50]:
            enriched["section_type"] = section_type
            break

    chunk.metadata = enriched
    return chunk


# ===================== 上下文标头注入 =====================

def add_contextual_header(chunk: Document, parent_metadata: dict) -> Document:
    """为分块内容添加结构化上下文标头，提升 Embedding 质量

    在 chunk 内容前插入 [法律名称: X | 章节: Y | 条号: 第Z条] 形式的标头。
    """
    parts = []

    law_name = parent_metadata.get("law_name", "")
    if law_name:
        parts.append(f"法律名称: {law_name}")

    chapter = chunk.metadata.get("chapter", "")
    if chapter:
        parts.append(f"章节: {chapter}")

    article = chunk.metadata.get("article_number", "")
    if article:
        parts.append(f"条号: 第{article}条")

    # 案例类元数据
    guiding = parent_metadata.get("guiding_number", "")
    if guiding:
        parts.append(f"案例: {guiding}")

    section_type = chunk.metadata.get("section_type", "")
    if section_type:
        parts.append(f"段落: {section_type}")

    crime_name = parent_metadata.get("crime_name", "")
    if crime_name:
        parts.append(f"罪名: {crime_name}")

    knowledge_type = chunk.metadata.get("knowledge_type", "")
    if knowledge_type:
        parts.append(f"知识类型: {knowledge_type}")

    if parts:
        header = "[" + " | ".join(parts) + "]\n"
        chunk.page_content = header + chunk.page_content

    return chunk


# ===================== Parent 结构切分 =====================

_LAW_ARTICLE_START = re.compile(
    r"(?m)^(?:《[^》\n]+》)?\s*第[一二三四五六七八九十百千万零〇两\d]+条(?:之一|之二|之三)?(?:规定)?"
)
_LAW_DIVISION = re.compile(r"第[一二三四五六七八九十百千万零〇两\d]+编[^\n]*")
_LAW_CHAPTER = re.compile(r"第[一二三四五六七八九十百千万零〇两\d]+章[^\n]*")
_LAW_SECTION = re.compile(r"第[一二三四五六七八九十百千万零〇两\d]+节[^\n]*")
_CASE_SECTION_START = re.compile(
    r"(?m)^\s*(裁判要旨|裁判要点|基本案情|案情简介|案件事实|裁判结果|判决结果|裁判理由|裁判说理|法院认为|关键词|相关法条)\s*[：:]?\s*$"
)
_CRIME_ENTRY_START = re.compile(r"(?m)^(?:=+)?【([^】]+)】(?:（([^）]+)）)?\s*$")
_CRIME_SECTION_START = re.compile(
    r"(?m)^\s*(概念与定义|犯罪构成特征|认定与区分|量刑处罚|相关法条|司法解释)\s*：\s*$"
)


def _soft_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Apply a character soft limit only after a semantic unit is identified."""
    value = text.strip()
    if not value:
        return []
    if len(value) <= chunk_size:
        return [value]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=separators,
        is_separator_regex=True,
        keep_separator=True,
    )
    return [part.strip() for part in splitter.split_text(value) if part.strip()]


def _law_parent_documents(text: str, filename: str, chunk_size: int) -> list[Document]:
    base = extract_law_metadata(text, filename)
    matches = list(_LAW_ARTICLE_START.finditer(text))
    if not matches:
        parts = _soft_split(
            text,
            chunk_size,
            ["\n\n", "\n", "。", "；", "，", ""],
        )
        return [Document(page_content=part, metadata=dict(base)) for part in parts]

    documents: list[Document] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        article_text = text[match.start():end].strip()
        metadata = dict(base)
        article = re.search(r"第([一二三四五六七八九十百千万零〇两\d]+)条(?:之一|之二|之三)?", match.group(0))
        if article:
            metadata["article_number"] = article.group(1)
        divisions = list(_LAW_DIVISION.finditer(text, 0, match.start()))
        chapters = list(_LAW_CHAPTER.finditer(text, 0, match.start()))
        sections = list(_LAW_SECTION.finditer(text, 0, match.start()))
        if divisions:
            metadata["division"] = divisions[-1].group(0).strip()
        if chapters:
            metadata["chapter"] = chapters[-1].group(0).strip()
        if sections:
            metadata["section"] = sections[-1].group(0).strip()
        for part in _soft_split(
            article_text,
            chunk_size,
            [
                r"\n[一二三四五六七八九十百]+、",
                r"\n（[一二三四五六七八九十百]+）",
                r"\n\(\d+\)",
                "\n\n",
                "。",
                "；",
                "，",
                "",
            ],
        ):
            documents.append(Document(page_content=part, metadata=dict(metadata)))
    return documents


def _case_parent_documents(text: str, filename: str, chunk_size: int) -> list[Document]:
    base = extract_case_metadata(text, filename)
    matches = list(_CASE_SECTION_START.finditer(text))
    sections: list[tuple[str, str]] = []
    if matches:
        prefix = text[:matches[0].start()].strip()
        if prefix:
            sections.append(("案例信息", prefix))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section_name = match.group(1)
            normalized = {
                "裁判要点": "裁判要旨",
                "案情简介": "基本案情",
                "案件事实": "基本案情",
                "判决结果": "裁判结果",
                "裁判说理": "裁判理由",
                "法院认为": "裁判理由",
            }.get(section_name, section_name)
            sections.append((normalized, text[match.start():end].strip()))
    else:
        sections.append(("案件事实", text))

    documents: list[Document] = []
    for section_type, section_text in sections:
        metadata = {**base, "section_type": section_type}
        for part in _soft_split(
            section_text,
            chunk_size,
            ["\n\n", "\n", "。", "；", "，", ""],
        ):
            documents.append(Document(page_content=part, metadata=dict(metadata)))
    return documents


def _crime_parent_documents(text: str, filename: str, chunk_size: int) -> list[Document]:
    entries = list(_CRIME_ENTRY_START.finditer(text))
    if not entries:
        base = extract_crime_metadata(text, filename)
        return [
            Document(page_content=part, metadata=dict(base))
            for part in _soft_split(text, chunk_size, ["\n\n", "\n", "。", "；", "，", ""])
        ]

    documents: list[Document] = []
    for entry_index, entry in enumerate(entries):
        entry_end = entries[entry_index + 1].start() if entry_index + 1 < len(entries) else len(text)
        crime_name = entry.group(1).strip()
        crime_category = (entry.group(2) or "").strip()
        entry_text = text[entry.end():entry_end]
        sections = list(_CRIME_SECTION_START.finditer(entry_text))
        if not sections:
            sections_and_text = [("罪名知识", entry_text)]
        else:
            sections_and_text = []
            for section_index, section in enumerate(sections):
                section_end = sections[section_index + 1].start() if section_index + 1 < len(sections) else len(entry_text)
                sections_and_text.append((
                    section.group(1),
                    entry_text[section.start():section_end].strip(),
                ))
        for knowledge_type, section_text in sections_and_text:
            metadata = {
                "doc_type": "crime",
                "source_file": filename,
                "crime_name": crime_name,
                "crime_category": crime_category,
                "knowledge_type": knowledge_type,
                "section_type": knowledge_type,
            }
            for part in _soft_split(
                section_text,
                chunk_size,
                ["\n\n", "\n", "。", "；", "，", ""],
            ):
                documents.append(Document(page_content=part, metadata=dict(metadata)))
    return documents


# ===================== 统一处理入口 =====================

def split_legal_document(
    text: str,
    doc_type: str = "law",
    filename: str = "",
    chunk_size: int | None = None,
    chunk_overlap: int = 0,
) -> list[Document]:
    """
    统一的法律文档分块入口

    Args:
        text: 文档全文
        doc_type: 文档类型 "law" | "case" | "crime"
        filename: 源文件名
        chunk_size: 分块大小
        chunk_overlap: 分块重叠

    Returns:
        带有丰富元数据的 Document 列表
    """
    del chunk_overlap  # Parent 之间不重叠；重叠仅用于 Parent 内的 Child。
    if doc_type == "case":
        parent_size = chunk_size or settings.CASE_PARENT_CHUNK_SIZE
        chunks = _case_parent_documents(text, filename, parent_size)
        parent_metadata = extract_case_metadata(text, filename)
    elif doc_type == "crime":
        parent_size = chunk_size or settings.CRIME_PARENT_CHUNK_SIZE
        chunks = _crime_parent_documents(text, filename, parent_size)
        parent_metadata = extract_crime_metadata(text, filename)
    else:
        parent_size = chunk_size or settings.LAW_PARENT_CHUNK_SIZE
        chunks = _law_parent_documents(text, filename, parent_size)
        parent_metadata = extract_law_metadata(text, filename)

    # 元数据增强
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        enriched = enrich_chunk_metadata(chunk, parent_metadata, i)
        enriched_chunks.append(enriched)

    # 可选：注入上下文标头
    if settings.CONTEXTUAL_CHUNKING:
        for chunk in enriched_chunks:
            add_contextual_header(chunk, chunk.metadata)

    return enriched_chunks
