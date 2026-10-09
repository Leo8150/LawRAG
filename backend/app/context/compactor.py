"""Deterministic four-stage context compaction for legal evidence.

The compactor deliberately performs no model calls.  Statute text is only
extractively clipped; it is never rewritten, which keeps quoted legal evidence
traceable to the source document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from app.utils.metadata import format_source_display


def estimate_tokens(text: str) -> int:
    """Estimate mixed Chinese/ASCII token usage without loading a tokenizer."""
    if not text:
        return 0
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    other = max(0, len(text) - chinese)
    return chinese + (other + 3) // 4


def _clip_exact(text: str, token_limit: int) -> str:
    """Extractively clip text while retaining both the beginning and ending."""
    if estimate_tokens(text) <= token_limit:
        return text.strip()
    char_limit = max(120, token_limit)
    head = int(char_limit * 0.78)
    tail = max(40, char_limit - head)
    return f"{text[:head].rstrip()}\n……（中间内容按上下文预算裁剪）……\n{text[-tail:].lstrip()}"


def _extractive_summary(text: str, token_limit: int) -> str:
    """Build an extractive summary from original sentences, without an LLM."""
    sentences = [s.strip() for s in re.split(r"(?<=[。！？；\n])", text) if s.strip()]
    if not sentences:
        return _clip_exact(text, token_limit)
    preferred = ("裁判", "法院", "认为", "判决", "争议", "构成", "处罚", "要件", "事实")
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (not any(k in item[1] for k in preferred), item[0]),
    )
    chosen: list[tuple[int, str]] = []
    used = 0
    for index, sentence in ranked:
        size = estimate_tokens(sentence)
        if chosen and used + size > token_limit:
            continue
        chosen.append((index, sentence))
        used += size
        if used >= token_limit:
            break
    return "".join(sentence for _, sentence in sorted(chosen)).strip()


@dataclass
class CompactionResult:
    context: str
    documents: list[Document]
    token_budget: int
    tokens_before: int
    tokens_after: int
    stages: list[str] = field(default_factory=lambda: ["budget", "snip", "micro", "summary"])

    def to_dict(self) -> dict:
        return {
            "token_budget": self.token_budget,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "stages": self.stages,
            "document_count": len(self.documents),
        }


class FourStageContextCompactor:
    """Budget -> Snip -> Micro -> Summary context compaction."""

    DEFAULT_PRIORITY = {"kg": 0, "law": 1, "statute": 1, "case": 2}
    PER_DOCUMENT_BUDGET = {"law": 1400, "statute": 1400, "case": 900, "kg": 700}

    def __init__(self, max_context_tokens: int = 4000):
        self.max_context_tokens = max_context_tokens

    def compact(
        self,
        documents: list[Document],
        *,
        question: str = "",
        memory_context: str = "",
        skill_instructions: str = "",
        priority: dict[str, int] | None = None,
    ) -> CompactionResult:
        # 1. Budget: reserve room already occupied by question, memory and skill.
        reserved = estimate_tokens(question) + estimate_tokens(memory_context) + estimate_tokens(skill_instructions)
        budget = max(256, self.max_context_tokens - reserved)
        tokens_before = sum(estimate_tokens(d.page_content) for d in documents)

        # 2. Snip: cap a single document so it cannot consume the whole prompt.
        snipped: list[Document] = []
        for doc in documents:
            doc_type = str(doc.metadata.get("doc_type", "other"))
            limit = self.PER_DOCUMENT_BUDGET.get(doc_type, 700)
            snipped.append(Document(
                page_content=_clip_exact(doc.page_content, limit),
                metadata=dict(doc.metadata),
            ))

        # 3. Micro: remove duplicated passages and sort by legal evidence priority.
        unique: list[Document] = []
        seen: set[str] = set()
        for doc in snipped:
            key = re.sub(r"\s+", "", doc.page_content)[:300]
            if key and key not in seen:
                seen.add(key)
                unique.append(doc)
        priorities = {**self.DEFAULT_PRIORITY, **(priority or {})}
        unique.sort(key=lambda d: (
            priorities.get(str(d.metadata.get("doc_type", "other")), 9),
            -float(d.metadata.get("rerank_score", 0) or 0),
        ))

        # 4. Summary: pack evidence; compact cases/KG extractively when necessary.
        packed: list[Document] = []
        used = 0
        for doc in unique:
            label = format_source_display(doc.metadata)
            overhead = estimate_tokens(label) + 8
            remaining = budget - used - overhead
            if remaining < 80:
                break
            content = doc.page_content
            size = estimate_tokens(content)
            if size > remaining:
                doc_type = str(doc.metadata.get("doc_type", "other"))
                content = _clip_exact(content, remaining) if doc_type in {"law", "statute"} else _extractive_summary(content, remaining)
            if not content:
                continue
            compacted = Document(page_content=content, metadata=dict(doc.metadata))
            packed.append(compacted)
            used += estimate_tokens(content) + overhead

        if not packed:
            context = "未找到相关参考资料。"
        else:
            parts = []
            for index, doc in enumerate(packed, 1):
                parts.append(f"[来源{index}] {format_source_display(doc.metadata)}\n{doc.page_content}")
            context = "\n\n".join(parts)

        return CompactionResult(
            context=context,
            documents=packed,
            token_budget=budget,
            tokens_before=tokens_before,
            tokens_after=estimate_tokens(context),
        )
