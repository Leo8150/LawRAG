"""Curated legal MCP tools exposed to the LawRAG agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from app.config import settings
from app.mcp.client import MCPHttpClient, MCPServer


STATUTE_SEARCH = "search_statutes"
STATUTE_ARTICLES = "get_statute_articles"
CASE_SEARCH = "search_court_cases"
CASE_DETAIL = "get_court_case"
WEB_SEARCH = "search_official_legal_web"
WEB_FETCH = "fetch_official_legal_page"


@dataclass
class ExternalToolResult:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def redact_sensitive_query(value: str) -> str:
    """Remove common Chinese personal identifiers before an external request."""
    value = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号]", value)
    value = re.sub(
        r"(?<![0-9A-Za-z])\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
        r"(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?![0-9A-Za-z])",
        "[身份证号]",
        value,
    )
    return value.strip()


class ExternalLegalMCPService:
    """Map stable LawRAG tool names to provider-specific MCP tools."""

    def __init__(self, client: MCPHttpClient | None = None):
        self.client = client or MCPHttpClient()
        self.flk = MCPServer(
            name="flk-npc",
            url=settings.FLK_MCP_URL,
            timeout_seconds=settings.MCP_TIMEOUT_SECONDS,
        )
        self.cases = MCPServer(
            name="rmfyalk",
            url=settings.RMFYALK_MCP_URL,
            timeout_seconds=settings.MCP_TIMEOUT_SECONDS,
        )
        tavily_headers = {
            "Authorization": f"Bearer {settings.TAVILY_API_KEY}",
            "DEFAULT_PARAMETERS": '{"include_images":false,"search_depth":"basic"}',
        }
        self.tavily = MCPServer(
            name="tavily",
            url=settings.TAVILY_MCP_URL,
            headers=tavily_headers,
            timeout_seconds=settings.MCP_TIMEOUT_SECONDS,
        )

    @property
    def enabled_tools(self) -> set[str]:
        if not settings.EXTERNAL_MCP_ENABLED:
            return set()
        names: set[str] = set()
        if settings.FLK_MCP_ENABLED:
            names.update({STATUTE_SEARCH, STATUTE_ARTICLES})
        if settings.RMFYALK_MCP_ENABLED:
            names.update({CASE_SEARCH, CASE_DETAIL})
        if settings.TAVILY_MCP_ENABLED and settings.TAVILY_API_KEY:
            names.update({WEB_SEARCH, WEB_FETCH})
        return names

    async def execute(self, name: str, arguments: dict[str, Any]) -> ExternalToolResult:
        if name not in self.enabled_tools:
            raise ValueError(f"External MCP tool is disabled or not configured: {name}")
        dispatch = {
            STATUTE_SEARCH: self._search_statutes,
            STATUTE_ARTICLES: self._get_statute_articles,
            CASE_SEARCH: self._search_cases,
            CASE_DETAIL: self._get_case,
            WEB_SEARCH: self._search_web,
            WEB_FETCH: self._fetch_web,
        }
        return await dispatch[name](arguments)

    async def _search_statutes(self, args: dict[str, Any]) -> ExternalToolResult:
        query = redact_sensitive_query(str(args.get("query", "")))
        page_size = _bounded_int(args.get("max_results"), 5, 1, 10)
        params = {
            "search_content": query,
            "search_type": 2,
            "search_range": 1 if bool(args.get("title_only", False)) else 2,
            "page_num": 1,
            "page_size": page_size,
        }
        if bool(args.get("effective_only", True)):
            params["sxx"] = 3
        content = await self.client.call_tool(self.flk, "flk_search", {"params": params})
        return self._result(content, "国家法律法规数据库", self.flk, query=query)

    async def _get_statute_articles(self, args: dict[str, Any]) -> ExternalToolResult:
        regulation_id = str(args.get("regulation_id", "")).strip()
        keyword = redact_sensitive_query(str(args.get("keyword", "")))
        params = {
            "bbbs": regulation_id,
            "search_content": keyword,
            "search_type": 2,
            "search_range": 2,
        }
        content = await self.client.call_tool(self.flk, "flk_hit_display", {"params": params})
        return self._result(
            content,
            "国家法律法规数据库",
            self.flk,
            regulation_id=regulation_id,
            query=keyword,
        )

    async def _search_cases(self, args: dict[str, Any]) -> ExternalToolResult:
        query = redact_sensitive_query(str(args.get("query", "")))
        params = {
            "keyword": query,
            "search_field": "qw",
            "page": 1,
            "page_size": _bounded_int(args.get("max_results"), 5, 1, 10),
        }
        content = await self.client.call_tool(self.cases, "rmfyalk_search", {"params": params})
        return self._result(content, "人民法院案例库", self.cases, query=query)

    async def _get_case(self, args: dict[str, Any]) -> ExternalToolResult:
        case_id = str(args.get("case_id", "")).strip()
        sections = args.get("sections") or ["key_points", "case_facts", "reasoning", "laws"]
        content = await self.client.call_tool(
            self.cases,
            "rmfyalk_get_case",
            {"params": {"case_id": case_id, "sections": list(sections)[:6]}},
        )
        return self._result(content, "人民法院案例库", self.cases, case_id=case_id)

    async def _search_web(self, args: dict[str, Any]) -> ExternalToolResult:
        query = redact_sensitive_query(str(args.get("query", "")))
        max_results = _bounded_int(args.get("max_results"), 5, 1, 8)
        provider_args = {
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_domains": list(settings.TAVILY_LEGAL_DOMAINS),
            "include_raw_content": False,
            "include_images": False,
        }
        content = await self.client.call_tool(self.tavily, "tavily-search", provider_args)
        return self._result(content, "官方法律网站检索", self.tavily, query=query)

    async def _fetch_web(self, args: dict[str, Any]) -> ExternalToolResult:
        url = str(args.get("url", "")).strip()
        host = (urlparse(url).hostname or "").lower()
        if not _domain_allowed(host, settings.TAVILY_LEGAL_DOMAINS):
            raise ValueError("Only allow-listed official legal domains may be fetched")
        content = await self.client.call_tool(
            self.tavily,
            "tavily-extract",
            {"urls": [url], "extract_depth": "basic", "include_images": False},
        )
        return self._result(content, host, self.tavily, source_url=url)

    @staticmethod
    def _result(content: str, title: str, server: MCPServer, **metadata: Any) -> ExternalToolResult:
        limited = content[: settings.MCP_TOOL_RESULT_MAX_CHARS]
        return ExternalToolResult(content=limited, metadata={
            "doc_type": "external_legal_source",
            "title": title,
            "source_file": title,
            "mcp_server": server.name,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        })


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        return min(maximum, max(minimum, int(value)))
    except (TypeError, ValueError):
        return default


def _domain_allowed(host: str, domains: list[str]) -> bool:
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)
