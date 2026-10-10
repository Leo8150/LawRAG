"""High-level tools exposed to the Agentic RAG coordinator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool


@dataclass
class ToolExecution:
    content: str
    documents: list[Document] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


RetrieveCallback = Callable[[str, str, int], Awaitable[ToolExecution]]
CrimeCallback = Callable[[str], Awaitable[ToolExecution]]
SourceCallback = Callable[[str], Awaitable[ToolExecution]]
ExternalCallback = Callable[[str, dict], Awaitable[ToolExecution]]


class AgentToolRegistry:
    """Tool schemas for the LLM plus validated execution callbacks."""

    def __init__(
        self,
        retrieve: RetrieveCallback,
        crime: CrimeCallback,
        source: SourceCallback,
        external: ExternalCallback | None = None,
        external_tool_names: set[str] | None = None,
    ):
        self._retrieve = retrieve
        self._crime = crime
        self._source = source
        self._external = external
        self._external_tool_names = external_tool_names or set()

        async def retrieve_legal_evidence(
            query: str,
            collection: Literal["all", "laws", "cases"] = "all",
            top_k: int = 5,
        ) -> str:
            """检索法律法规和裁判案例，返回经过 Child 精排及 Parent 回查的证据。"""
            return (await self._retrieve(query, collection, top_k)).content

        async def lookup_crime_knowledge(crime_name: str) -> str:
            """查询刑事罪名的概念、构成要件、认定规则、处罚和关联法条。"""
            return (await self._crime(crime_name)).content

        async def get_original_source(chunk_id: str) -> str:
            """根据 parent_chunk_id 或 child_chunk_id 回查 MySQL 中的完整来源。"""
            return (await self._source(chunk_id)).content

        async def search_statutes(
            query: str,
            title_only: bool = False,
            effective_only: bool = True,
            max_results: int = 5,
        ) -> str:
            """在线检索国家法律法规数据库；用于核验现行法、法规名称、效力状态与法规 ID。"""
            return (await self._call_external("search_statutes", locals())).content

        async def get_statute_articles(regulation_id: str, keyword: str) -> str:
            """根据法规搜索结果中的 regulation_id 获取与关键词直接命中的具体法条。"""
            return (await self._call_external("get_statute_articles", locals())).content

        async def search_court_cases(query: str, max_results: int = 5) -> str:
            """在线检索人民法院案例库中的指导性案例和参考案例。"""
            return (await self._call_external("search_court_cases", locals())).content

        async def get_court_case(case_id: str, sections: list[str] | None = None) -> str:
            """根据案例搜索结果中的 case_id 获取裁判要点、案情、理由及关联法条。"""
            return (await self._call_external("get_court_case", locals())).content

        async def search_official_legal_web(query: str, max_results: int = 5) -> str:
            """仅在官方法律域名中搜索最新司法解释、政策、公告或本地库缺失的信息。"""
            return (await self._call_external("search_official_legal_web", locals())).content

        async def fetch_official_legal_page(url: str) -> str:
            """提取官方白名单 URL 正文；应在网页搜索后用于取得可引用的完整证据。"""
            return (await self._call_external("fetch_official_legal_page", locals())).content

        self.tools = [
            StructuredTool.from_function(
                coroutine=retrieve_legal_evidence,
                name="retrieve_legal_evidence",
                description=retrieve_legal_evidence.__doc__,
            ),
            StructuredTool.from_function(
                coroutine=lookup_crime_knowledge,
                name="lookup_crime_knowledge",
                description=lookup_crime_knowledge.__doc__,
            ),
            StructuredTool.from_function(
                coroutine=get_original_source,
                name="get_original_source",
                description=get_original_source.__doc__,
            ),
        ]
        external_functions = {
            "search_statutes": search_statutes,
            "get_statute_articles": get_statute_articles,
            "search_court_cases": search_court_cases,
            "get_court_case": get_court_case,
            "search_official_legal_web": search_official_legal_web,
            "fetch_official_legal_page": fetch_official_legal_page,
        }
        for name, function in external_functions.items():
            if name in self._external_tool_names:
                self.tools.append(StructuredTool.from_function(
                    coroutine=function,
                    name=name,
                    description=function.__doc__,
                ))
        self._by_name = {tool.name: tool for tool in self.tools}

    async def _call_external(self, name: str, arguments: dict) -> ToolExecution:
        if self._external is None or name not in self._external_tool_names:
            raise ValueError(f"外部 MCP 工具未启用: {name}")
        return await self._external(name, arguments)

    async def execute(self, name: str, arguments: dict) -> ToolExecution:
        if name == "retrieve_legal_evidence":
            collection = str(arguments.get("collection", "all"))
            if collection not in {"all", "laws", "cases"}:
                collection = "all"
            top_k = min(20, max(1, int(arguments.get("top_k", 5))))
            return await self._retrieve(str(arguments.get("query", "")), collection, top_k)
        if name == "lookup_crime_knowledge":
            return await self._crime(str(arguments.get("crime_name", "")))
        if name == "get_original_source":
            return await self._source(str(arguments.get("chunk_id", "")))
        if name in self._external_tool_names:
            return await self._call_external(name, arguments)
        raise ValueError(f"未注册的 Agent 工具: {name}")

    def available(self, name: str) -> bool:
        return name in self._by_name
