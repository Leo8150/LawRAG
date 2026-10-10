"""FastAPI 应用入口"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.skills import skill_loader
from app.api import chat, knowledge, performance, sources

app = FastAPI(
    title=f"LawRAG — {settings.APP_NAME}",
    description="基于 LangGraph、LangChain、MySQL 与 ChromaDB 的法律 Agentic RAG 系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 启动时只扫描 SKILL.md frontmatter，将 name + description 缓存为技能目录。
# 完整正文仅在运行期调用 load_skill(name) 时读取。
skill_loader.scan()

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix=settings.API_PREFIX)
app.include_router(knowledge.router, prefix=settings.API_PREFIX)
app.include_router(performance.router, prefix=settings.API_PREFIX)
app.include_router(sources.router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "name_en": "LawRAG",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """全局健康检查 — 包含云端模型配置状态（不发起计费请求）。"""
    from app.services.kb_service import get_kb_stats
    from app.services.perf_service import get_model_status
    stats = get_kb_stats()
    model_service = await get_model_status()
    return {
        "status": "healthy",
        "models": {
            "llm": settings.LLM_MODEL,
            "embedding": settings.EMBEDDING_MODEL,
        },
        "model_service": model_service,
        "knowledge_base": stats,
    }
