"""Tool exposed to the model for loading one complete legal skill."""

from langchain_core.tools import tool

from app.skills.loader import skill_loader


@tool
def load_skill(name: str) -> str:
    """按名称加载一个法律领域 Skill 的完整 SKILL.md 内容。"""
    return skill_loader.load_content(name)
