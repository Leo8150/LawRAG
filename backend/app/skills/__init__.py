"""On-demand legal skill routing and loading."""

from app.skills.loader import LegalSkill, SkillSummary, skill_loader
from app.skills.tool import load_skill

__all__ = ["LegalSkill", "SkillSummary", "skill_loader", "load_skill"]
