"""Two-stage Skill Loading: catalog in system prompt, full skill in tool result."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillSummary:
    name: str
    description: str


@dataclass(frozen=True)
class LegalSkill:
    name: str
    description: str
    display_name: str
    collection_names: tuple[str, ...]
    use_kg: bool
    context_priority: dict[str, int]
    content: str

    @property
    def instructions(self) -> str:
        """Backward-compatible alias for context budget calculation."""
        return self.content


class SkillLoader:
    """Scan frontmatter first; load complete SKILL.md only on tool request."""

    def __init__(self, skill_root: Path | None = None):
        self.skill_root = skill_root or Path(__file__).resolve().parents[2] / "skills"
        self._catalog: dict[str, SkillSummary] | None = None
        self._cache: dict[str, LegalSkill] = {}

    @staticmethod
    def _read_frontmatter(path: Path) -> dict[str, str]:
        """Read YAML scalar frontmatter without reading the skill body."""
        metadata: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as stream:
            if stream.readline().strip() != "---":
                return metadata
            for line in stream:
                stripped = line.strip()
                if stripped == "---":
                    break
                if ":" not in stripped or stripped.startswith("#"):
                    continue
                key, value = stripped.split(":", 1)
                metadata[key.strip()] = value.strip().strip('"\'')
        return metadata

    def scan(self, refresh: bool = False) -> list[SkillSummary]:
        """Scan */SKILL.md and retain only name + description in the catalog."""
        if self._catalog is not None and not refresh:
            return list(self._catalog.values())
        catalog: dict[str, SkillSummary] = {}
        for path in sorted(self.skill_root.glob("*/SKILL.md")):
            metadata = self._read_frontmatter(path)
            name = metadata.get("name")
            description = metadata.get("description")
            if name and description:
                catalog[name] = SkillSummary(name=name, description=description)
        self._catalog = catalog
        return list(catalog.values())

    def available(self) -> list[str]:
        return [item.name for item in self.scan()]

    def catalog_prompt(self) -> str:
        lines = ["可用法律技能目录（需要完整说明时必须调用 load_skill）："]
        lines.extend(f"- {item.name}: {item.description}" for item in self.scan())
        return "\n".join(lines)

    def load(self, name: str) -> LegalSkill:
        """Load one complete skill after load_skill(name) is requested."""
        if name in self._cache:
            return self._cache[name]
        if name not in self.available():
            name = "general_legal"
        directory = self.skill_root / name
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        content = (directory / "SKILL.md").read_text(encoding="utf-8").strip()
        summary = next(item for item in self.scan() if item.name == name)
        skill = LegalSkill(
            name=name,
            description=summary.description,
            display_name=config["display_name"],
            collection_names=tuple(config.get("collection_names", ["laws", "cases"])),
            use_kg=bool(config.get("use_kg", False)),
            context_priority=dict(config.get("context_priority", {})),
            content=content,
        )
        self._cache[name] = skill
        return skill

    def load_content(self, name: str) -> str:
        return self.load(name).content


skill_loader = SkillLoader()
