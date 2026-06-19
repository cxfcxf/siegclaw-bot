"""Claude-style skills: SKILL.md folders with progressive disclosure.

Each skill lives at SKILLS_DIR/<name>/SKILL.md with YAML frontmatter providing
`name` and `description`. The frontmatter is surfaced to the model as an index in
the system prompt; the full body is loaded on demand via the `load_skill` tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import SKILLS_DIR
from .tools.registry import Tool


@dataclass
class Skill:
    name: str
    description: str
    body: str
    dir: Path


def _parse_skill_md(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from the markdown body."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                meta = {}
            return meta, parts[2].strip()
    return {}, text.strip()


def discover_skills() -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    if not SKILLS_DIR.exists():
        return skills
    for child in sorted(SKILLS_DIR.iterdir()):
        skill_md = child / "SKILL.md"
        if not (child.is_dir() and skill_md.exists()):
            continue
        meta, body = _parse_skill_md(skill_md.read_text(errors="replace"))
        name = meta.get("name") or child.name
        description = meta.get("description") or "(no description)"
        skills[name] = Skill(name=name, description=description, body=body, dir=child)
    return skills


def skills_index(skills: dict[str, Skill]) -> str:
    """A compact list injected into the system prompt."""
    if not skills:
        return ""
    lines = ["# Available skills", "Load one with the `load_skill` tool when its description fits the task.\n"]
    for s in skills.values():
        lines.append(f"- **{s.name}**: {s.description}")
    return "\n".join(lines)


def load_skill_tool(skills: dict[str, Skill]) -> Tool:
    def load_skill(name: str) -> str:
        skill = skills.get(name)
        if skill is None:
            available = ", ".join(skills) or "(none)"
            return f"Error: no skill named '{name}'. Available: {available}"
        files = [p.name for p in skill.dir.iterdir() if p.name != "SKILL.md"]
        extra = f"\n\nBundled files in this skill's directory ({skill.dir}): {', '.join(files)}" if files else ""
        return f"# Skill: {skill.name}\n\n{skill.body}{extra}"

    return Tool(
        "load_skill",
        "Load the full instructions for a named skill listed in the system prompt.",
        {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "The skill name."}},
            "required": ["name"],
        },
        load_skill,
    )
