from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from .fs import ensure_parent, write_text


def format_section(title: str, lines: Iterable[str]) -> str:
    body = "\n".join(f"- {line}" for line in lines)
    return f"## {title}\n{body}\n"


def format_key_values(items: Mapping[str, str | int | float]) -> str:
    lines = [f"- **{key}**: {value}" for key, value in items.items()]
    return "\n".join(lines)


def write_markdown_report(path: Path, sections: list[tuple[str, str]]) -> None:
    ensure_parent(path)
    content = "\n".join(f"## {title}\n{body}\n" for title, body in sections)
    write_text(path, content)


def append_to_markdown(path: Path, title: str, body: str) -> None:
    ensure_parent(path)
    markdown = Path(path)
    block = f"\n## {title}\n{body}\n"
    if markdown.exists():
        existing = markdown.read_text(encoding="utf-8")
        markdown.write_text(existing + block, encoding="utf-8")
    else:
        markdown.write_text(block, encoding="utf-8")
