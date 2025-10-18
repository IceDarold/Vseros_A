from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


def ensure_dirs(paths: Iterable[Path]) -> list[Path]:
    """Создаёт отсутствующие директории и возвращает их список."""
    created = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
    return created


def ensure_parent(path: Path) -> Path:
    """Гарантирует существование родительской директории файла."""
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    Path(path).write_text(content, encoding="utf-8")


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_json(path: Path, data: Mapping) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_yaml(path: Path, data: Mapping) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def read_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_bytes(num: int, precision: int = 2) -> str:
    """Возвращает человекочитаемый размер."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.{precision}f} {unit}"
        value /= 1024.0
    return f"{value:.{precision}f} PB"


def dir_size(path: Path) -> int:
    total = 0
    for file in Path(path).rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def list_files(path: Path, suffixes: Sequence[str] | None = None) -> list[Path]:
    base = Path(path)
    if not base.exists():
        return []
    result = [p for p in base.rglob("*") if p.is_file()]
    if suffixes:
        suffixes_lower = {s.lower() for s in suffixes}
        result = [p for p in result if p.suffix.lower() in suffixes_lower]
    return sorted(result)
