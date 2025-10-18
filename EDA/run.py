from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml

from . import stages


app = typer.Typer(help="Управление пайплайном EDA для аудиоданных.")

StageName = str

STAGES: dict[StageName, str] = {
    "prepare": "Подготовка окружения и каталогов",
    "download": "Скачивание и распаковка данных",
    "inventory": "Инвентаризация аудио",
    "labels": "Валидация разметки word_bounds",
    "convert": "Конвертация в WAV 16кГц моно",
    "vad": "VAD-анализ конвертированных WAV",
    "negatives": "Отбор позитивных и негативных окон",
    "duplicates": "Поиск дубликатов",
    "cv": "Построение кросс-валидации",
    "windowing": "Расчёт seg_size и stride",
    "slices": "Определение срезов качества",
    "asr": "Мини-ASR зонд",
    "report": "Формирование итогового отчёта",
}


def load_config(path: Path) -> dict:
    if not path.exists():
        raise typer.BadParameter(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise typer.BadParameter("Config must contain a YAML mapping.")
    return data


class StageRunner:
    """Хранит конфиг и общие параметры запуска."""

    def __init__(self, config: dict, force: bool, limit_files: Optional[int]):
        self.config = stages.normalize_paths(config)
        self.force = force
        self.limit_files = limit_files

    def run_stage(self, stage: StageName) -> None:
        stage_title = STAGES.get(stage, stage)
        typer.echo(f"[{stage}] {stage_title}")
        fn = stages.STAGE_MAPPING.get(stage)
        if not fn:
            typer.echo("  Неизвестный этап, пропускаем.")
            return
        result = fn(self.config, force=self.force, limit_files=self.limit_files)
        if result.message:
            typer.echo(f"  {result.message}")
        for key, value in result.outputs.items():
            typer.echo(f"  - {key}: {value}")


def make_runner(config_path: Path, force: bool, limit_files: Optional[int]) -> StageRunner:
    cfg = load_config(config_path)
    return StageRunner(config=cfg, force=force, limit_files=limit_files)


@app.command()
def prepare(
    config: Path = typer.Option(
        Path("EDA/configs/default.yaml"),
        "--config",
        "-c",
        help="Путь к YAML конфигурации.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Пересоздавать артефакты, даже если они существуют."
    ),
    limit_files: Optional[int] = typer.Option(
        None, "--limit-files", help="Ограничить число файлов для отладки."
    ),
) -> None:
    """Подготовка окружения и каталогов."""
    runner = make_runner(config, force, limit_files)
    runner.run_stage("prepare")


@app.command()
def download(
    config: Path = typer.Option(
        Path("EDA/configs/default.yaml"),
        "--config",
        "-c",
        help="Путь к YAML конфигурации.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Пересоздавать артефакты, даже если они существуют."
    ),
    limit_files: Optional[int] = typer.Option(
        None, "--limit-files", help="Ограничить число файлов для отладки."
    ),
) -> None:
    """Скачивание и распаковка данных."""
    runner = make_runner(config, force, limit_files)
    runner.run_stage("download")


@app.command()
def inventory(
    config: Path = typer.Option(
        Path("EDA/configs/default.yaml"),
        "--config",
        "-c",
        help="Путь к YAML конфигурации.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Пересоздавать артефакты, даже если они существуют."
    ),
    limit_files: Optional[int] = typer.Option(
        None, "--limit-files", help="Ограничить число файлов для отладки."
    ),
) -> None:
    """Инвентаризация аудио."""
    runner = make_runner(config, force, limit_files)
    runner.run_stage("inventory")


@app.command()
def labels(
    config: Path = typer.Option(
        Path("EDA/configs/default.yaml"),
        "--config",
        "-c",
        help="Путь к YAML конфигурации.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Пересоздавать артефакты, даже если они существуют."
    ),
    limit_files: Optional[int] = typer.Option(
        None, "--limit-files", help="Ограничить число файлов для отладки."
    ),
) -> None:
    """Валидация word_bounds.json."""
    runner = make_runner(config, force, limit_files)
    runner.run_stage("labels")


@app.command()
def convert(
    config: Path = typer.Option(
        Path("EDA/configs/default.yaml"),
        "--config",
        "-c",
        help="Путь к YAML конфигурации.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Пересоздавать артефакты, даже если они существуют."
    ),
    limit_files: Optional[int] = typer.Option(
        None, "--limit-files", help="Ограничить число файлов для отладки."
    ),
) -> None:
    """Конвертация в WAV 16кГц моно."""
    runner = make_runner(config, force, limit_files)
    runner.run_stage("convert")


@app.command()
def vad(
    config: Path = typer.Option(
        Path("EDA/configs/default.yaml"),
        "--config",
        "-c",
        help="Путь к YAML конфигурации.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Пересоздавать артефакты, даже если они существуют."
    ),
    limit_files: Optional[int] = typer.Option(
        None, "--limit-files", help="Ограничить число файлов для отладки."
    ),
) -> None:
    """VAD-анализ."""
    runner = make_runner(config, force, limit_files)
    runner.run_stage("vad")


@app.command()
def negatives(
    config: Path = typer.Option(
        Path("EDA/configs/default.yaml"),
        "--config",
        "-c",
        help="Путь к YAML конфигурации.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Пересоздавать артефакты, даже если они существуют."
    ),
    limit_files: Optional[int] = typer.Option(
        None, "--limit-files", help="Ограничить число файлов для отладки."
    ),
) -> None:
    """Сбор позитивных и негативных окон."""
    runner = make_runner(config, force, limit_files)
    runner.run_stage("negatives")


@app.command()
def duplicates(
    config: Path = typer.Option(
        Path("EDA/configs/default.yaml"),
        "--config",
        "-c",
        help="Путь к YAML конфигурации.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Пересоздавать артефакты, даже если они существуют."
    ),
    limit_files: Optional[int] = typer.Option(
        None, "--limit-files", help="Ограничить число файлов для отладки."
    ),
) -> None:
    """Поиск дубликатов."""
    runner = make_runner(config, force, limit_files)
    runner.run_stage("duplicates")


@app.command()
def cv(
    config: Path = typer.Option(
        Path("EDA/configs/default.yaml"),
        "--config",
        "-c",
        help="Путь к YAML конфигурации.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Пересоздавать артефакты, даже если они существуют."
    ),
    limit_files: Optional[int] = typer.Option(
        None, "--limit-files", help="Ограничить число файлов для отладки."
    ),
) -> None:
    """Кросс-валидационные сплиты."""
    runner = make_runner(config, force, limit_files)
    runner.run_stage("cv")


@app.command()
def windowing(
    config: Path = typer.Option(
        Path("EDA/configs/default.yaml"),
        "--config",
        "-c",
        help="Путь к YAML конфигурации.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Пересоздавать артефакты, даже если они существуют."
    ),
    limit_files: Optional[int] = typer.Option(
        None, "--limit-files", help="Ограничить число файлов для отладки."
    ),
) -> None:
    """Расчёт параметров окна."""
    runner = make_runner(config, force, limit_files)
    runner.run_stage("windowing")


@app.command()
def slices(
    config: Path = typer.Option(
        Path("EDA/configs/default.yaml"),
        "--config",
        "-c",
        help="Путь к YAML конфигурации.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Пересоздавать артефакты, даже если они существуют."
    ),
    limit_files: Optional[int] = typer.Option(
        None, "--limit-files", help="Ограничить число файлов для отладки."
    ),
) -> None:
    """Определение срезов качества."""
    runner = make_runner(config, force, limit_files)
    runner.run_stage("slices")


@app.command()
def asr(
    config: Path = typer.Option(
        Path("EDA/configs/default.yaml"),
        "--config",
        "-c",
        help="Путь к YAML конфигурации.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Пересоздавать артефакты, даже если они существуют."
    ),
    limit_files: Optional[int] = typer.Option(
        None, "--limit-files", help="Ограничить число файлов для отладки."
    ),
) -> None:
    """Мини-ASR зонд."""
    runner = make_runner(config, force, limit_files)
    runner.run_stage("asr")


@app.command()
def report(
    config: Path = typer.Option(
        Path("EDA/configs/default.yaml"),
        "--config",
        "-c",
        help="Путь к YAML конфигурации.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Пересоздавать артефакты, даже если они существуют."
    ),
    limit_files: Optional[int] = typer.Option(
        None, "--limit-files", help="Ограничить число файлов для отладки."
    ),
) -> None:
    """Финальный отчёт."""
    runner = make_runner(config, force, limit_files)
    runner.run_stage("report")


@app.command()
def all(
    config: Path = typer.Option(
        Path("EDA/configs/default.yaml"),
        "--config",
        "-c",
        help="Путь к YAML конфигурации.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Пересоздавать артефакты, даже если они существуют."
    ),
    limit_files: Optional[int] = typer.Option(
        None, "--limit-files", help="Ограничить число файлов для отладки."
    ),
) -> None:
    """Запуск всех этапов по порядку."""
    runner = make_runner(config, force, limit_files)
    for stage in STAGES:
        runner.run_stage(stage)


if __name__ == "__main__":
    app()
