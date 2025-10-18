from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import librosa
import librosa.display as librosa_display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

from .utils import audio, fs, io, report


plt.switch_backend("Agg")
sns.set_theme(style="whitegrid")

RNG = random.Random(42)


@dataclass
class StageResult:
    name: str
    outputs: Dict[str, Any]
    message: str = ""


def normalize_paths(cfg: dict) -> dict:
    """Преобразует строковые пути в Path там, где это уместно."""
    normalized: dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            normalized[key] = normalize_paths(value)
        elif isinstance(value, str) and value.startswith("/"):
            normalized[key] = Path(value)
        else:
            normalized[key] = value
    return normalized


# ---------- Вспомогательные функции ----------

def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_histogram(series: pd.Series, path: Path, title: str, xlabel: str, bins: int = 40) -> Optional[Path]:
    if series.dropna().empty:
        return None
    _ensure_parent(path)
    plt.figure(figsize=(7, 4))
    sns.histplot(series.dropna(), bins=bins, kde=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _save_boxplot(series: pd.Series, path: Path, title: str, ylabel: str) -> Optional[Path]:
    if series.dropna().empty:
        return None
    _ensure_parent(path)
    plt.figure(figsize=(5, 4))
    sns.boxplot(y=series.dropna())
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _save_scatter(df: pd.DataFrame, x: str, y: str, path: Path, title: str, xlabel: str, ylabel: str) -> Optional[Path]:
    if df.empty:
        return None
    _ensure_parent(path)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _save_mel_grid(samples: list[tuple[str, np.ndarray, int]], path: Path, title: str, cols: int = 3) -> Optional[Path]:
    if not samples:
        return None
    _ensure_parent(path)
    rows = (len(samples) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for idx, (label, waveform, sr) in enumerate(samples, start=1):
        plt.subplot(rows, cols, idx)
        mel = audio.mel_spectrogram(waveform, sr)
        librosa_display.specshow(mel, sr=sr, x_axis="time", y_axis="mel")
        plt.title(label)
        plt.colorbar(format="%+2.0f dB")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _load_inventory(config: dict) -> pd.DataFrame:
    table = Path(config["paths"]["tables"]) / "raw_inventory.parquet"
    if not table.exists():
        raise FileNotFoundError("Не найден raw_inventory.parquet. Запустите stage 'inventory'.")
    return pd.read_parquet(table)


def _load_labels(config: dict) -> pd.DataFrame:
    table = Path(config["paths"]["tables"]) / "labels_validity.parquet"
    if not table.exists():
        raise FileNotFoundError("Не найден labels_validity.parquet. Запустите stage 'labels'.")
    return pd.read_parquet(table)


def _load_word_bounds(config: dict) -> dict:
    wb_path = Path(config["dataset"]["word_bounds_path"])
    if not wb_path.exists():
        raise FileNotFoundError(f"Не найден word_bounds.json: {wb_path}")
    with wb_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def _bounds_to_intervals(bounds: Any) -> list[tuple[float, float]]:
    if isinstance(bounds, dict):
        return [(float(v[0]), float(v[1])) for v in bounds.values()]
    if isinstance(bounds, (list, tuple)) and bounds and isinstance(bounds[0], (list, tuple)):
        return [(float(x[0]), float(x[1])) for x in bounds]
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        return [(float(bounds[0]), float(bounds[1]))]
    return []


def _intervals_overlap(intervals: Iterable[tuple[float, float]], start: float, end: float) -> bool:
    for left, right in intervals:
        if right > start and left < end:
            return True
    return False


def _extract_positive_segment(
    waveform: np.ndarray,
    sr: int,
    bounds: tuple[float, float],
    seg_samples: int,
    margin_before: float,
    margin_after: float,
) -> np.ndarray:
    n_samples = waveform.size
    start_sec, end_sec = bounds
    left = max(0, int(round(start_sec * sr)) - int(round(margin_before * sr)))
    right = min(n_samples, int(round(end_sec * sr)) + int(round(margin_after * sr)))
    if right <= left:
        left = max(0, int(round(start_sec * sr)))
        right = min(n_samples, left + seg_samples)
    if right - left >= seg_samples:
        center = (left + right) // 2
        left = max(0, center - seg_samples // 2)
        right = min(n_samples, left + seg_samples)
    else:
        extra = seg_samples - (right - left)
        left = max(0, left - extra // 2)
        right = min(n_samples, left + seg_samples)
    if right - left < seg_samples:
        pad = seg_samples - (right - left)
        segment = waveform[left:right]
        return np.pad(segment, (0, pad))
    return waveform[left:right]


def _extract_negative_segment(
    waveform: np.ndarray,
    sr: int,
    seg_samples: int,
) -> tuple[np.ndarray, float]:
    if waveform.size <= seg_samples:
        segment = np.pad(waveform, (0, max(0, seg_samples - waveform.size)))
        return segment[:seg_samples], 0.0
    start = RNG.randint(0, waveform.size - seg_samples)
    segment = waveform[start : start + seg_samples]
    return segment, start / sr


# ---------- Этапы ----------

def stage_prepare(config: dict, force: bool = False, **_) -> StageResult:
    paths = config["paths"]
    directories = [
        paths["data_root"],
        paths["raw_train"].parent if isinstance(paths["raw_train"], Path) else Path(paths["raw_train"]).parent,
        paths["raw_test"].parent if isinstance(paths["raw_test"], Path) else Path(paths["raw_test"]).parent,
        paths["meta"],
        paths["interim_wav"],
        paths["tables"],
        paths["samples"],
        paths["reports"],
        paths["figs"],
    ]
    created = fs.ensure_dirs(directories)
    env_log = Path(paths["env_log"])
    if force or not env_log.exists():
        content = [
            "Заполнить при запуске в Colab:",
            "!ffmpeg -version",
            "!python --version",
            "import torchaudio, librosa",
            "print(torchaudio.__version__, librosa.__version__)",
        ]
        fs.write_text(env_log, "\n".join(content))
    return StageResult(
        name="prepare",
        outputs={
            "created_dirs": len(created),
            "env_log": str(env_log),
        },
        message="Каталоги проверены, env_versions.txt готов.",
    )


def stage_download(config: dict, force: bool = False, **_) -> StageResult:
    outputs: dict[str, Any] = {}
    for name, item in config["downloads"].items():
        url = item["url"]
        output = Path(item["output"])
        archive = io.download_from_mailru(url, output, force=force)
        io.extract_archive(archive, Path(item["extract_to"]))
        outputs[f"{name}_archive"] = str(archive)
        outputs[f"{name}_dir"] = str(item["extract_to"])
    return StageResult(
        name="download",
        outputs=outputs,
        message="Архивы скачаны и распакованы.",
    )


def stage_inventory(
    config: dict,
    force: bool = False,
    limit_files: Optional[int] = None,
    **_,
) -> StageResult:
    paths = config["paths"]
    figs_dir = Path(paths["figs"]) / "inventory"
    table_path = Path(paths["tables"]) / "raw_inventory.parquet"
    if table_path.exists() and not force:
        df = pd.read_parquet(table_path)
        return StageResult(
            name="inventory",
            outputs={
                "inventory_table": str(table_path),
                "rows": len(df),
            },
            message="Инвентаризация ранее выполнена.",
        )

    audio_cfg = config["audio"]
    records: list[dict[str, Any]] = []
    for split, directory in (("train", paths["raw_train"]), ("test", paths["raw_test"])):
        base_dir = Path(directory)
        files = sorted(base_dir.rglob("*.opus"))
        if limit_files:
            files = files[:limit_files]
        for path in files:
            try:
                info = audio.ffprobe_info(path)
            except audio.FFmpegError:
                info = {}
            duration_meta = float(info.get("format", {}).get("duration", 0.0) or 0.0)
            stream = info.get("streams", [{}])[0] if info.get("streams") else {}
            sr_meta = int(stream.get("sample_rate") or 0)
            channels_meta = int(stream.get("channels") or 0)

            try:
                waveform, sr = audio.load_audio(path, sample_rate=audio_cfg["target_sample_rate"], mono=True)
            except Exception:
                waveform, sr = np.array([], dtype=np.float32), audio_cfg["target_sample_rate"]
            actual_duration = waveform.size / sr if sr else duration_meta
            rms = audio.rms_energy(waveform)
            peak = audio.peak_amplitude(waveform)
            silence = audio.silence_ratio(waveform)
            if waveform.size:
                spec_centroid = float(librosa.feature.spectral_centroid(y=waveform, sr=sr).mean())
                spec_rolloff = float(librosa.feature.spectral_rolloff(y=waveform, sr=sr).mean())
            else:
                spec_centroid = 0.0
                spec_rolloff = 0.0
            records.append(
                {
                    "split": split,
                    "path": str(path),
                    "file_id": path.stem,
                    "duration_sec": actual_duration if actual_duration else duration_meta,
                    "duration_ffprobe": duration_meta,
                    "sample_rate": sr if waveform.size else sr_meta,
                    "channels": 1 if waveform.size else channels_meta,
                    "size_bytes": path.stat().st_size,
                    "rms": rms,
                    "peak": peak,
                    "silence_ratio": silence,
                    "spec_centroid": spec_centroid,
                    "spec_rolloff": spec_rolloff,
                }
            )
    df = pd.DataFrame(records)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(table_path)

    figs = []
    duration_hist = _save_histogram(
        df["duration_sec"],
        figs_dir / "duration_hist.png",
        "Распределение длительностей (sec)",
        "duration, sec",
    )
    if duration_hist:
        figs.append(str(duration_hist))
    rms_hist = _save_histogram(df["rms"], figs_dir / "rms_hist.png", "Распределение RMS", "rms")
    if rms_hist:
        figs.append(str(rms_hist))
    silence_hist = _save_histogram(
        df["silence_ratio"], figs_dir / "silence_ratio_hist.png", "Доля тишины", "ratio"
    )
    if silence_hist:
        figs.append(str(silence_hist))
    sr_box = _save_boxplot(df["sample_rate"], figs_dir / "sample_rate_box.png", "Sample rate", "Hz")
    if sr_box:
        figs.append(str(sr_box))

    return StageResult(
        name="inventory",
        outputs={
            "inventory_table": str(table_path),
            "rows": len(df),
            "figures": figs,
        },
        message="Сформирована таблица raw_inventory и базовые гистограммы.",
    )


def stage_labels(config: dict, force: bool = False, **_) -> StageResult:
    paths = config["paths"]
    figs_dir = Path(paths["figs"]) / "labels"
    table_path = Path(paths["tables"]) / "labels_validity.parquet"
    inventory = _load_inventory(config)
    inventory_map = inventory.set_index("file_id")["duration_sec"].to_dict()

    if table_path.exists() and not force:
        df = pd.read_parquet(table_path)
        return StageResult(
            name="labels",
            outputs={
                "labels_table": str(table_path),
                "rows": len(df),
            },
            message="Результаты валидации меток уже существуют.",
        )

    word_bounds = _load_word_bounds(config)
    records = []
    invalid = 0
    for file_id, bounds in word_bounds.items():
        for start, end in _bounds_to_intervals(bounds):
            clip_duration = inventory_map.get(file_id)
            duration = max(0.0, end - start)
            within_audio = clip_duration is None or end <= clip_duration + 1e-3
            valid = start >= 0.0 and end >= 0.0 and duration > 0.0 and within_audio
            if not valid:
                invalid += 1
            position = (start / clip_duration) if clip_duration and clip_duration > 0 else np.nan
            records.append(
                {
                    "file_id": file_id,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "audio_duration": clip_duration,
                    "position": position,
                    "valid": valid,
                }
            )
    df = pd.DataFrame(records)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(table_path)

    figs = []
    duration_hist = _save_histogram(
        df["duration"],
        figs_dir / "label_duration_hist.png",
        "Длительности меток",
        "seconds",
    )
    if duration_hist:
        figs.append(str(duration_hist))
    position_hist = _save_histogram(
        df["position"].dropna(),
        figs_dir / "label_position_hist.png",
        "Позиция метки в файле",
        "start / duration",
    )
    if position_hist:
        figs.append(str(position_hist))

    return StageResult(
        name="labels",
        outputs={
            "labels_table": str(table_path),
            "total_labels": len(df),
            "invalid": invalid,
            "figures": figs,
        },
        message="Проверены границы меток, построены распределения.",
    )


def stage_convert(
    config: dict,
    force: bool = False,
    limit_files: Optional[int] = None,
    **_,
) -> StageResult:
    audio_cfg = config["audio"]
    paths = config["paths"]
    output_dir = Path(paths["interim_wav"])
    output_dir.mkdir(parents=True, exist_ok=True)

    conversions: list[tuple[Path, Path]] = []
    for split, raw_dir in (("train", paths["raw_train"]), ("test", paths["raw_test"])):
        base = Path(raw_dir)
        files = sorted(base.rglob("*.opus"))
        if limit_files:
            files = files[:limit_files]
        for src in files:
            rel = src.relative_to(base)
            dst = output_dir / split / rel.with_suffix(".wav")
            conversions.append((src, dst))

    converted = audio.batch_convert(
        conversions,
        sample_rate=audio_cfg["target_sample_rate"],
        channels=audio_cfg["target_channels"],
        force=force,
    )
    return StageResult(
        name="convert",
        outputs={
            "converted_files": len(converted),
            "output_dir": str(output_dir),
        },
        message="Файлы преобразованы в WAV 16кГц моно.",
    )


def stage_vad(
    config: dict,
    force: bool = False,
    limit_files: Optional[int] = None,
    **_,
) -> StageResult:
    paths = config["paths"]
    figs_dir = Path(paths["figs"]) / "vad"
    table_path = Path(paths["tables"]) / "vad_stats.parquet"
    if table_path.exists() and not force:
        df = pd.read_parquet(table_path)
        return StageResult(
            name="vad",
            outputs={
                "vad_table": str(table_path),
                "rows": len(df),
            },
            message="VAD-таблица уже существует.",
        )

    audio_cfg = config["audio"]
    analysis_cfg = config["analysis"]
    inventory = _load_inventory(config)
    label_counts = (
        _load_labels(config).groupby("file_id").size().to_dict()
        if (Path(paths["tables"]) / "labels_validity.parquet").exists()
        else {}
    )
    word_bounds = _load_word_bounds(config)

    wav_root = Path(paths["interim_wav"])
    wav_files = sorted(wav_root.rglob("*.wav"))
    if limit_files:
        wav_files = wav_files[:limit_files]
    elif analysis_cfg.get("vad_probe_limit"):
        wav_files = wav_files[: analysis_cfg["vad_probe_limit"]]

    entries = []
    for wav_path in wav_files:
        waveform, sr = audio.load_audio(wav_path, sample_rate=audio_cfg["target_sample_rate"], mono=True)
        duration_sec = waveform.size / sr if sr else 0.0
        intervals = librosa.effects.split(waveform, top_db=analysis_cfg.get("vad_top_db", 40))
        speech_samples = sum((end - start) for start, end in intervals)
        speech_ratio = (speech_samples / waveform.size) if waveform.size else 0.0
        speech_segments = (np.array(intervals) / sr).tolist() if intervals.size else []
        file_id = wav_path.stem
        label_list = _bounds_to_intervals(word_bounds.get(file_id, []))
        label_over_silence = False
        if label_list:
            label_over_silence = not any(
                _intervals_overlap(speech_segments, start, end) for start, end in label_list
            )
        entries.append(
            {
                "path": str(wav_path),
                "file_id": file_id,
                "duration_sec": duration_sec,
                "speech_ratio": speech_ratio,
                "n_segments": len(speech_segments),
                "avg_segment_sec": float(np.mean([e - s for s, e in speech_segments])) if speech_segments else 0.0,
                "median_segment_sec": float(np.median([e - s for s, e in speech_segments])) if speech_segments else 0.0,
                "label_count": label_counts.get(file_id, 0),
                "label_on_silence": bool(label_over_silence),
            }
        )
    df = pd.DataFrame(entries)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(table_path)

    figs = []
    speech_hist = _save_histogram(
        df["speech_ratio"],
        figs_dir / "speech_ratio_hist.png",
        "% речи по файлам",
        "speech ratio",
    )
    if speech_hist:
        figs.append(str(speech_hist))
    scatter = _save_scatter(
        df,
        "duration_sec",
        "label_count",
        figs_dir / "duration_vs_labels.png",
        "Связь длительности и числа меток",
        "duration, sec",
        "# labels",
    )
    if scatter:
        figs.append(str(scatter))

    return StageResult(
        name="vad",
        outputs={
            "vad_table": str(table_path),
            "rows": len(df),
            "label_on_silence_cases": int(df["label_on_silence"].sum()),
            "figures": figs,
        },
        message="Рассчитаны метрики VAD и построены графики.",
    )


def stage_negatives(
    config: dict,
    force: bool = False,
    limit_files: Optional[int] = None,
    **_,
) -> StageResult:
    paths = config["paths"]
    tables_dir = Path(paths["tables"])
    samples_dir = Path(paths["samples"])
    figs_dir = Path(paths["figs"]) / "negatives"
    table_path = tables_dir / "hard_negatives.parquet"
    if table_path.exists() and not force:
        df = pd.read_parquet(table_path)
        return StageResult(
            name="negatives",
            outputs={
                "negatives_table": str(table_path),
                "rows": len(df),
            },
            message="Hard negatives уже сформированы.",
        )

    analysis_cfg = config["analysis"]
    audio_cfg = config["audio"]
    window_cfg = config["windowing"]
    inventory = _load_inventory(config)
    word_bounds = _load_word_bounds(config)

    train_inventory = inventory[inventory["split"] == "train"]
    if limit_files:
        train_inventory = train_inventory.head(limit_files)

    pos_ids = [fid for fid in train_inventory["file_id"] if fid in word_bounds]
    neg_ids = [fid for fid in train_inventory["file_id"] if fid not in word_bounds]

    seg_size = window_cfg.get("seg_size_default", 1.5)
    seg_samples = int(seg_size * audio_cfg["target_sample_rate"])
    margin_before = window_cfg.get("margin_before", 0.25)
    margin_after = window_cfg.get("margin_after", 0.25)

    pos_target = min(analysis_cfg.get("pos_samples", 50), len(pos_ids))
    neg_target = analysis_cfg.get("neg_samples", 50)

    pos_dir = samples_dir / "pos"
    neg_dir = samples_dir / "neg"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    pos_previews: list[tuple[str, np.ndarray, int]] = []
    neg_previews: list[tuple[str, np.ndarray, int]] = []

    # Положительные сегменты
    RNG.shuffle(pos_ids)
    for idx, file_id in enumerate(pos_ids[:pos_target], start=1):
        opus_path = train_inventory.loc[train_inventory["file_id"] == file_id, "path"].iloc[0]
        opus_path = Path(opus_path)
        raw_root = Path(config["paths"]["raw_train"])
        rel = opus_path.relative_to(raw_root)
        wav_path = Path(paths["interim_wav"]) / "train" / rel.with_suffix(".wav")
        if not wav_path.exists():
            continue
        waveform, sr = audio.load_audio(wav_path, sample_rate=audio_cfg["target_sample_rate"], mono=True)
        bounds_list = _bounds_to_intervals(word_bounds[file_id])
        if not bounds_list:
            continue
        segment = _extract_positive_segment(
            waveform,
            sr,
            bounds_list[0],
            seg_samples,
            margin_before,
            margin_after,
        )
        sample_path = pos_dir / f"pos_{idx:03d}_{file_id}.wav"
        audio.save_audio(sample_path, segment, sr)
        silence_ratio = audio.silence_ratio(segment)
        meta_rows.append(
            {
                "file_id": file_id,
                "type": "positive",
                "sample_path": str(sample_path),
                "source_path": str(wav_path),
                "start_sec": bounds_list[0][0],
                "end_sec": bounds_list[0][1],
                "silence_ratio": silence_ratio,
            }
        )
        if len(pos_previews) < analysis_cfg.get("preview_count", 12):
            pos_previews.append((file_id, segment, sr))

    # Негативные сегменты
    RNG.shuffle(neg_ids)
    generated = 0
    for file_id in neg_ids:
        if generated >= neg_target:
            break
        opus_path = train_inventory.loc[train_inventory["file_id"] == file_id, "path"].iloc[0]
        opus_path = Path(opus_path)
        raw_root = Path(config["paths"]["raw_train"])
        rel = opus_path.relative_to(raw_root)
        wav_path = Path(paths["interim_wav"]) / "train" / rel.with_suffix(".wav")
        if not wav_path.exists():
            continue
        waveform, sr = audio.load_audio(wav_path, sample_rate=audio_cfg["target_sample_rate"], mono=True)
        segment, start_sec = _extract_negative_segment(waveform, sr, seg_samples)
        sample_path = neg_dir / f"neg_{generated + 1:03d}_{file_id}.wav"
        audio.save_audio(sample_path, segment, sr)
        silence_ratio = audio.silence_ratio(segment)
        category = "silence" if silence_ratio > 0.8 else "speech" if silence_ratio < 0.3 else "noisy"
        meta_rows.append(
            {
                "file_id": file_id,
                "type": f"negative_{category}",
                "sample_path": str(sample_path),
                "source_path": str(wav_path),
                "start_sec": start_sec,
                "end_sec": start_sec + seg_size,
                "silence_ratio": silence_ratio,
            }
        )
        generated += 1
        if len(neg_previews) < analysis_cfg.get("preview_count", 12):
            neg_previews.append((file_id, segment, sr))

    df = pd.DataFrame(meta_rows)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(table_path)

    figs = []
    pos_mel = _save_mel_grid(
        pos_previews,
        figs_dir / "positives_mel.png",
        "Мел-карты позитивных окон",
        cols=analysis_cfg.get("mel_preview_cols", 3),
    )
    if pos_mel:
        figs.append(str(pos_mel))
    neg_mel = _save_mel_grid(
        neg_previews,
        figs_dir / "negatives_mel.png",
        "Мел-карты негативных окон",
        cols=analysis_cfg.get("mel_preview_cols", 3),
    )
    if neg_mel:
        figs.append(str(neg_mel))

    return StageResult(
        name="negatives",
        outputs={
            "negatives_table": str(table_path),
            "rows": len(df),
            "samples_dir": str(samples_dir),
            "figures": figs,
        },
        message="Подготовлены позитивные и негативные окна, сохранены примеры.",
    )


def stage_duplicates(
    config: dict,
    force: bool = False,
    limit_files: Optional[int] = None,
    **_,
) -> StageResult:
    paths = config["paths"]
    table_path = Path(paths["tables"]) / "dups.parquet"
    figs_dir = Path(paths["figs"]) / "duplicates"
    if table_path.exists() and not force:
        df = pd.read_parquet(table_path)
        return StageResult(
            name="duplicates",
            outputs={
                "duplicates_table": str(table_path),
                "rows": len(df),
            },
            message="Информация о дубликатах уже сохранена.",
        )

    inventory = _load_inventory(config)
    if limit_files:
        inventory = inventory.head(limit_files)

    hashes = []
    for path in inventory["path"]:
        file_path = Path(path)
        hashes.append(_file_md5(file_path))
    inventory = inventory.assign(md5=hashes)

    records = []
    group_idx = 0
    exact_seen: set[str] = set()
    for md5, group in inventory.groupby("md5"):
        if len(group) <= 1:
            continue
        group_idx += 1
        for row in group.itertuples():
            records.append(
                {
                    "group_id": f"exact_{group_idx}",
                    "type": "exact",
                    "file_id": row.file_id,
                    "split": row.split,
                    "path": row.path,
                    "duration_sec": row.duration_sec,
                    "size_bytes": row.size_bytes,
                    "md5": row.md5,
                }
            )
            exact_seen.add(row.file_id)

    inventory_remaining = inventory[~inventory["file_id"].isin(exact_seen)].copy()
    inventory_remaining["duration_bucket"] = (inventory_remaining["duration_sec"] * 100).round().astype(int)
    for _, group in inventory_remaining.groupby("duration_bucket"):
        if len(group) <= 1:
            continue
        group_idx += 1
        for row in group.itertuples():
            records.append(
                {
                    "group_id": f"near_{group_idx}",
                    "type": "near",
                    "file_id": row.file_id,
                    "split": row.split,
                    "path": row.path,
                    "duration_sec": row.duration_sec,
                    "size_bytes": row.size_bytes,
                    "md5": row.md5,
                }
            )

    df = pd.DataFrame(records)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(table_path)

    fig_path = None
    if not df.empty:
        summary = df.groupby("type")["file_id"].nunique()
        fig_path = _save_histogram(summary.reindex(["exact", "near"]).fillna(0), figs_dir / "duplicates_count.png", "Типы совпадений", "count")

    return StageResult(
        name="duplicates",
        outputs={
            "duplicates_table": str(table_path),
            "groups": df["group_id"].nunique(),
            "figures": [str(fig_path)] if fig_path else [],
        },
        message="Построены группы exact и near дубликатов.",
    )


def stage_cv(
    config: dict,
    force: bool = False,
    limit_files: Optional[int] = None,
    **_,
) -> StageResult:
    paths = config["paths"]
    cv_path = Path(paths["tables"]) / "cv_splits_v1.json"
    summary_path = Path(paths["tables"]) / "cv_summary.parquet"
    if cv_path.exists() and summary_path.exists() and not force:
        return StageResult(
            name="cv",
            outputs={
                "cv_splits": str(cv_path),
                "cv_summary": str(summary_path),
            },
            message="CV сплиты уже рассчитаны.",
        )

    analysis_cfg = config["analysis"]
    inventory = _load_inventory(config)
    labels = _load_labels(config)
    train_df = inventory[inventory["split"] == "train"].copy()
    if limit_files:
        train_df = train_df.head(limit_files)
    label_counts = labels.groupby("file_id").size().to_dict()
    train_df["label"] = train_df["file_id"].map(lambda x: 1 if x in label_counts else 0)

    folds = analysis_cfg.get("cv_folds", 5)
    folds = max(2, min(folds, len(train_df)))
    unique_labels = train_df["label"].nunique()
    if unique_labels < 2:
        # Если нет двух классов, fallback к простому разбиению
        mid = len(train_df) // 5 or 1
        splits = [{"train": train_df["file_id"].tolist()[mid:], "val": train_df["file_id"].tolist()[:mid]}]
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
            train_ids = train_df.iloc[train_idx]["file_id"].tolist()
            val_ids = train_df.iloc[val_idx]["file_id"].tolist()
            splits.append({"fold": fold_idx, "train": train_ids, "val": val_ids})

    fs.write_json(cv_path, {"folds": splits})

    summary_rows = []
    for fold_entry in splits:
        val_ids = fold_entry["val"]
        val_df = train_df[train_df["file_id"].isin(val_ids)]
        summary_rows.append(
            {
                "fold": fold_entry.get("fold", 0),
                "val_files": len(val_ids),
                "val_pos": int(val_df["label"].sum()),
                "val_neg": int(len(val_df) - val_df["label"].sum()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(summary_path)

    return StageResult(
        name="cv",
        outputs={
            "cv_splits": str(cv_path),
            "cv_summary": str(summary_path),
        },
        message="Сформированы стратифицированные CV-сплиты.",
    )


def stage_windowing(config: dict, force: bool = False, **_) -> StageResult:
    paths = config["paths"]
    window_cfg_path = Path(paths["tables"]) / "windowing_config.yaml"
    if window_cfg_path.exists() and not force:
        return StageResult(
            name="windowing",
            outputs={"windowing_config": str(window_cfg_path)},
            message="Параметры окна уже сохранены.",
        )

    labels = _load_labels(config)
    valid = labels[labels["valid"]]
    if valid.empty:
        seg_size = config["windowing"].get("seg_size_default", 1.5)
    else:
        seg_size = float(valid["duration"].quantile(0.75) + 0.3)
    seg_size = round(max(seg_size, 0.5), 2)
    stride = round(max(0.1, min(seg_size / 2, 0.5)), 2)
    margin_before = config["windowing"].get("margin_before", 0.25)
    margin_after = config["windowing"].get("margin_after", 0.25)
    cfg = {
        "seg_size": seg_size,
        "stride": stride,
        "margin_before": margin_before,
        "margin_after": margin_after,
    }
    fs.write_yaml(window_cfg_path, cfg)
    return StageResult(
        name="windowing",
        outputs={"windowing_config": str(window_cfg_path), **cfg},
        message="Рассчитаны и сохранены параметры окна.",
    )


def stage_slices(config: dict, force: bool = False, **_) -> StageResult:
    paths = config["paths"]
    slices_path = Path(paths["tables"]) / "slices.yaml"
    if slices_path.exists() and not force:
        return StageResult(
            name="slices",
            outputs={"slices": str(slices_path)},
            message="slices.yaml уже существует.",
        )

    inventory = _load_inventory(config)
    vad_df = pd.read_parquet(Path(paths["tables"]) / "vad_stats.parquet") if (Path(paths["tables"]) / "vad_stats.parquet").exists() else pd.DataFrame()
    labels = _load_labels(config)

    train_inv = inventory[inventory["split"] == "train"]
    duration_quantiles = train_inv["duration_sec"].quantile([0.33, 0.66]).to_dict() if not train_inv.empty else {}
    silence_quantiles = vad_df["speech_ratio"].quantile([0.33, 0.66]).to_dict() if not vad_df.empty else {}
    label_counts = labels.groupby("file_id").size()
    label_density = []
    for _, row in train_inv.iterrows():
        count = label_counts.get(row["file_id"], 0)
        denom = row["duration_sec"] if row["duration_sec"] else np.nan
        label_density.append(count / denom if denom else np.nan)
    label_density = pd.Series(label_density).dropna()
    density_quantiles = label_density.quantile([0.33, 0.66]).to_dict() if not label_density.empty else {}

    slices_cfg = {
        "duration_sec": {
            "short": {"max": float(duration_quantiles.get(0.33, 10.0))},
            "medium": {"min": float(duration_quantiles.get(0.33, 10.0)), "max": float(duration_quantiles.get(0.66, 30.0))},
            "long": {"min": float(duration_quantiles.get(0.66, 30.0))},
        },
        "speech_ratio": {
            "low": {"max": float(silence_quantiles.get(0.33, 0.3))},
            "medium": {"min": float(silence_quantiles.get(0.33, 0.3)), "max": float(silence_quantiles.get(0.66, 0.7))},
            "high": {"min": float(silence_quantiles.get(0.66, 0.7))},
        },
        "label_density": {
            "sparse": {"max": float(density_quantiles.get(0.33, 0.5))},
            "balanced": {"min": float(density_quantiles.get(0.33, 0.5)), "max": float(density_quantiles.get(0.66, 1.0))},
            "dense": {"min": float(density_quantiles.get(0.66, 1.0))},
        },
    }
    fs.write_yaml(slices_path, slices_cfg)
    return StageResult(
        name="slices",
        outputs={"slices": str(slices_path)},
        message="Определены срезы по длительности, речи и плотности меток.",
    )


def stage_asr(config: dict, force: bool = False, **_) -> StageResult:
    paths = config["paths"]
    table_path = Path(paths["tables"]) / "asr_probe.parquet"
    if table_path.exists() and not force:
        df = pd.read_parquet(table_path)
        return StageResult(
            name="asr",
            outputs={
                "asr_table": str(table_path),
                "rows": len(df),
            },
            message="ASR-зонд уже выполнен.",
        )

    # Плейсхолдер: запуск ASR опционален, поэтому создаём пустую таблицу.
    df = pd.DataFrame(columns=["file_id", "segment_idx", "transcript", "confidence"])
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(table_path)
    return StageResult(
        name="asr",
        outputs={
            "asr_table": str(table_path),
            "rows": 0,
        },
        message="ASR-зонд не запускался (опционально).",
    )


def stage_report(config: dict, force: bool = False, **_) -> StageResult:
    summary_path = Path(config["report"]["summary_path"])
    if summary_path.exists() and not force:
        return StageResult(
            name="report",
            outputs={"summary": str(summary_path)},
            message="Отчёт уже существует.",
        )

    sections = [
        (
            "Статус",
            "Пайплайн EDA выполнен. Обновите раздел после реального запуска.",
        ),
        (
            "Артефакты",
            "\n".join(
                [
                    f"- raw_inventory: {config['paths']['tables']}/raw_inventory.parquet",
                    f"- labels_validity: {config['paths']['tables']}/labels_validity.parquet",
                    f"- vad_stats: {config['paths']['tables']}/vad_stats.parquet",
                    f"- hard_negatives: {config['paths']['tables']}/hard_negatives.parquet",
                    f"- cv_splits_v1.json: {config['paths']['tables']}/cv_splits_v1.json",
                ]
            ),
        ),
    ]
    report.write_markdown_report(summary_path, sections)
    return StageResult(
        name="report",
        outputs={"summary": str(summary_path)},
        message="Черновой отчёт EDA сформирован.",
    )


STAGE_MAPPING = {
    "prepare": stage_prepare,
    "download": stage_download,
    "inventory": stage_inventory,
    "labels": stage_labels,
    "convert": stage_convert,
    "vad": stage_vad,
    "negatives": stage_negatives,
    "duplicates": stage_duplicates,
    "cv": stage_cv,
    "windowing": stage_windowing,
    "slices": stage_slices,
    "asr": stage_asr,
    "report": stage_report,
}
