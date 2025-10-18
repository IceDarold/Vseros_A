from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


class FFmpegError(RuntimeError):
    """Исключение для ошибок ffmpeg."""


def ffprobe_info(path: Path) -> dict:
    """
    Возвращает метаданные через ffprobe в формате dict.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise FFmpegError("ffprobe executable not found") from exc
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.strip() or "ffprobe failed")
    return json.loads(proc.stdout or "{}")


def convert_to_wav(
    src: Path,
    dst: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    force: bool = False,
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """
    Конвертирует аудио в WAV с заданными параметрами.
    """
    src = Path(src)
    dst = Path(dst)
    if dst.exists() and not force:
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(src),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-vn",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.strip())
    return dst


def load_audio(path: Path, sample_rate: Optional[int] = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Загружает аудио через librosa.
    """
    waveform, sr = librosa.load(path, sr=sample_rate, mono=mono)
    return waveform, sr


def save_audio(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    """
    Сохраняет аудио массив в WAV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sample_rate)


def rms_energy(waveform: np.ndarray) -> float:
    """Корень среднего квадратичного значения."""
    waveform = np.asarray(waveform, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0


def peak_amplitude(waveform: np.ndarray) -> float:
    """Максимальная абсолютная амплитуда."""
    waveform = np.asarray(waveform, dtype=np.float32)
    return float(np.max(np.abs(waveform))) if waveform.size else 0.0


def silence_ratio(
    waveform: np.ndarray,
    threshold: float = 1e-3,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> float:
    """
    Оценивает долю тишины: процент фреймов, где RMS ниже порога.
    """
    if waveform.size == 0:
        return 1.0

    rms = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    ).flatten()
    if rms.size == 0:
        return 1.0
    silent = np.count_nonzero(rms < threshold)
    return float(silent / rms.size)


def mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 64,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """
    Строит mel-спектрограмму в децибелах.
    """
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    return librosa.power_to_db(mel, ref=np.max)


def normalize_waveform(waveform: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """
    Нормализует сигнал по пику.
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    max_amp = np.max(np.abs(waveform)) if waveform.size else 0.0
    if max_amp == 0.0:
        return waveform
    return waveform * (peak / max_amp)


def batch_convert(
    inputs: Iterable[tuple[Path, Path]],
    *,
    sample_rate: int,
    channels: int,
    force: bool = False,
) -> list[Path]:
    """
    Конвертирует набор файлов и возвращает список выходных путей.
    """
    outputs = []
    for src, dst in inputs:
        converted = convert_to_wav(
            src,
            dst,
            sample_rate=sample_rate,
            channels=channels,
            force=force,
        )
        outputs.append(converted)
    return outputs
