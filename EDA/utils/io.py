from __future__ import annotations

import re
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import requests
from tqdm import tqdm


MAILRU_PATTERN = re.compile(r'dispatcher.*?weblink_get.*?url":"(.*?)"')


class DownloadError(RuntimeError):
    """Исключение для ошибок скачивания."""


def get_direct_file_link(mailru_file_url: str) -> str:
    """
    Преобразует публичную ссылку Mail.ru в прямой CDN-URL.
    """
    response = requests.get(mailru_file_url, timeout=30)
    if response.status_code != 200:
        raise DownloadError(
            f"Не удалось запросить {mailru_file_url}: статус {response.status_code}"
        )

    match = MAILRU_PATTERN.search(response.text)
    if not match:
        raise DownloadError("Не удалось выделить CDN-ссылку из ответа Mail.ru")

    cdn_base = match.group(1)
    parts = mailru_file_url.strip("/").split("/")[-3:]
    return f"{cdn_base}/{parts[0]}/{parts[1]}/{parts[2]}"


def download_file(
    url: str,
    destination: Path,
    *,
    force: bool = False,
    chunk_size: int = 8192,
    show_progress: bool = True,
) -> Path:
    """
    Скачивает файл по прямому URL на диск.
    """
    destination = Path(destination)
    if destination.exists() and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with destination.open("wb") as fh, tqdm(
            total=total_size if total_size else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {destination.name}",
            disable=not show_progress,
        ) as bar:
            for chunk in response.iter_content(chunk_size):
                fh.write(chunk)
                bar.update(len(chunk))
    return destination


def download_from_mailru(
    file_url: str,
    destination: Path,
    *,
    force: bool = False,
    show_progress: bool = True,
) -> Path:
    """
    Скачивает файл с Mail.ru, используя публичную ссылку.
    """
    direct = get_direct_file_link(file_url)
    return download_file(
        direct,
        destination=destination,
        force=force,
        show_progress=show_progress,
    )


def extract_archive(archive_path: Path, target_dir: Path, members: Optional[Iterable[str]] = None) -> None:
    """
    Распаковывает архив (tar.* или zip) в указанную директорию.
    """
    archive_path = Path(archive_path)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tar:
            if members:
                valid = [tar.getmember(m) for m in members if m in tar.getnames()]
                tar.extractall(path=target_dir, members=valid)
            else:
                tar.extractall(path=target_dir)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            if members:
                to_extract = [m for m in members if m in zf.namelist()]
                for name in to_extract:
                    zf.extract(name, path=target_dir)
            else:
                zf.extractall(path=target_dir)
    else:
        raise ValueError(f"Неподдерживаемый формат архива: {archive_path}")
