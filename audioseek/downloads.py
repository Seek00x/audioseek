"""Verified model downloads with visible progress and bounded retries."""
from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class ModelDownloadError(RuntimeError):
    """A model file could not be downloaded and verified."""


def _size(value):
    if value < 1024:
        return f"{value} B"
    if value < 1024**2:
        return f"{value / 1024:.1f} KiB"
    return f"{value / 1024**2:.1f} MiB"


def _digests(path: Path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
            git.update(block)
    return size, sha.hexdigest(), git.hexdigest()


def _verify(path: Path, item: dict):
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        return None
    size, sha, git = _digests(path)
    expected_sha = item.get("lfs_sha256")
    valid = sha == expected_sha if expected_sha else git == item["git_blob_sha1"]
    return sha if valid else None


def fetch_verified_file(path, item, url, report=print, attempts=3):
    """Publish only verified bytes; completed files survive later failures.

    A retry starts the current file again. It does not append unverified bytes
    or alter files that already match the pinned upstream metadata.
    """
    path = Path(path)
    if attempts < 1:
        raise ValueError("At least one download attempt is required")
    digest = _verify(path, item)
    if digest:
        return digest
    if path.exists():
        report(f"  Replacing invalid cache file: {item['filename']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    total = item["bytes"]
    last_error = None
    for attempt in range(1, attempts + 1):
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex[:8]}.part")
        report(f"  Downloading {item['filename']} ({_size(total)}), attempt {attempt}/{attempts}")
        received, last_percent, last_time = 0, -1, 0.0
        try:
            request = Request(url, headers={"User-Agent": "AudioSeek/0.1"})
            with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > total:
                        raise ValueError("Downloaded file is larger than the pinned size")
                    stream.write(block)
                    percent = int(100 * received / total)
                    now = time.monotonic()
                    if percent == 100 or (percent >= last_percent + 5 and now - last_time >= 1):
                        report(f"    {item['filename']}: {_size(received)} / {_size(total)} ({percent}%)")
                        last_percent, last_time = percent, now
            digest = _verify(temporary, item)
            if digest is None:
                raise ValueError("Downloaded file failed its pinned size or checksum check")
            temporary.replace(path)
            report(f"  Verified {item['filename']}")
            return digest
        except (OSError, ValueError) as error:
            last_error = error
            # A missing/private revision cannot be fixed by rapid retries.
            if isinstance(error, HTTPError) and error.code not in {408, 429, 500, 502, 503, 504}:
                break
            if attempt < attempts:
                reason = f"HTTP {error.code}" if isinstance(error, HTTPError) else type(error).__name__
                report(f"  Download interrupted ({reason}). Retrying this file; verified files are kept.")
                time.sleep(min(2 ** (attempt - 1), 4))
        finally:
            temporary.unlink(missing_ok=True)
    reason = f"HTTP {last_error.code}" if isinstance(last_error, HTTPError) else str(last_error)
    raise ModelDownloadError(
        f"Could not prepare {item['filename']}: {reason}. "
        "Check access to huggingface.co and available disk space, then run the same command again. "
        "Already verified model files will be reused."
    ) from last_error


def download_models(root, report=print):
    """Download both checkpoints without importing inference libraries."""
    from .fetch_semantic_model import fetch as fetch_search
    from .fetch_asr_model import fetch as fetch_speech
    report("[1/2] Preparing the search model...")
    semantic = fetch_search(root, report=report)
    report("[2/2] Preparing the speech model...")
    speech = fetch_speech(root, report=report)
    total = sum(item["bytes"] for model in (semantic, speech) for item in model["files"])
    report(f"Models ready: {total / 1024**2:.1f} MiB of verified local files.")
    return {"search": semantic, "speech": speech}
