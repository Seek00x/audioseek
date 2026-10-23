"""Download only the four pinned, checksummed CTranslate2 ASR model files."""
import json
from datetime import datetime, timezone
from pathlib import Path

from .io import CONFIG_ROOT, DEFAULT_ROOT
from .downloads import fetch_verified_file
ROOT = DEFAULT_ROOT


def fetch(root=ROOT, report=print):
    root = Path(root).expanduser().resolve()
    (root / "data").mkdir(parents=True, exist_ok=True)
    config = json.loads((CONFIG_ROOT / "upload.json").read_text("utf-8"))
    metadata = json.loads((CONFIG_ROOT / "asr_model_files.json").read_text("utf-8"))
    if metadata["checkpoint"] != config["asr_checkpoint"] or metadata["revision"] != config["asr_revision"]:
        raise ValueError("ASR version mismatch")
    directory = (root / config["asr_model_directory"]).resolve()
    directory.relative_to(root.resolve())
    files = []
    for item in metadata["files"]:
        path = directory / item["filename"]
        path.resolve().relative_to(directory)
        url = f"https://huggingface.co/{metadata['checkpoint']}/resolve/{metadata['revision']}/{item['filename']}"
        digest = fetch_verified_file(path, item, url, report=report)
        files.append({"path": path.relative_to(root).as_posix(), "bytes": item["bytes"], "sha256": digest, "url": url})
    report(f"  Speech checkpoint verified ({len(files)} files).")
    manifest = {"checkpoint": metadata["checkpoint"], "revision": metadata["revision"], "files": files,
                "verified_at_utc": datetime.now(timezone.utc).isoformat()}
    target = root / "data/asr_model_manifest.json"
    if target.exists():
        old = json.loads(target.read_text("utf-8"))
        if old["files"] != files:
            raise ValueError("ASR manifest changed")
        return old
    target.write_text(json.dumps(manifest, indent=2)+"\n", "utf-8")
    return manifest


if __name__ == "__main__":
    result = fetch()
    print(json.dumps({"files": len(result["files"]), "revision": result["revision"]}, indent=2))
