"""Fetch a pinned, standard Sentence Transformers model with verified files.

Only safetensors weights and built-in Transformer/Pooling/Normalize modules are
used. The checkpoint is the same immutable revision as the existing tokenizer.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .io import CONFIG_ROOT, DEFAULT_ROOT
from .downloads import fetch_verified_file
ROOT = DEFAULT_ROOT


def fetch(root=ROOT, report=print):
    root = Path(root).expanduser().resolve()
    (root / "data").mkdir(parents=True, exist_ok=True)
    config = json.loads((CONFIG_ROOT / "semantic.json").read_text("utf-8"))
    files = json.loads((CONFIG_ROOT / "semantic_model_files.json").read_text("utf-8"))
    if files["revision"] != config["revision"] or files["checkpoint"] != config["checkpoint"]:
        raise ValueError("Model metadata and configuration disagree")
    model_dir = root / config["model_directory"]
    model_dir.resolve().relative_to(root.resolve())
    records = []
    for item in files["files"]:
        target = model_dir / item["filename"]
        target.resolve().relative_to(model_dir.resolve())
        url = f"https://huggingface.co/{config['checkpoint']}/resolve/{config['revision']}/{item['filename']}"
        sha256 = fetch_verified_file(target, item, url, report=report)
        records.append({"path": target.relative_to(root).as_posix(), "url": url, "bytes": item["bytes"], "sha256": sha256})
    report(f"  Search checkpoint verified ({len(records)} files).")
    modules = json.loads((model_dir / "modules.json").read_text("utf-8"))
    types = [m["type"] for m in modules]
    if types != ["sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling", "sentence_transformers.models.Normalize"]:
        raise ValueError("Unexpected checkpoint modules")
    pooling = json.loads((model_dir / "1_Pooling/config.json").read_text("utf-8"))
    if pooling.get("word_embedding_dimension") != 384 or not pooling.get("pooling_mode_mean_tokens"):
        raise ValueError("Unexpected pooling/dimension")
    manifest = {"checkpoint": config["checkpoint"], "revision": config["revision"],
                "verified_at_utc": datetime.now(timezone.utc).isoformat(), "weight_format": "safetensors",
                "trust_remote_code": False, "modules": modules, "pooling": pooling, "files": records}
    path = root / "data/semantic_model_manifest.json"
    if path.exists():
        old = json.loads(path.read_text("utf-8"))
        if old["files"] != records:
            raise ValueError("Model manifest changed; refusing silent replacement")
        return old
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return manifest


if __name__ == "__main__":
    manifest = fetch()
    print(json.dumps({"revision": manifest["revision"], "verified_files": len(manifest["files"]),
                      "total_bytes": sum(r["bytes"] for r in manifest["files"])}, indent=2))
