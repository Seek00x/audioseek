"""Application resources and local runtime storage."""
from pathlib import Path
import json
import os

CONFIG_ROOT = Path(__file__).resolve().parent / "config"
DEFAULT_ROOT = Path(os.environ.get("AUDIOSEEK_DATA_DIR", Path.home() / ".cache" / "audioseek")).expanduser()

def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))

def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8")
