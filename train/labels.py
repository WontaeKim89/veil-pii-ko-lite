"""라벨 사전: labels.yaml → BIOES 태그 집합 (32×4+1 = 129)."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_labels():
    cfg = yaml.safe_load(open(ROOT / "configs/labels.yaml"))
    ents = list(cfg["labels"])
    tags = ["O"] + [f"{p}-{e}" for e in ents for p in ("B", "I", "E", "S")]
    return ents, tags, {t: i for i, t in enumerate(tags)}, {i: t for i, t in enumerate(tags)}
