import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def yolo_to_xyxy(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2


def run(images_dir: Path, labels_dir: Path, out_dir: Path, class_map_yaml: Path) -> None:
    import yaml

    loaded = yaml.safe_load(class_map_yaml.read_text())
    # Normalize to a list indexed by class id
    if isinstance(loaded, dict):
        nm = loaded.get("names", loaded)
        if isinstance(nm, list):
            names = nm
        elif isinstance(nm, dict):
            if all(isinstance(k, int) for k in nm.keys()):
                # id -> name mapping
                names = [nm[i] for i in sorted(nm.keys())]
            elif all(isinstance(v, int) for v in nm.values()):
                # name -> id mapping
                max_id = max(nm.values())
                tmp = [None] * (max_id + 1)
                for name, idx in nm.items():
                    tmp[idx] = name
                names = [n if n is not None else str(i) for i, n in enumerate(tmp)]
            else:
                raise ValueError(
                    "names-yaml must provide names as list, id->name dict, or name->id dict"
                )
        else:
            raise ValueError(
                "names-yaml must provide names as list, id->name dict, or name->id dict"
            )
    elif isinstance(loaded, list):
        names = loaded
    else:
        raise ValueError("Unsupported names-yaml format")

    out_train = out_dir / "images" / "train"
    out_val = out_dir / "images" / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    for img_path in images_dir.glob("*.jpg"):
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            parts = ln.split()
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_to_xyxy((cx, cy, bw, bh), w, h)
            crop = img.crop((x1, y1, x2, y2))
            cls_name = names[cls_id]
            # simple split: 1 in 5 to val
            dst = out_val if i % 5 == 0 else out_train
            (dst / cls_name).mkdir(parents=True, exist_ok=True)
            crop.save(dst / cls_name / f"{stem}_{i}.jpg")


def main() -> int:
    p = argparse.ArgumentParser(description="Create classifier crops from YOLO labels")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--names-yaml", type=Path, required=True, help="YAML with 'names' list or id->name dict")
    args = p.parse_args()

    run(args.images, args.labels, args.out, args.names_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


