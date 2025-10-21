import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple


def find_json_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.json")))
        elif p.suffix.lower() == ".json":
            files.append(p)
    return files


def parse_count_from_label(label: str) -> int:
    # Accept formats: "count_7", "7", or any label ending with digits (e.g., "chips_7")
    if label.startswith("count_"):
        try:
            return int(label.split("_")[1])
        except Exception:
            pass
    if label.isdigit():
        return int(label)
    m = re.search(r"(\d+)$", label)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse count from label: {label}")


def polygon_to_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max, y_max


def to_yolo_line(x_min: float, y_min: float, x_max: float, y_max: float, img_w: int, img_h: int, class_id: int) -> str:
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_file(input_json: Path, labels_out_dir: Path, max_k: int = 0) -> Path:
    data = json.loads(input_json.read_text())
    img_w = int(data.get("imageWidth"))
    img_h = int(data.get("imageHeight"))
    image_path = data.get("imagePath")

    stem = Path(image_path).stem if image_path else input_json.stem
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = labels_out_dir / f"{stem}.txt"

    lines: List[str] = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        points = shape.get("points", [])
        if not points:
            continue
        try:
            count = parse_count_from_label(label)
        except ValueError:
            # Skip unknown labels
            continue
        if max_k and not (1 <= count <= max_k):
            # Skip out-of-range counts when max_k is specified
            continue
        class_id = count - 1  # class indices: 0..K-1 for counts 1..K
        x_min, y_min, x_max, y_max = polygon_to_bbox(points)
        line = to_yolo_line(x_min, y_min, x_max, y_max, img_w, img_h, class_id)
        lines.append(line)

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert LabelMe JSON to YOLOv8 detection labels")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSON files or directories")
    parser.add_argument("--out-labels", required=True, type=Path, help="Output labels directory")
    parser.add_argument("--max-k", type=int, default=0, help="Max count K (optional; filters out-of-range)")
    args = parser.parse_args()

    files = find_json_files(args.inputs)
    if not files:
        print("No JSON files found.")
        return 1
    for jf in files:
        out = convert_file(jf, args.out_labels, max_k=args.max_k)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


