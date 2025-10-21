import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def find_json_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.json")))
        elif p.suffix.lower() == ".json":
            files.append(p)
    return files


def parse_count_from_label(label: str, mapping: Optional[Dict[str, int]] = None, default_count: Optional[int] = None) -> int:
    # Explicit mapping first
    if mapping and label in mapping:
        return mapping[label]
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
    if default_count is not None:
        return default_count
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


def _get_image_size_from_json(data: dict, json_path: Path) -> Tuple[int, int]:
    w = data.get("imageWidth")
    h = data.get("imageHeight")
    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
        return int(w), int(h)
    # Fallback: try to open the referenced image to get size
    image_path = data.get("imagePath")
    if image_path:
        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = (json_path.parent / img_path).resolve()
        try:
            from PIL import Image  # type: ignore

            with Image.open(img_path) as im:
                return im.width, im.height
        except Exception:
            pass
    raise ValueError("Missing image dimensions and unable to infer from imagePath")


def convert_file(input_json: Path, labels_out_dir: Path, max_k: int = 0, mapping: Optional[Dict[str, int]] = None, default_count: Optional[int] = None) -> Path:
    data = json.loads(input_json.read_text())
    img_w, img_h = _get_image_size_from_json(data, input_json)
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
            count = parse_count_from_label(label, mapping=mapping, default_count=default_count)
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
    parser.add_argument("--map", action="append", default=[], help="Label-to-count mapping entries, e.g. --map chip=5 (can repeat)")
    parser.add_argument("--default-count", type=int, default=None, help="Fallback count if label can't be parsed")
    args = parser.parse_args()

    files = find_json_files(args.inputs)
    if not files:
        print("No JSON files found.")
        return 1
    mapping: Dict[str, int] = {}
    for entry in args.map:
        if "=" in entry:
            k, v = entry.split("=", 1)
            try:
                mapping[k] = int(v)
            except ValueError:
                print(f"Skipping invalid map entry: {entry}")
        else:
            print(f"Skipping invalid map entry: {entry}")
    for jf in files:
        out = convert_file(jf, args.out_labels, max_k=args.max_k, mapping=mapping or None, default_count=args.default_count)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


