"""Run SAM3 prediction through Ultralytics.

Example:
    python scripts/predict_ultralytics_sam3.py --text person --text car
    python scripts/predict_ultralytics_sam3.py --source IMG_2838.MOV --text person
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from ultralytics.models.sam import SAM3SemanticPredictor, SAM3VideoSemanticPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "source"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VIDEO_SUFFIXES = {
    ".asf",
    ".avi",
    ".gif",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
    ".webm",
    ".wmv",
}
MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES


def resolve_source(value: str) -> Path:
    source = Path(value)
    if source.is_dir():
        media = sorted(path for path in source.iterdir() if path.suffix.lower() in MEDIA_SUFFIXES)
        if not media:
            raise FileNotFoundError(f"No image or video files found in {source}")
        return media[0]
    if source.exists() or source.is_absolute():
        return source
    source_in_data = DEFAULT_SOURCE_DIR / value
    if source_in_data.is_dir():
        media = sorted(path for path in source_in_data.iterdir() if path.suffix.lower() in MEDIA_SUFFIXES)
        if not media:
            raise FileNotFoundError(f"No image or video files found in {source_in_data}")
        return media[0]
    if source_in_data.exists():
        return source_in_data
    return source


def parse_bbox(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be x1,y1,x2,y2")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox values must be numbers") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict with SAM3 via Ultralytics.")
    parser.add_argument("--model", default="sam3.pt", help="Path to sam3.pt.")
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE_DIR),
        help="Input image/video path, file name inside data/source, or data/source directory.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for saved visualized outputs.")
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Text prompt. Repeat this flag for multiple prompts.",
    )
    parser.add_argument(
        "--bbox",
        action="append",
        type=parse_bbox,
        default=[],
        help="Exemplar box as x1,y1,x2,y2. Repeat this flag for multiple boxes.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size.")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 inference.")
    parser.add_argument("--no-save", action="store_true", help="Do not save visualized outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model_path = Path(args.model)
    source_path = resolve_source(args.source)
    output_dir = Path(args.output_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source image/video not found: {source_path}")
    if not args.text and not args.bbox:
        raise ValueError("Provide at least one --text prompt or --bbox exemplar.")
    output_dir.mkdir(parents=True, exist_ok=True)

    call_args: dict[str, object] = {}
    if args.text:
        call_args["text"] = args.text
    if args.bbox:
        call_args["bboxes"] = args.bbox

    overrides = {
        "model": str(model_path),
        "task": "segment",
        "mode": "predict",
        "compile": False,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "half": not args.no_half,
        "save": not args.no_save,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "exist_ok": True,
    }

    if source_path.suffix.lower() in VIDEO_SUFFIXES:
        predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        if args.bbox:
            call_args["labels"] = [1] * len(args.bbox)
        results = predictor(source=str(source_path), stream=True, **call_args)
        frame_count = sum(1 for _ in results)
        print(f"Completed video prediction. Frames processed: {frame_count}")
        return

    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.set_image(str(source_path))
    results = predictor(**call_args)
    result_count = len(results) if results is not None else 0
    print(f"Completed image prediction. Results: {result_count}")


if __name__ == "__main__":
    main()
