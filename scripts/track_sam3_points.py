"""Track one object in a video with first-frame positive/negative points.

Controls in interactive mode:
    Left click  = positive point
    Right click = negative point
    u           = undo last point
    c or Enter  = run tracking
    q or Esc    = quit

Example:
    python scripts/track_sam3_points.py --model sam3.pt
    python scripts/track_sam3_points.py --source IMG_2838.MOV --point 320,240,1 --point 450,250,0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics.models.sam import SAM3VideoPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "source"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
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


def resolve_source(value: str | None) -> Path:
    if value is None:
        videos = sorted(path for path in DEFAULT_SOURCE_DIR.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)
        if not videos:
            raise FileNotFoundError(f"No video files found in {DEFAULT_SOURCE_DIR}")
        return videos[0]

    source = Path(value)
    if source.exists() or source.is_absolute():
        return source
    source_in_data = DEFAULT_SOURCE_DIR / value
    if source_in_data.exists():
        return source_in_data
    return source


def parse_point(value: str) -> tuple[float, float, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("point must be x,y,label where label is 1 or 0")
    try:
        x, y = float(parts[0]), float(parts[1])
        label = int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("point values must be numeric") from exc
    if label not in {0, 1}:
        raise argparse.ArgumentTypeError("point label must be 1 for positive or 0 for negative")
    return x, y, label


def read_first_frame(source: Path):
    cap = cv2.VideoCapture(str(source))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from {source}")
    return frame


def get_video_fps(source: Path, vid_stride: int) -> float:
    cap = cv2.VideoCapture(str(source))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if fps <= 0:
        fps = 30.0
    return max(fps / max(vid_stride, 1), 1.0)


def default_output_path(source: Path, output_dir: Path) -> Path:
    return output_dir / source.stem / f"{source.stem}_sam3_track.mp4"


def default_run_dir(source: Path, output_dir: Path) -> Path:
    return output_dir / source.stem


def format_yolo_value(value: float) -> str:
    return f"{min(max(float(value), 0.0), 1.0):.6f}"


def write_dataset_yaml(output_dir: Path, class_id: int, class_name: str) -> Path:
    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: images",
                "val: images",
                "names:",
                f"  {class_id}: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def write_yolo_label(result, label_path: Path, class_id: int, label_format: str) -> int:
    lines: list[str] = []
    if label_format == "segment":
        if result.masks is not None:
            for segment in result.masks.xyn:
                if len(segment) < 3:
                    continue
                coords = [format_yolo_value(value) for point in segment for value in point]
                lines.append(f"{class_id} {' '.join(coords)}")
    else:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes.xywhn.cpu().numpy():
                coords = [format_yolo_value(value) for value in box[:4]]
                lines.append(f"{class_id} {' '.join(coords)}")

    label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return len(lines)


def collect_points_interactively(source: Path, max_display_size: int) -> tuple[list[list[float]], list[int]]:
    frame = read_first_frame(source)
    height, width = frame.shape[:2]
    scale = min(1.0, max_display_size / max(width, height))
    display_size = (int(width * scale), int(height * scale))
    points: list[list[float]] = []
    labels: list[int] = []

    window = "SAM3 first frame prompts"

    def redraw() -> None:
        canvas = frame.copy()
        for idx, ((x, y), label) in enumerate(zip(points, labels), start=1):
            color = (0, 220, 0) if label == 1 else (0, 0, 255)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 8, color, -1)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 10, (255, 255, 255), 2)
            cv2.putText(
                canvas,
                str(idx),
                (int(round(x)) + 12, int(round(y)) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        shown = cv2.resize(canvas, display_size, interpolation=cv2.INTER_AREA) if scale != 1.0 else canvas
        cv2.imshow(window, shown)

    def on_mouse(event, x, y, _flags, _userdata) -> None:
        if event not in {cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN}:
            return
        points.append([x / scale, y / scale])
        labels.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
        redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in {13, ord("c")}:
            break
        if key in {27, ord("q")}:
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("Tracking cancelled")
        if key == ord("u") and points:
            points.pop()
            labels.pop()
            redraw()

    cv2.destroyWindow(window)
    return points, labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track a SAM3 object in video using first-frame point prompts.")
    parser.add_argument("--model", default="sam3.pt", help="Path to sam3.pt.")
    parser.add_argument(
        "--source",
        default=None,
        help="Input video path or file name inside data/source. Defaults to the first video in data/source.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for saved visualized outputs.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output MP4 path. Defaults to data/output/<source-name>/<source-name>_sam3_track.mp4.",
    )
    parser.add_argument(
        "--point",
        action="append",
        type=parse_point,
        default=[],
        help="Point prompt as x,y,label. label: 1=positive, 0=negative. Repeat for multiple points.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=840, help="Inference image size.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Video frame stride.")
    parser.add_argument("--max-display-size", type=int, default=1280, help="Max side length for the first-frame UI.")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 inference.")
    parser.add_argument("--no-save", action="store_true", help="Do not save visualized output.")
    parser.add_argument("--no-preview", action="store_true", help="Disable live preview while tracking.")
    parser.add_argument(
        "--yolo-format",
        choices=("segment", "box"),
        default="segment",
        help="YOLOv8 label format to export. Use segment for YOLOv8-seg or box for YOLOv8-detect.",
    )
    parser.add_argument("--class-id", type=int, default=0, help="Class id written to YOLO label files.")
    parser.add_argument("--class-name", default="object", help="Class name written to dataset.yaml.")
    parser.add_argument("--no-save-yolo", action="store_true", help="Disable YOLO label export.")
    parser.add_argument("--no-save-frames", action="store_true", help="Do not save frame images for YOLO training.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = Path(args.model)
    source_path = resolve_source(args.source)
    output_dir = Path(args.output_dir)
    run_dir = default_run_dir(source_path, output_dir)
    output_path = Path(args.output) if args.output else default_output_path(source_path, output_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_dir = output_path.parent / "labels"
    images_dir = output_path.parent / "images"
    if not args.no_save_yolo:
        labels_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_save_frames:
            images_dir.mkdir(parents=True, exist_ok=True)
        dataset_yaml = write_dataset_yaml(output_path.parent, args.class_id, args.class_name)
    else:
        dataset_yaml = None

    if args.point:
        points = [[x, y] for x, y, _label in args.point]
        labels = [_label for _x, _y, _label in args.point]
    else:
        points, labels = collect_points_interactively(source_path, args.max_display_size)

    if not points:
        raise ValueError("Provide at least one positive point.")
    if 1 not in labels:
        raise ValueError("At least one positive point is required.")

    overrides = {
        "model": str(model_path),
        "task": "segment",
        "mode": "predict",
        "conf": args.conf,
        "imgsz": args.imgsz,
        "half": not args.no_half,
        "save": False,
        "vid_stride": args.vid_stride,
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
    }
    predictor = SAM3VideoPredictor(overrides=overrides)

    # Wrap points once so all positive/negative clicks refine the same object.
    results = predictor(source=str(source_path), points=[points], labels=[labels], stream=True)
    preview = not args.no_preview
    preview_window = "SAM3 tracking preview"
    if preview:
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)

    writer = None
    fps = get_video_fps(source_path, args.vid_stride)
    frame_count = 0
    label_count = 0
    try:
        for frame_count, result in enumerate(results, start=1):
            plotted = result.plot()
            if writer is None and not args.no_save:
                height, width = plotted.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open MP4 writer: {output_path}")
                print(f"Writing MP4 to {output_path}")
            if writer is not None:
                writer.write(plotted)

            if not args.no_save_yolo:
                item_stem = f"{source_path.stem}_{frame_count:06d}"
                label_count += write_yolo_label(
                    result,
                    labels_dir / f"{item_stem}.txt",
                    args.class_id,
                    args.yolo_format,
                )
                if not args.no_save_frames:
                    cv2.imwrite(str(images_dir / f"{item_stem}.jpg"), result.orig_img)

            if preview:
                cv2.imshow(preview_window, plotted)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    print("Preview stopped by user.")
                    break

            if frame_count % 25 == 0:
                print(f"Processed {frame_count} frames")
    finally:
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyWindow(preview_window)

    print(f"Completed tracking. Frames processed: {frame_count}")
    if not args.no_save and frame_count > 0:
        print(f"Saved MP4: {output_path}")
    if not args.no_save_yolo and frame_count > 0:
        print(f"Saved YOLOv8 {args.yolo_format} labels: {labels_dir}")
        if not args.no_save_frames:
            print(f"Saved YOLOv8 frame images: {images_dir}")
        print(f"YOLO objects written: {label_count}")
        print(f"Dataset YAML: {dataset_yaml}")


if __name__ == "__main__":
    main()
