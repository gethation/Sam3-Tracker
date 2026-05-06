# SAM3 Ultralytics Predict

This folder is trimmed for SAM3 prediction through Ultralytics.

See `INSTALL.md` for environment setup.
PyTorch compile is disabled by default to avoid Triton setup issues. Add
`--compile` if your local PyTorch/Triton setup supports it.

## Kept Files

- `data/source/`: input images and videos
- `data/output/`: saved prediction/tracking results
- `scripts/predict_ultralytics_sam3.py`: image/video prediction helper
- `scripts/track_sam3_points.py`: first-frame point tracking helper

## Download `sam3.pt`

The official SAM3 weights are gated on Hugging Face. Log in and accept access for
`facebook/sam3`, then run:

```powershell
& 'D:\Users\miniconda3\condabin\conda.bat' run -n DeepLearning hf download facebook/sam3 sam3.pt --local-dir .
```

## Predict

```powershell
& 'D:\Users\miniconda3\condabin\conda.bat' run -n DeepLearning python scripts\predict_ultralytics_sam3.py --model sam3.pt --source "IMG_2838.MOV" --text "person"
```

Use repeated `--text` flags for multiple prompts:

```powershell
& 'D:\Users\miniconda3\condabin\conda.bat' run -n DeepLearning python scripts\predict_ultralytics_sam3.py --model sam3.pt --source "IMG_2838.MOV" --text "person" --text "car"
```

If `--source` is omitted, the script reads from `data/source`.
Saved visualized outputs go to `data/output`.

## Track One Object With First-Frame Points

Use this when you want to choose a specific object with positive and negative
points on the first frame.

```powershell
& 'D:\Users\miniconda3\condabin\conda.bat' run -n DeepLearning python scripts\track_sam3_points.py --model sam3.pt
```

The tracking helper defaults to `--imgsz 840`, which is lighter for a 6GB GPU
and divisible by SAM3's stride.
Use Triton/TorchInductor locally with:

```powershell
& 'D:\Users\miniconda3\condabin\conda.bat' run -n DeepLearning python scripts\track_sam3_points.py --model sam3.pt --compile
```

Controls:

- Left click: positive point
- Right click: negative point
- `u`: undo last point
- `c` or Enter: run tracking
- `q` or Esc: quit

During tracking, a preview window shows the mask result frame by frame.
Press `q` or Esc in the preview window to stop early.

You can also pass points directly:

```powershell
& 'D:\Users\miniconda3\condabin\conda.bat' run -n DeepLearning python scripts\track_sam3_points.py --model sam3.pt --source "IMG_2838.MOV" --point 320,240,1 --point 450,250,0
```

If `--source` is omitted, the script uses the first video found in `data/source`.
Saved tracking output goes under `data/output/<source-name>/`.

Use `--no-preview` to disable the live preview window.
Use `--output "data\output\custom_name.mp4"` to choose a different MP4 path.

## Output Files

For an input video named `IMG_2838.MOV`, tracking output is written under:

```text
data/output/IMG_2838/
```

The folder contains:

```text
data/output/IMG_2838/
  IMG_2838_sam3_track.mp4
  images/
    IMG_2838_000001.jpg
    IMG_2838_000002.jpg
    ...
  labels/
    IMG_2838_000001.txt
    IMG_2838_000002.txt
    ...
  labels-seg/
    IMG_2838_000001.txt
    IMG_2838_000002.txt
    ...
  dataset.yaml
```

`IMG_2838_sam3_track.mp4` is the rendered preview video with SAM3 masks drawn on
top of the original frames.

`images/` contains the frame images used for YOLO training. Each image has a
matching label file with the same stem. For example:

```text
images/IMG_2838_000001.jpg
labels/IMG_2838_000001.txt
labels-seg/IMG_2838_000001.txt
```

`labels/` is the primary YOLOv8 detection dataset. Each `.txt` file uses box
labels:

```text
class x_center y_center width height
```

The coordinates are normalized from 0 to 1. `x_center` and `width` are relative
to image width. `y_center` and `height` are relative to image height.

`labels-seg/` is the auxiliary YOLOv8 segmentation dataset. Each `.txt` file uses
polygon labels:

```text
class x1 y1 x2 y2 x3 y3 ...
```

Each pair is one normalized polygon point. These files contain many more numbers
because they describe the object outline rather than a rectangular box.

`dataset.yaml` points YOLOv8 to the `images/` and primary `labels/` folders:

```yaml
path: data/output/IMG_2838
train: images
val: images
names:
  0: object
```

The default `--yolo-format both` writes both formats. To write only one format:

```powershell
& 'D:\Users\miniconda3\condabin\conda.bat' run -n DeepLearning python scripts\track_sam3_points.py --yolo-format segment
& 'D:\Users\miniconda3\condabin\conda.bat' run -n DeepLearning python scripts\track_sam3_points.py --yolo-format box
```

Use `--class-id` and `--class-name` to set the YOLO class:

```powershell
& 'D:\Users\miniconda3\condabin\conda.bat' run -n DeepLearning python scripts\track_sam3_points.py --class-id 0 --class-name "target"
```
