# SAM3 Ultralytics Predict

This folder is trimmed for SAM3 prediction through Ultralytics.

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

The tracker also exports YOLOv8 training files by default:

- `data/output/IMG_2838/IMG_2838_sam3_track.mp4`: preview MP4
- `data/output/IMG_2838/images/`: extracted frame images
- `data/output/IMG_2838/labels/`: YOLO label txt files
- `data/output/IMG_2838/dataset.yaml`: dataset config

Default label format is YOLOv8 segmentation:

```text
class x1 y1 x2 y2 x3 y3 ...
```

Use detection box labels instead with:

```powershell
& 'D:\Users\miniconda3\condabin\conda.bat' run -n DeepLearning python scripts\track_sam3_points.py --yolo-format box
```

Use `--class-id` and `--class-name` to set the YOLO class:

```powershell
& 'D:\Users\miniconda3\condabin\conda.bat' run -n DeepLearning python scripts\track_sam3_points.py --class-id 0 --class-name "target"
```
