<div align="center">

# Chin Detection (YOLO Pose Fine-Tuning)

**Fine-tuning YOLO11 Pose to detect a single keypoint: the chin. Custom annotation in CVAT, training in a notebook, ready-to-use weights.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111F68)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![CVAT](https://img.shields.io/badge/Annotation-CVAT-orange)

My first ML project · February 2025

</div>

---

## Overview

The goal is to teach a pretrained **YOLO11s-pose** model to find a keypoint that is not in the standard COCO skeleton: the **chin**. This is my first hands-on computer vision project: I labeled the data myself, set up the dataset, and fine-tuned the model end to end.

<div align="center">
  <img src="https://github.com/user-attachments/assets/e13e3aa4-72de-4551-bc23-eba758cbbde1" width="500" />
  <br />
  <sub>Model prediction on a validation image</sub>
</div>

Training logs, graphs and prediction batches are saved by Ultralytics in `runs/pose/train/`.

---

## 1. Model Preparation

The base model is the pretrained `yolo11s-pose.pt`. Ultralytics downloads it automatically on the first run.

## 2. Dataset Structure

```
dataset/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/        # no labels, used only for inference
│
├── labels/
│   ├── train/
│   └── val/
│
├── data.yaml
└── train_model.ipynb
```

## 3. Data Annotation

Data is labeled in CVAT or a similar tool. Each line in a `.txt` file has this format:

```
class x_center y_center width height kpt_x kpt_y visibility
```

Example:

```
0 0.493021 0.371556 0.05 0.05 0.493021 0.371556 2
```

| Parameter | Description |
|---|---|
| `class` | object class (`0` = chin) |
| `x_center, y_center` | center of the bounding box |
| `width, height` | size of the box |
| `kpt_x, kpt_y` | keypoint coordinates |
| `visibility` | `0` not labeled, `1` labeled but not visible, `2` visible |

The chin is a single point, so the bounding box is a small fixed-size square (0.05 x 0.05) centered on the keypoint.

## 4. `data.yaml`

```yaml
path: dataset

train: images/train
val: images/val
test: images/test

names:
  0: chin

kpt_shape: [1, 3]
flip_idx: [0]
```

- `names`: object classes (only `chin`).
- `kpt_shape: [1, 3]`: one keypoint, each with `(x, y, visibility)`.
- `flip_idx`: how keypoints are swapped when an image is flipped horizontally. With a single point it is `[0]`. For several points, for example nose (0), left eye (1), right eye (2), it would be `[0, 2, 1]`.

## 5. Training

```bash
pip install ultralytics mlflow
```

Open `train_model.ipynb`, check that the dataset and model paths are correct, and run all cells. Training metrics are logged with MLflow.

## 6. Output

After training, the best weights are saved to:

```
runs/pose/train/weights/best.pt
```

The same folder contains training graphs, validation batches and metrics.

## 7. Inference

```python
from ultralytics import YOLO

model = YOLO("runs/pose/train/weights/best.pt")
results = model("image.jpg")

# chin keypoint coordinates (x, y)
print(results[0].keypoints.xy)
```

---

## Possible improvements

- Larger and more diverse dataset (lighting, angles, partial occlusion).
- Real-time inference on webcam video.
- Export to ONNX for faster deployment.
