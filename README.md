# DOTA Center Keypoint Detection (Heatmap Model)

This repo is a full pipeline for detecting object centers (keypoints) in DOTA aerial images using a heatmap-based CNN model.

## Why Heatmaps?

Directly predicting (x, y) coords is hard for small, dense objects. Instead, I train a model to output a 2D heatmap — each bright spot is an object center.

It works better because:
- Handles any number of objects
- Easier for CNNs to learn
- More accurate for dense scenes

## How It Works
- ResNet18 backbone
- Upsample to output a `(1, H, W)` heatmap
- Train with MSE loss between predicted and GT heatmaps
- Extract (x, y) keypoints at inference using peak detection

## Project Structure

```
scripts/         # all code
├── model.py     # CNN with heatmap head
├── dataset.py   # dataset loading + heatmap labels
├── utils.py     # heatmap creation + constants
├── train.py     # training loop
├── infer.py     # run model on 1 image
├── preprocess.py     # bounding box files => center keypoints, and saves them to train_keypoints.json
└── evaluate.py  # run model on all images

data/
├── raw/
│   ├── train/images/     # raw DOTA images (not included)
│   └── train/labelTxt/   # annotation txts
├── processed/
│   └── train_keypoints.json

models/checkpoints/   # model weights (ignored)
outputs/               # inference results (ignored)
```

## Setup

1. Download DOTA images + annotations:
   - Put `.png` images in `data/raw/train/images/`
   - Put `.txt` files in `data/raw/train/labelTxt/`

2. Extract keypoints:
```bash
python scripts/preprocess.py
```

3. Install deps:
```bash
conda create -n dota-env python=3.11 -y
conda activate dota-env
pip install -r requirements.txt
```

4. Train:
```bash
python scripts/train.py
```

5. Inference:
```bash
python scripts/infer.py
```

6. Evaluate all:
```bash
python scripts/evaluate.py
```

## Config

- Input image size: 1024×1024 (default)
- Heatmap size: 256×256
- Edit in `utils.py`: `HEATMAP_SIZE`, `INPUT_IMAGE_SIZE`, `GAUSSIAN_RADIUS`

## Alternatives / Ideas

- Could use YOLOv8 / CenterNet for keypoints
- Use Albumentations for data aug
- Add validation loop, TTA, early stopping
- Visualize with TensorBoard or wandb
- Serve with FastAPI or Flask

## Notes

I've tested only a small subset of DOTA (around 500 images) for the interest of time, and couldn't see a satisfiable result. This project focuses more on clean architecture and process rather than achieving SOTA accuracy.

- Don't commit raw images, `outputs/`, or `checkpoints/`
- You *can* commit `train_keypoints.json` if it's small

---

Made by Vlad Zakharin

