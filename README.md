# ZeitView - DOTA Keypoint Detection — Vlad Zakharin

## 🔁 Heatmap-Based Model (v2)

- Replaced fixed-size keypoint regression with a heatmap output model
- Model now outputs a `(1, 128, 128)` heatmap for each input image
- Training uses MSE between predicted and ground truth Gaussian blobs
- Heatmaps are generated inline inside `dataset.py`

## Training

```bash
python scripts/train.py
```