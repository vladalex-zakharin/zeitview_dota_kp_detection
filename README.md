# ZeitView - DOTA Keypoint Detection — Vlad Zakharin

This is a simple keypoint detection pipeline using a fixed-number regression model.

## Structure

- `scripts/model.py`: ResNet18 model with a fully connected head
- `scripts/dataset.py`: Loads images and center points from JSON
- `scripts/train.py`: Trains the model using MSE on fixed keypoints

## Training

```bash
python scripts/train.py
