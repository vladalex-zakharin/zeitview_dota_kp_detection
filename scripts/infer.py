# scripts/infer.py

import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
from model import KeypointHeatmapResNet
from utils import extract_keypoints_from_heatmap
from utils import HEATMAP_SIZE
from utils import get_transforms
import matplotlib.pyplot as plt
import os

# Config
CHECKPOINT_PATH = "models/checkpoints/epoch_3.pth"
IMAGE_PATH = "data/raw/train/images/P0001.png"
OUTPUT_PATH = "outputs/infer_P0001.png"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = KeypointHeatmapResNet(output_size=HEATMAP_SIZE).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# Image preprocessing
transform = get_transforms()

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# Inference
with torch.no_grad():
    pred_heatmap = model(input_tensor)

heatmap_tensor = pred_heatmap[0][0].cpu().detach()

plt.imshow(heatmap_tensor, cmap="hot")
plt.title("Predicted Heatmap - Keypoint 0")
plt.colorbar()
plt.savefig("outputs/pred_heatmap.png")  # Save if needed
plt.show()

keypoints = extract_keypoints_from_heatmap(pred_heatmap[0], threshold=0.3)

scale_x = image.width / HEATMAP_SIZE[1]
scale_y = image.height / HEATMAP_SIZE[0]
scaled_keypoints = [(round(x * scale_x), round(y * scale_y)) for x, y in keypoints]
print(f"Detected keypoints: {scaled_keypoints}")

# Draw and save visualization
draw = ImageDraw.Draw(image)
for x, y in scaled_keypoints:
    draw.ellipse([
        (x - 3, y - 3),
        (x + 3, y + 3)
    ], fill="red")


os.makedirs("outputs", exist_ok=True)
image.save(OUTPUT_PATH)
print(f"Saved output to {OUTPUT_PATH}")