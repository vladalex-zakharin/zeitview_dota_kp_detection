import os
import torch
from torch.utils.data import DataLoader
from dataset import DOTAKeypointHeatmapDataset
from model import KeypointHeatmapResNet
from utils import extract_keypoints_from_heatmap
from utils import HEATMAP_SIZE
from utils import get_transforms
from PIL import ImageDraw, Image
import torchvision.transforms as T
from tqdm import tqdm

# Config
CHECKPOINT_PATH = "models/checkpoints/epoch_10.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = "outputs/eval"
os.makedirs(SAVE_DIR, exist_ok=True)

# Dataset
dataset = DOTAKeypointHeatmapDataset(
    images_dir="data/raw/train/images",
    json_path="data/processed/train_keypoints.json"
)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Model
model = KeypointHeatmapResNet(output_size=HEATMAP_SIZE).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

transform = get_transforms()

# Evaluation loop
for idx, (image_tensor, _) in enumerate(tqdm(loader)):
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        pred_heatmap = model(image_tensor)
    keypoints = extract_keypoints_from_heatmap(pred_heatmap[0], threshold=0.3)

    # Load original image
    img_path = dataset.data[idx]['image_file']
    img_full = Image.open(os.path.join("data/raw/train/images", img_path)).convert("RGB")
    draw = ImageDraw.Draw(img_full)

    # Draw keypoints
    scale_x = img_full.width / HEATMAP_SIZE[1]
    scale_y = img_full.height / HEATMAP_SIZE[0]
    for x, y in keypoints:
        draw.ellipse([
            (x * scale_x - 3, y * scale_y - 3),
            (x * scale_x + 3, y * scale_y + 3)
        ], fill="red")

    img_full.save(os.path.join(SAVE_DIR, f"pred_{idx}.png"))

print(f"Saved evaluation results to {SAVE_DIR}")