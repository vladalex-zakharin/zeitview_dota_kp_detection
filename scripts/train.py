import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DOTAKeypointHeatmapDataset
from model import KeypointHeatmapResNet
from utils import HEATMAP_SIZE
import torchvision.transforms as T
from tqdm import tqdm
from utils import get_transforms

# Configuration
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset & DataLoader
transform = get_transforms()

dataset = DOTAKeypointHeatmapDataset(
    images_dir="data/raw/train/images",
    json_path="data/processed/train_keypoints.json",
    transform=transform
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = KeypointHeatmapResNet(output_size=HEATMAP_SIZE).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, heatmaps in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = images.to(DEVICE)
        heatmaps = heatmaps.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, heatmaps)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"models/checkpoints/epoch_{epoch+1}.pth")

print("Training complete.")