import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DOTAKeypointDataset
from model import KeypointResNet
import torchvision.transforms as T
from tqdm import tqdm

NUM_KEYPOINTS = 10
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

dataset = DOTAKeypointDataset(
    images_dir="data/raw/train/images",
    json_path="data/processed/train_keypoints.json",
    transform=transform,
    max_keypoints=NUM_KEYPOINTS
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = KeypointResNet(num_keypoints=NUM_KEYPOINTS).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, keypoints in tqdm(loader):
        images = images.to(DEVICE)
        keypoints = keypoints.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")
    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"models/checkpoints/epoch_{epoch+1}.pth")
