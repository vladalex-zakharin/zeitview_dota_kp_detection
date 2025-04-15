import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from utils import create_heatmap, HEATMAP_SIZE

class DOTAKeypointHeatmapDataset(Dataset):
    def __init__(self, images_dir, json_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform or T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])

        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.images_dir, item['image_file'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Downscale keypoints to heatmap resolution
        keypoints = item['keypoints']
        resized_kps = [
            (int(x * HEATMAP_SIZE[1] / image.shape[2]), int(y * HEATMAP_SIZE[0] / image.shape[1]))
            for x, y in keypoints
        ]
        heatmap = create_heatmap(resized_kps)  # (1, H, W)

        return image, heatmap