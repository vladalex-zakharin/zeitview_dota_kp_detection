import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as T

class DOTAKeypointHeatmapDataset(Dataset):
    def __init__(self, images_dir, json_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform or T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])
        self.heatmap_size = (128, 128)
        self.gaussian_radius = 2

        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def create_heatmap(self, keypoints):
        H, W = self.heatmap_size
        heatmap = np.zeros((H, W), dtype=np.float32)
        for x, y in keypoints:
            x = int(x * W / 512)
            y = int(y * H / 512)
            if x < 0 or y < 0 or x >= W or y >= H:
                continue
            tmp = np.zeros((H, W), dtype=np.float32)
            tmp[y, x] = 1
            tmp = cv2.GaussianBlur(tmp, (0, 0), sigmaX=self.gaussian_radius, sigmaY=self.gaussian_radius)
            tmp = tmp / tmp.max()
            heatmap = np.maximum(heatmap, tmp)
        return torch.tensor(heatmap).unsqueeze(0)  # shape: (1, H, W)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.images_dir, item['image_file'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        keypoints = item['keypoints']
        heatmap = self.create_heatmap(keypoints)
        return image, heatmap
