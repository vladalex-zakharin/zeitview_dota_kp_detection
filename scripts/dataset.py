import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class DOTAKeypointDataset(Dataset):
    def __init__(self, images_dir, json_path, transform=None, max_keypoints=10):
        self.images_dir = images_dir
        self.transform = transform or T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])
        self.max_keypoints = max_keypoints

        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.images_dir, item['image_file'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        keypoints = torch.tensor(item['keypoints'], dtype=torch.float32)
        if len(keypoints) < self.max_keypoints:
            pad = torch.zeros(self.max_keypoints - len(keypoints), 2)
            keypoints = torch.cat([keypoints, pad], dim=0)
        else:
            keypoints = keypoints[:self.max_keypoints]

        return image, keypoints
