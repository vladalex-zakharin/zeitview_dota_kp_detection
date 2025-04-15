import torch
import torch.nn as nn
import torchvision.models as models

class KeypointHeatmapResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(KeypointHeatmapResNet, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16 → 32
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32 → 64
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x  # shape: (B, 1, 64, 64)
