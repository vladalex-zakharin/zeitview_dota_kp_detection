import torch
import torch.nn as nn
import torchvision.models as models

class KeypointHeatmapResNet(nn.Module):
    def __init__(self, output_size=(128, 128), pretrained=True):
        super(KeypointHeatmapResNet, self).__init__()

        self.output_size = output_size  # heatmap size

        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool + fc

        # Head: upsample to heatmap resolution
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16 -> 32
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32 -> 64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64 -> 128
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Upsample(size=output_size, mode='bilinear', align_corners=False),   # 128 -> to custom output size
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x  # shape: (B, 1, H, W)
