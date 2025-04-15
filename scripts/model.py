import torch
import torch.nn as nn
import torchvision.models as models

class KeypointResNet(nn.Module):
    def __init__(self, num_keypoints=10, pretrained=True):
        super(KeypointResNet, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_keypoints * 2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.view(x.size(0), -1, 2)
