import torch
import torch.nn as nn
from torchvision import models


class segment(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ResNetUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Pretrained ResNet50 encoder
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        self.encoder = resnet

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            2048, 1024, 2, stride=2
        )

        self.dec4 = segment(
            2048, 1024
        )

        self.up3 = nn.ConvTranspose2d(
            1024, 512, 2, stride=2
        )

        self.dec3 = segment(
            1024, 512
        )

        self.up2 = nn.ConvTranspose2d(
            512, 256, 2, stride=2
        )

        self.dec2 = segment(
            512, 256
        )

        self.up1 = nn.ConvTranspose2d(
            256, 64, 2, stride=2
        )

        self.dec1 = segment(
            128, 64
        )

        self.up0 = nn.ConvTranspose2d(
            64, 64, 2, stride=2
        )

        self.output = nn.Conv2d(
            64, 1, kernel_size=1
        )

    def forward(self, x):

        # Encoder
        x1 = self.encoder.conv1(x)
        x1 = self.encoder.bn1(x1)
        x1 = self.encoder.relu(x1)

        x = self.encoder.maxpool(x1)

        x2 = self.encoder.layer1(x)
        x3 = self.encoder.layer2(x2)
        x4 = self.encoder.layer3(x3)
        x5 = self.encoder.layer4(x4)

        # Decoder
        x = self.up4(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        x = self.up0(x)
        x = self.output(x)

        return x