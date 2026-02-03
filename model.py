import os
import torch
from torch import nn
from PIL import Image
import torchvision.models as models
from torchvision import transforms as T
from pathlib import Path
from dataset import BOPDataset

BASE_DIR = Path(__file__).resolve().parent

# random_data_path = os.path.join(BASE_DIR, "dataset", "custom", "random")

resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1, progress=True)
print(resnet18)
# transforms = models.ResNet18_Weights.IMAGENET1K_V1.transforms()

# teapot = Image.open(os.path.join(random_data_path, "teapot.webp"))

# tensor_teapot = transforms(teapot).unsqueeze(0)

# resnet18.eval()

# output = resnet18(tensor_teapot)

# print(type(output))
# print(output.shape)
# print(f"max idx: {torch.argmax(output, dim=1)}")


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels, out_channels, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        x = self.layers(x)
        out = self.upsample(x)
        return out


class PVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1, progress=True
        )

        self.conv1 = self.backbone.conv1
        self.layer0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
        )
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4  # bottleneck features

        self.decoder1 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder3 = DecoderBlock(128, 64, 64)
        self.decoder4 = DecoderBlock(64, 64, 64)

        self.head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )

    def forward(self, x):
        x0 = self.conv1(x)    # 1/2 resolution
        x1 = self.layer1(x0)  # 1/4 resolution
        x2 = self.layer2(x1)  # 1/8 resolution
        x3 = self.layer3(x2)  # 1/16 resolution

        # Bottleneck features
        x4 = self.layer4(x3)  # 1/32 resolution

        # Skip connections from encoder layer3 works here
        # because we upsample the bottleneck features
        x5 = self.decoder1(x=x4, skip=x3)
        x6 = self.decoder2(x=x5, skip=x2)
        x7 = self.decoder3(x=x6, skip=x1)
        x8 = self.decoder4(x=x7, skip=x0)

        return self.head(x8)


transforms = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
print(transforms)

dataset_path = os.path.join(BASE_DIR, "dataset", "clean", "scene_000010")
dataset = BOPDataset(dataset_path=dataset_path)

cam_params, image, mask, keypoints, vector_field = dataset[0]

print(f"image shape: {image.size} ")
tensor_image = transforms(image).unsqueeze(0)
print(f"tensor image shape: {tensor_image.shape} ")

pvnet = PVNet()
pvnet.eval()

pvnet(tensor_image)
