import os
import torch
from torch import nn
from PIL import Image
import torchvision.models as models
from pathlib import Path
from dataset import BOPDataset
import torchinfo

BASE_DIR = Path(__file__).resolve().parent

# random_data_path = os.path.join(BASE_DIR, "dataset", "custom", "random")

# resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1, progress=True)
# torchinfo.summary(resnet18, input_size=(1, 3, 224, 224))

# print(resnet18)

# img = visualtorch.layered_view(resnet18, input_shape=(1, 3, 224, 224), legend=True, draw_volume=False)
# print(type(img))

# img.save(os.path.join(BASE_DIR, "resnet18_architecture_flat.png"))

# torch.save(resnet18.state_dict(), os.path.join(BASE_DIR, "resnet18_weights.pth"))
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
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
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

        self.mask_channels = 1
        self.vfield_channels = 16

        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2

        # Set dilation for layer3 and layer4

        print(self.backbone.layer3)

        for module in self.backbone.layer3.modules():
            if isinstance(module, nn.Conv2d):
                if module.kernel_size == (1, 1):
                    continue  # Used to skip downsample

                module.dilation = (2, 2)
                # Paddig set to 2 because:
                # padding = dilation * (kernel_size - 1) // 2
                module.padding = (2, 2)
                module.stride = (1, 1)

        # Set to 1 the stride of the first BasicBlock in layer3 to prevent downsampling
        # We want to prevent downsampling because originally the model was shrinking the
        # feature maps but since with set dilation to 2 and the stride to 1, we no longer
        # need the downsampling for the residual connection to work.
        self.backbone.layer3[0].downsample[0].stride = (1, 1)
        self.layer3 = self.backbone.layer3

        for module in self.backbone.layer4.modules():
            if isinstance(module, nn.Conv2d):
                if module.kernel_size == (1, 1):
                    continue  # Used to skip downsample

                module.stride = (1, 1)
                module.dilation = (4, 4)
                module.padding = (4, 4)

        # Same has layer3
        self.backbone.layer4[0].downsample[0].stride = (1, 1)
        self.layer4 = self.backbone.layer4

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = DecoderBlock(256, skip_channels=128, out_channels=128)
        self.decoder2 = DecoderBlock(128, skip_channels=64, out_channels=64)
        self.decoder3 = DecoderBlock(64, skip_channels=64, out_channels=32)
        self.head = nn.Sequential(
            nn.Conv2d(3 + 32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32,
                self.mask_channels + self.vfield_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))  # (64, H/2, W/2), skip for decoder3
        x0_pool = self.maxpool(x0)  # (64, H/4, W/4)

        # Encoder (ResNet blocks)
        x1 = self.layer1(x0_pool)  # (64,  H/4, W/4), skip for decoder2
        x2 = self.layer2(x1)  # (128, H/8, W/8), skip for decoder1
        x3 = self.layer3(x2)  # (256, H/8, W/8)
        x4 = self.layer4(x3)  # (512, H/8, W/8)

        x5 = self.bottleneck(x4)  # (256, H/8, W/8)

        # Decoder with skip connections
        d1 = self.decoder1(x=x5, skip=x2)  # (128, H/4, W/4)
        d2 = self.decoder2(x=d1, skip=x1)  # (64, H/2, W/2)
        d3 = self.decoder3(x=d2, skip=x0)  # (32, H, W)

        d4 = torch.cat([d3, x], dim=1)
        out = self.head(d4)  # (18, H, W)

        mask = out[:, : self.mask_channels, :, :]
        vfield = out[:, self.mask_channels :, :, :]

        return mask, vfield


# transforms = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
# print(transforms)

# dataset_path = os.path.join(BASE_DIR, "dataset", "clean", "scene_000010")
# dataset = BOPDataset(dataset_path=dataset_path)

# cam_params, image, mask, keypoints, vector_field = dataset[0]

# print(f"image shape: {image.size} ")
# tensor_image = transforms(image).unsqueeze(0)
# print(f"tensor image shape: {tensor_image.shape} ")

pvnet = PVNet()
torchinfo.summary(pvnet, input_size=(1, 3, 224, 224))
# pvnet.eval()

# pvnet(tensor_image)
