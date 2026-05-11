import torch
import numpy as np
import torchvision.transforms.v2 as v2
import torchvision.transforms as T
from torchvision import tv_tensors
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PVNetRandomTranform:
    """PVNET dataset random transform"""

    MEAN = IMAGENET_MEAN
    STD = IMAGENET_STD

    def __init__(self, resize: int = 256, crop_size: int = 224, spatial_aug: bool = False):
        self.resize = resize
        self.crop_size = crop_size

        spatial_transforms = []
        if spatial_aug:
            # Simulate the object appearing off-center and at varying scales,
            # bridging the gap from Blender's always-centered renders to real footage.
            # degrees=10 covers plausible camera roll; translate=0.15 shifts up to ~34px
            # on a 256px image; scale covers ±20%. Applied before CenterCrop.
            spatial_transforms = [
                v2.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.8, 1.2)),
            ]

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(resize),
                *spatial_transforms,
                v2.CenterCrop(crop_size),
                v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                v2.RandomApply([v2.GaussianBlur(5)], p=0.3),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __call__(self, image, mask, keypoints):
        orig_h, orig_w = image.shape[0], image.shape[1]

        # Do NOT pre-wrap image as tv_tensors.Image — v2.ToImage() handles HWC→CHW.
        # Pre-wrapping with tv_tensors.Image on a HWC numpy array leaves the shape
        # as (H, W, C) since tv_tensors.Image doesn't permute, causing ColorJitter
        # to see H as the channel count and crash.
        mask = tv_tensors.Mask(mask[None])  # (H, W) -> (1, H, W)
        keypoints = tv_tensors.KeyPoints(
            torch.from_numpy(keypoints).float(), canvas_size=(orig_h, orig_w)
        )

        image, mask, keypoints = self.transform(image, mask, keypoints)

        # ToDtype(scale=True) skips Masks, so values stay 0/255 — normalize to [0,1]
        mask = (mask > 0).float()

        # Convert KeyPoints tensor back to numpy array for generate_vector_field
        return image, mask, keypoints.numpy()


class PVNetTransform:
    """PVNet dataset transform using torchvision v1 transforms."""

    MEAN = IMAGENET_MEAN
    STD = IMAGENET_STD

    def __init__(self, resize: int = 256, crop_size: int = 224):
        self.resize = resize
        self.crop_size = crop_size

        self.image_transform = T.Compose(
            [
                T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop((crop_size, crop_size)),
                T.ToTensor(),  # Converts PIL to float32 [0, 1]
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.mask_transform = T.Compose(
            [
                T.Resize(resize, interpolation=T.InterpolationMode.NEAREST),
                T.CenterCrop((crop_size, crop_size)),
                T.ToTensor(),  # Converts PIL uint8 [0,255] to float32 [0,1]
            ]
        )

    @staticmethod
    def unnormalize_image(tensor: torch.Tensor) -> torch.Tensor:
        """Unnormalize an image tensor for visualization.

        Args:
            tensor: (3, H, W) normalized image tensor
        """
        device = tensor.device

        mean = torch.tensor(IMAGENET_MEAN).reshape(3, 1, 1).to(device)
        std = torch.tensor(IMAGENET_STD).reshape(3, 1, 1).to(device)

        image = (tensor * std) + mean
        image = image.clamp(0.0, 1.0)
        image = (image * 255.0).to(torch.uint8)
        return image

    def _transform_keypoints(
        self, keypoints: np.ndarray, orig_h: int, orig_w: int
    ) -> np.ndarray:
        """Apply the same resize + center-crop to keypoint coordinates.

        Args:
            keypoints: (K, 2) array of (x, y) coordinates
            orig_h: original image height
            orig_w: original image width

        Returns:
            (K, 2) array of transformed (x, y) coordinates
        """
        scale = self.resize / min(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        keypoints = keypoints * np.array([new_w / orig_w, new_h / orig_h])

        # Shift keypoints origin to account for center crop
        crop_top = (new_h - self.crop_size) / 2.0
        crop_left = (new_w - self.crop_size) / 2.0
        keypoints = keypoints - np.array([crop_left, crop_top])

        return keypoints

    def __call__(self, image, mask, keypoints):
        orig_h, orig_w = image.shape[0], image.shape[1]

        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)

        image = self.image_transform(image_pil)
        mask = self.mask_transform(mask_pil)

        keypoints = self._transform_keypoints(keypoints, orig_h, orig_w)

        return image, mask, keypoints


class PVNetTransformV2:
    """Transform for PVNet dataset samples, using torchvision v2 transforms."""

    MEAN = IMAGENET_MEAN
    STD = IMAGENET_STD

    def __init__(self, resize: int = 256, crop_size: int = 224):
        self.resize = resize
        self.crop_size = crop_size

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(resize),
                v2.CenterCrop((crop_size, crop_size)),
                # If dtype is a torch.dtype, ToDtype work only for Image and Video (not for Mask)
                v2.ToDtype(
                    torch.float32, scale=True
                ),  # Not sure if this is also Called for tv_tensor.Mask ?
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _transform_keypoints(
        self, keypoints: np.ndarray, orig_h: int, orig_w: int
    ) -> np.ndarray:
        """Transform keypoints according to the same transformations applied to the image.
           i.e: resize and center crop.

        Args:
            keypoints: (K, 2) array of (x, y) coordinates
            orig_h: original image height
            orig_w: original image width

        Returns:
            (K, 2) array of transformed (x, y) coordinates
        """
        scale = self.resize / min(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        keypoints = keypoints * np.array([new_w / orig_w, new_h / orig_h])

        crop_top = (new_h - self.crop_size) / 2.0
        crop_left = (new_w - self.crop_size) / 2.0
        keypoints = keypoints - np.array([crop_left, crop_top])

        return keypoints
    
    def inverse_transform_keypoints(self, keypoints: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        """Transform back keypoints from transformed image space to original image space"""
        scale = self.resize / min(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        crop_top = (new_h - self.crop_size) / 2.0
        crop_left = (new_w - self.crop_size) / 2.0
        
        # Invert transform: crop then scale
        keypoints = keypoints + np.array([crop_left, crop_top])        
        keypoints = keypoints * np.array([orig_w/ new_w, orig_h/ new_h])

        return keypoints


    def __call__(self, image, mask, keypoints):
        orig_h, orig_w = image.shape[0], image.shape[1]

        # Do NOT pre-wrap image — let v2.ToImage() handle HWC→CHW conversion.
        mask = tv_tensors.Mask(mask)

        image, mask = self.transform(image, mask)

        keypoints = self._transform_keypoints(keypoints, orig_h, orig_w)

        return image, mask, keypoints
