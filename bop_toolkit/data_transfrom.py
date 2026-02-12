import torch
import numpy as np
import torchvision.transforms.v2 as v2
import torchvision.transforms as T
from torchvision import tv_tensors
from PIL import Image


class PVNetTransform:
    """PVNet dataset transform using torchvision v1 transforms."""
    
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, resize: int = 256, crop_size: int = 224):
        self.resize = resize
        self.crop_size = crop_size

        self.image_transform = T.Compose(
            [
                T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop((crop_size, crop_size)),
                T.ToTensor(),  # Converts PIL to float32 [0, 1]
                T.Normalize(mean=PVNetTransform.MEAN, std=PVNetTransform.STD),
            ]
        )

        self.mask_transform = T.Compose(
            [
                T.Resize(resize, interpolation=T.InterpolationMode.NEAREST),
                T.CenterCrop((crop_size, crop_size)),
                T.ToTensor(),  # Converts PIL to float32 [0, 1]
            ]
        )

    @staticmethod
    def unnormalize_image(tensor: torch.Tensor) -> torch.Tensor:
        """Unnormalize an image tensor for visualization.

        Args:
            tensor: (3, H, W) normalized image tensor
        """
        device = tensor.device

        mean = torch.tensor(PVNetTransform.MEAN).reshape(3, 1, 1).to(device)
        std = torch.tensor(PVNetTransform.STD).reshape(3, 1, 1).to(device)

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

    def __init__(self, resize: int = 256, crop_size: int = 224):
        self.resize = resize
        self.crop_size = crop_size

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(resize),
                v2.CenterCrop((crop_size, crop_size)),
                # If dtype is a torch.dtype, ToDtype work only for Image and Video (not for Mask)
                v2.ToDtype(torch.float32, scale=True),  # Not sure if this is also Called for tv_tensor.Mask ?
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    def __call__(self, image, mask, keypoints):
        orig_h, orig_w = image.shape[0], image.shape[1]

        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        image, mask = self.transform(image, mask)

        keypoints = self._transform_keypoints(keypoints, orig_h, orig_w)

        return image, mask, keypoints
