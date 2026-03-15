import os
import torch
import cv2
import numpy as np
import argparse
from model import PVNet
from pathlib import Path
from bop_toolkit.data_transfrom import PVNetTransform
from bop_toolkit.bop_dataset import BOPDirectDataset, BOPSubSet
from bop_toolkit.visualization import show_vfield

BASE_DIR = Path(__file__).resolve().parent


def show_keypoints(image: torch.Tensor, keypoints_2d: torch.Tensor) -> None:
    """Draw predicted keypoints on the image and display it."""
    img = PVNetTransform.unnormalize_image(image.squeeze())
    img_bgr = cv2.cvtColor(img.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)

    for (x, y) in keypoints_2d.squeeze().cpu().numpy():
        cv2.circle(img_bgr, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

    resized = cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Keypoints", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_mask(image: torch.Tensor, pred_mask: torch.Tensor) -> None:
    """Apply the predicted binary mask to the image and display it."""
    img = PVNetTransform.unnormalize_image(image.squeeze())
    img_bgr = cv2.cvtColor(img.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)

    binary_mask = (torch.sigmoid(pred_mask[0]) > 0.5).float()
    mask_np = binary_mask.squeeze().cpu().numpy().astype(np.uint8)
    masked_img = img_bgr * np.repeat(mask_np[:, :, None], 3, axis=2)

    resized = cv2.resize(masked_img, (640, 480), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Mask", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test PVNet model")
    parser.add_argument("-id", "--obj-id", type=int, required=True, help="Object ID to test on")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset name (e.g., 'drill')")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["vfield", "keypoints", "mask"],
        default="vfield",
        help="Visualization mode: 'vfield' (vector field per keypoint), "
             "'keypoints' (keypoints on image), 'mask' (predicted segmentation mask). "
             "Default: vfield",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pvnet = PVNet()
    pvnet.load_state_dict(torch.load(args.checkpoint, weights_only=True)["model_state_dict"])
    pvnet.to(device)
    pvnet.eval()

    dataset = BOPDirectDataset(
        dataset_dir=os.path.join(BASE_DIR, "dataset", args.dataset),
        obj_id=args.obj_id,
        transform=PVNetTransform(),
        subset=BOPSubSet.TEST,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for image, mask, vfield, keypoints_2d in dataloader:
        image = image.to(device)
        pred_mask, pred_vfield = pvnet(image)

        if args.mode == "vfield":
            show_vfield(image, pred_mask, pred_vfield, keypoints_2d)
        elif args.mode == "keypoints":
            show_keypoints(image, keypoints_2d)
        elif args.mode == "mask":
            show_mask(image, pred_mask)
