import cv2
import yaml
import torch
import argparse
import numpy as np

from pathlib import Path
from pixelvote6d.pose import PVNetRansac
from pixelvote6d.models import PVNet
from pixelvote6d.dataset import PVNetTransformV2

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

parser = argparse.ArgumentParser(description="Run inference with a trained PVNet model")

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to the folder containing images to run inference on",
)
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to a YAML config file with inference parameters",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Show each image with keypoints and coordinate axes drawn",
)

args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

device = config["model"]["device"]

pvnet = PVNet()
pvnet.eval()
pvnet.to(device)

checkpoint = torch.load(config["model"]["checkpoint"], map_location=device)
pvnet.load_state_dict(checkpoint["model_state_dict"])

# Load camera calibration
calib_dir = Path(config["paths"]["calibration_dir"])
camera_matrix = np.loadtxt(calib_dir / "camera_matrix.txt").astype(np.float32)
dist_coeffs = np.loadtxt(calib_dir / "distortion_coefficients.txt").astype(np.float32)

# Load 3D model keypoints
keypoints_3d = np.loadtxt(config["paths"]["model_keypoints"]).astype(np.float32)

pvnet_transform = PVNetTransformV2()

mask_thresh = config["pipeline"]["mask_thresh"]
ransac_iter = config["pipeline"]["ransac_iter"]

# Aggregate image paths from the input directory
image_paths = sorted(
    p for p in Path(args.input).iterdir()
    if p.suffix.lower() in SUPPORTED_FORMATS
)


def draw_keypoints(image, keypoints):
    for kp in keypoints:
        cv2.circle(image, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)


def draw_axes(image, rvec, tvec):
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.05)


for img_path in image_paths:
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Warning: could not read {img_path}, skipping")
        continue

    orig_h, orig_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_tensor = pvnet_transform.transform(image_rgb).to(device).unsqueeze(0)

    with torch.no_grad():
        mask_pred, vfield_pred = pvnet.forward(image_tensor)
        mask_bin = (torch.sigmoid(mask_pred.squeeze(0).squeeze(0)) > mask_thresh).float()

        if mask_bin.sum() < config["pipeline"]["mask_min_pixels"]:
            print(f"Warning: not enough mask pixels in {img_path}, skipping")
            continue

        keypoints_2d = PVNetRansac(mask_bin, vfield_pred.squeeze(0), ransac_iter).ransac()

    keypoints_2d = keypoints_2d.cpu().numpy()
    keypoints_2d = pvnet_transform.inverse_transform_keypoints(keypoints_2d, orig_h, orig_w)

    _, rvec, tvec = cv2.solvePnP(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs)

    print(f"{img_path.name}: rvec={rvec.ravel()}, tvec={tvec.ravel()}")

    if args.debug:
        debug_image = image.copy()
        draw_keypoints(debug_image, keypoints_2d)
        draw_axes(debug_image, rvec, tvec)
        cv2.imshow("Debug", debug_image)
        cv2.waitKey(0)

cv2.destroyAllWindows()

        



    
    

