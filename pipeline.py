import cv2
import os
import torch
import numpy as np
from model import PVNet
from pathlib import Path
from bop_toolkit.bop_dataset import BOPDirectDataset, BOPSubSet
from bop_toolkit.data_transfrom import PVNetTransform
from ransac import PVNetRansac

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

OBJ_ID = 15
BASE_DIR = Path(__file__).resolve().parent
dataset_path = os.path.join(BASE_DIR, "dataset", "ycbv")

# load 3d keypoints
keypoints_3d = np.loadtxt(
    os.path.join(dataset_path, "models", f"obj_{str(OBJ_ID).zfill(6)}_keypoints.txt")
)
print(f"3D keypoints shape: {keypoints_3d.shape}")

pvnet = PVNet()
pvnet.load_state_dict(
    torch.load(
        "checkpoints/pvnet_2026-02-15_10-08-26_epoch2_obj15_ycbv.pth",
        weights_only=True,
    )["model_state_dict"]
)
pvnet.to(device)
pvnet.eval()

dataset = BOPDirectDataset(
    dataset_dir=dataset_path,
    obj_id=OBJ_ID,
    transform=PVNetTransform(),
    subset=BOPSubSet.REAL,
)

datasetloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Build a scaled/cropped K that matches the transformed image coordinates.
# The transform does: Resize(256) then CenterCrop(224).
# We need K in the same coordinate frame as the predicted 2D keypoints.
transform = dataset.transform
sample_0 = dataset.samples[0]
orig_K = sample_0["K"]

# Read the original image size to compute the scale
sample_scene_dir = os.path.join(dataset.data_dir, sample_0["scene"])
sample_img = cv2.imread(
    os.path.join(sample_scene_dir, "rgb", f"{int(sample_0['frame']):06d}.png")
)
orig_h, orig_w = sample_img.shape[:2]

AXIS_LENGTH = 30  # pixels for drawn axes


def transform_K(K: np.ndarray, orig_h: int, orig_w: int, tfm: PVNetTransform) -> np.ndarray:
    """Adjust camera intrinsics to match resize + center-crop transform."""
    scale = tfm.resize / min(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)

    K_new = K.copy()
    K_new[0, :] *= new_w / orig_w  # scale fx, cx
    K_new[1, :] *= new_h / orig_h  # scale fy, cy

    crop_left = (new_w - tfm.crop_size) / 2.0
    crop_top = (new_h - tfm.crop_size) / 2.0
    K_new[0, 2] -= crop_left
    K_new[1, 2] -= crop_top

    return K_new


def draw_axes(
    img: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    length: float = 30.0,
) -> np.ndarray:
    """Draw RGB coordinate axes on the image."""
    dist_coeffs = np.zeros(4)
    axes_3d = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length]])
    origin = np.float32([[0, 0, 0]])

    origin_2d, _ = cv2.projectPoints(origin, rvec, tvec, K, dist_coeffs)
    axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist_coeffs)

    o = tuple(origin_2d[0].ravel().astype(int))
    x = tuple(axes_2d[0].ravel().astype(int))
    y = tuple(axes_2d[1].ravel().astype(int))
    z = tuple(axes_2d[2].ravel().astype(int))

    cv2.line(img, o, x, (0, 0, 255), 2)  # X = red
    cv2.line(img, o, y, (0, 255, 0), 2)  # Y = green
    cv2.line(img, o, z, (255, 0, 0), 2)  # Z = blue

    return img


for image, mask, vfield, keypoints_2d in datasetloader:
    image = image.to(device)

    with torch.no_grad():
        pred_mask, pred_vfield = pvnet(image)

    pred_mask = pred_mask.squeeze()
    pred_vfield = pred_vfield.squeeze(0)

    pvnet_ransac = PVNetRansac(mask=pred_mask, vfield=pred_vfield, num_iter=256)
    final_keypoints = pvnet_ransac.ransac()

    img = PVNetTransform.unnormalize_image(image.squeeze())
    img = img.permute(1, 2, 0).cpu().numpy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Draw predicted keypoints
    kp_2d = final_keypoints.cpu().numpy().astype(np.float64)  # (K, 2)
    for kp in kp_2d:
        x, y = int(kp[0]), int(kp[1])
        if 0 <= x < img_bgr.shape[1] and 0 <= y < img_bgr.shape[0]:
            cv2.circle(img_bgr, (x, y), 5, (0, 255, 0), -1)

    # Get sample-specific K and transform it
    sample = dataset.samples[0]  # NOTE: for proper per-frame K, track idx
    K_crop = transform_K(sample["K"], orig_h, orig_w, transform)

    pts_3d = keypoints_3d.astype(np.float64)
    pts_2d = kp_2d.reshape(-1, 1, 2)

    success, rvec, tvec = cv2.solvePnP(
        pts_3d, pts_2d, K_crop, distCoeffs=np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        img_bgr = draw_axes(img_bgr, rvec, tvec, K_crop, length=AXIS_LENGTH)
    else:
        print("solvePnP failed for this frame")

    img_bgr = cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("PVNet Pose", img_bgr)
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()