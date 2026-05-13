import cv2
import torch
import numpy as np

from pixelvote6d.models import PVNet
from pixelvote6d.pose.ransac import PVNetRansac
from pixelvote6d.dataset.transforms import PVNetTransformV2
from pixelvote6d.pose.smoothing import PoseSmoother

RANSAC_THRESHOLD = 0.5
MIN_MASK_PIXELS = 600
AXIS_LENGTH_MM = 60.0

# Semantic frame offset used to draw the coordinate system at the drill bottom.
R_offset = np.array(
    [
        [0.99286581, 0.00891773, 0.11890312],
        [0.00891773, 0.98885283, -0.14862890],
        [-0.11890312, 0.14862890, 0.98171865],
    ],
    dtype=np.float32,
)
t_offset = np.array(
    [-16.46857737, -24.41427828, 28.85704377],
    dtype=np.float32,
)


def is_point_in_image(x, y, img_width, img_height):
    return 0 <= x < img_width and 0 <= y < img_height


def inverse_transform_mask(
    mask: np.ndarray, orig_h: int, orig_w: int, tfm: PVNetTransformV2
) -> np.ndarray:
    scale = tfm.resize / min(orig_h, orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)

    crop_top = int((new_h - tfm.crop_size) / 2.0)
    crop_left = int((new_w - tfm.crop_size) / 2.0)

    canvas = np.zeros((new_h, new_w), dtype=np.float32)
    canvas[
        crop_top : crop_top + tfm.crop_size, crop_left : crop_left + tfm.crop_size
    ] = mask.astype(np.float32)
    return cv2.resize(canvas, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def draw_keypoints(img, keypoints, radius=4, font_scale=0.5, thickness=1):
    for idx, (x, y) in enumerate(keypoints):
        if not np.isfinite([x, y]).all():
            continue
        cv2.circle(img, (int(x), int(y)), radius, (0, 255, 0), -1)
        cv2.putText(
            img,
            str(idx),
            (int(x) + radius + 2, int(y) - radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
    return img


def draw_axes(img, rvec, tvec, camera_matrix, axis_length=AXIS_LENGTH_MM):
    axes_local = np.float32(
        [[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length], [0, 0, 0]]
    )
    axes_3d = (axes_local @ R_offset.T) + t_offset
    points_2d, _ = cv2.projectPoints(
        axes_3d, rvec, tvec, camera_matrix, np.zeros(4, dtype=np.float64)
    )
    points_2d = points_2d.reshape(-1, 2).astype(int)

    origin = tuple(points_2d[3])
    cv2.line(img, origin, tuple(points_2d[0]), (0, 0, 255), 4)
    cv2.line(img, origin, tuple(points_2d[1]), (0, 255, 0), 4)
    cv2.line(img, origin, tuple(points_2d[2]), (255, 0, 0), 4)
    return img


def draw_mask_overlay(img, mask_prob_full, alpha=0.45):
    mask_u8 = (np.clip(mask_prob_full, 0.0, 1.0) * 255).astype(np.uint8)
    mask_color = cv2.applyColorMap(mask_u8, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(img, 1.0 - alpha, mask_color, alpha, 0.0)

    binary_mask = (mask_prob_full > RANSAC_THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    return overlay


def build_debug_panel(axes_img, keypoints_img, overlay_img, mask_prob_full):
    cell0 = axes_img.copy()
    cv2.putText(cell0, "pose", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(
        cell0, "pose", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )

    mask_u8 = (np.clip(mask_prob_full, 0.0, 1.0) * 255).astype(np.uint8)
    cell1 = cv2.applyColorMap(mask_u8, cv2.COLORMAP_INFERNO)
    cv2.putText(
        cell1, "pred mask", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )

    cell2 = keypoints_img.copy()
    cv2.putText(
        cell2, "pred keypoints", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3
    )
    cv2.putText(
        cell2,
        "pred keypoints",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    cell3 = overlay_img.copy()
    cv2.putText(
        cell3, "mask overlay", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3
    )
    cv2.putText(
        cell3,
        "mask overlay",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    top = np.concatenate([cell0, cell1], axis=1)
    bottom = np.concatenate([cell2, cell3], axis=1)
    return np.concatenate([top, bottom], axis=0)


CHECKPOINT_PATH = "checkpoints/2026-04-02_14-56-01_obj1_drill_hd+drill_cut+sl_drill2+sl_real/checkpoint.pth"
CAMERA_CALIB_PATH = "dataset/realfootage/drill2/calibration/"
OBJ_3D_KEYPOINTS_PATH = "dataset/drill_hd/models/obj_000001_keypoints.txt"

camera_matrix = np.loadtxt(CAMERA_CALIB_PATH + "camera_matrix.txt")
distortion_coeffs = np.loadtxt(CAMERA_CALIB_PATH + "distortion_coefficients.txt")
keypoints_3d = np.loadtxt(OBJ_3D_KEYPOINTS_PATH).astype(np.float64)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("cannot open the camera")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PVNet model and checkpoint
pvnet = PVNet()
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
pvnet.load_state_dict(checkpoint["model_state_dict"], strict=False)
pvnet.eval()
pvnet.to(device)

# Init image transform
pvnet_transform = PVNetTransformV2()

smoother = PoseSmoother(alpha_rvec=0.8, alpha_tvec=0.8)

while True:
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    display_frame = cv2.undistort(frame, camera_matrix, distortion_coeffs)
    axes_frame = display_frame.copy()
    keypoints_frame = display_frame.copy()

    img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img = pvnet_transform.transform(img)

    with torch.no_grad():
        mask, vfield = pvnet.forward(img.unsqueeze(0).to(device))

    mask_proba = torch.sigmoid(mask.squeeze(0).squeeze(0))
    mask_binary = (mask_proba > RANSAC_THRESHOLD).float()
    mask_prob_full = inverse_transform_mask(
        mask_proba.cpu().numpy(),
        display_frame.shape[0],
        display_frame.shape[1],
        pvnet_transform,
    )

    overlay_frame = draw_mask_overlay(display_frame.copy(), mask_prob_full)
    orig_keypoints = np.empty((0, 2), dtype=np.float64)

    if int(mask_binary.sum().item()) >= MIN_MASK_PIXELS:
        keypoints = PVNetRansac(mask_binary, vfield.squeeze(0), num_iter=512).ransac()
        orig_keypoints = pvnet_transform.inverse_transform_keypoints(
            keypoints.cpu().numpy(), display_frame.shape[0], display_frame.shape[1]
        )
        draw_keypoints(keypoints_frame, orig_keypoints)

        valid_keypoints = np.isfinite(orig_keypoints).all(axis=1)
        if valid_keypoints.sum() >= 4:
            success, rvec, tvec, _ = cv2.solvePnPRansac(
                keypoints_3d[valid_keypoints],
                orig_keypoints[valid_keypoints].astype(np.float64).reshape(-1, 1, 2),
                camera_matrix,
                distCoeffs=np.zeros(4, dtype=np.float64),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if success:
                rvec, tvec = smoother.update(rvec, tvec)
                draw_axes(axes_frame, rvec, tvec, camera_matrix)

    panel = build_debug_panel(
        axes_frame, keypoints_frame, overlay_frame, mask_prob_full
    )

    cv2.imshow("frame", panel)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
