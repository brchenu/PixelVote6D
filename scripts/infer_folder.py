"""
Offline inference on a folder of real (unlabelled) images.

Default mode  — processes all images and writes an MP4 video:
    python inference.py \
        --images     dataset/realfootage/drill1/frames/ \
        --calib      dataset/realfootage/drill1/calibration/ \
        --checkpoint checkpoints/pvnet_2026-02-28_09-42-05_epoch9_obj1_drill.pth \
        --keypoints  dataset/drill/models/obj_000001_keypoints.txt

Debug mode  — shows each frame interactively and logs estimated pose:
    python inference.py ... --debug

The calibration folder must contain (as saved by np.savetxt / the calibration script):
  - camera_matrix.txt
  - distortion_coefficients.txt
"""

import os
import cv2
import torch
import logging
import argparse
import numpy as np

from PIL import Image
from pathlib import Path
from datetime import datetime

import torchvision.transforms as T

from pixelvote6d.models import PVNet
from pixelvote6d.dataset import PVNetTransform
from pixelvote6d.pose import PVNetRansac, PoseSmoother

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_FPS = 30
AXIS_LENGTH = 60.0  # mm — length of drawn CRS axes
RANSAC_THRESHOLD = 0.5  # mask sigmoid threshold

# Draw-time semantic object frames for the drill. These values are derived from
# the inverse of the Blender mesh transform used to move the desired drill frame
# onto the world origin/orientation.
AXIS_FRAME_ROTATION = np.array(
    [
        [0.99286581, 0.00891773, 0.11890312],
        [0.00891773, 0.98885283, -0.14862890],
        [-0.11890312, 0.14862890, 0.98171865],
    ],
    dtype=np.float32,
)
AXIS_FRAME_TRANSLATION_BASE = np.array(
    [-16.46857737, -24.41427828, 28.85704377],
    dtype=np.float32,
)
AXIS_FRAME_TRANSLATION_TIP = np.array(
    [32.28170131, -85.35212663, 431.36168842],
    dtype=np.float32,
)
AXIS_FRAME_TRANSLATION_MIDDLE = (
    0.5 * (AXIS_FRAME_TRANSLATION_BASE + AXIS_FRAME_TRANSLATION_TIP)
).astype(np.float32)

# Active semantic frame used for drawing.
AXIS_FRAME_TRANSLATION = AXIS_FRAME_TRANSLATION_BASE


# ---------------------------------------------------------------------------
# Calibration / camera helpers
# ---------------------------------------------------------------------------


def load_calibration(calib_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load camera_matrix.txt and distortion_coefficients.txt."""
    K = np.loadtxt(os.path.join(calib_dir, "camera_matrix.txt"))
    dist = np.loadtxt(os.path.join(calib_dir, "distortion_coefficients.txt"))
    return K, dist


def transform_K(
    K: np.ndarray, orig_h: int, orig_w: int, tfm: PVNetTransform
) -> np.ndarray:
    """Adjust camera intrinsics for PVNetTransform (resize -> center-crop)."""
    scale = tfm.resize / min(orig_h, orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    K_new = K.copy()
    K_new[0, :] *= new_w / orig_w
    K_new[1, :] *= new_h / orig_h
    K_new[0, 2] -= (new_w - tfm.crop_size) / 2.0
    K_new[1, 2] -= (new_h - tfm.crop_size) / 2.0
    return K_new


def inverse_transform_keypoints(
    keypoints: np.ndarray, orig_h: int, orig_w: int, tfm: PVNetTransform
) -> np.ndarray:
    """Map crop-space keypoints back to the original undistorted image space."""
    scale = tfm.resize / min(orig_h, orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)

    crop_top = (new_h - tfm.crop_size) / 2.0
    crop_left = (new_w - tfm.crop_size) / 2.0

    keypoints = keypoints + np.array([crop_left, crop_top])
    keypoints = keypoints * np.array([orig_w / new_w, orig_h / new_h])
    return keypoints


def inverse_transform_mask(
    mask: np.ndarray, orig_h: int, orig_w: int, tfm: PVNetTransform
) -> np.ndarray:
    """Map a crop-space mask/probability map back to the original undistorted image space."""
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


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def preprocess_image(
    bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    tfm: PVNetTransform,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Undistort + apply PVNetTransform (resize, center-crop, normalize).

    Returns:
        tensor : (1, 3, H, W) float32, ready for the network
        bgr_undist : undistorted BGR image at the original resolution
    """
    bgr_undist = cv2.undistort(bgr, K, dist)
    rgb = cv2.cvtColor(bgr_undist, cv2.COLOR_BGR2RGB)

    image_transform = T.Compose(
        [
            T.Resize(tfm.resize, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop((tfm.crop_size, tfm.crop_size)),
            T.ToTensor(),
            T.Normalize(mean=PVNetTransform.MEAN, std=PVNetTransform.STD),
        ]
    )
    tensor = image_transform(Image.fromarray(rgb)).unsqueeze(0)
    return tensor, bgr_undist


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_keypoints(
    img: np.ndarray,
    kp_2d: np.ndarray,
    radius: int = 5,
    font_scale: float = 0.45,
    thickness: int = 1,
) -> np.ndarray:
    """Draw green circles at each predicted 2-D keypoint."""
    for idx, (x, y) in enumerate(kp_2d):
        if np.isnan(x) or np.isnan(y):
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


def draw_axes(
    img: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    length: float = AXIS_LENGTH,
) -> np.ndarray:
    """Draw RGB coordinate axes (X=red, Y=green, Z=blue)."""
    axes_local = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length], [0, 0, 0]])
    axes_3d = (axes_local @ AXIS_FRAME_ROTATION.T) + AXIS_FRAME_TRANSLATION
    pts, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, np.zeros(4))
    pts = pts.reshape(-1, 2).astype(int)
    o = tuple(pts[3])
    cv2.line(img, o, tuple(pts[0]), (0, 0, 255), 5)  # X red
    cv2.line(img, o, tuple(pts[1]), (0, 255, 0), 5)  # Y green
    cv2.line(img, o, tuple(pts[2]), (255, 0, 0), 5)  # Z blue
    return img


def build_debug_panel(
    axes_img: np.ndarray,
    keypoints_img: np.ndarray,
    overlay_img: np.ndarray,
    mask_prob_full: np.ndarray,
) -> np.ndarray:
    """
    Build a 2×2 diagnostic panel:
      ┌─────────────────┬───────────────────┐
      │    axes view    │  predicted mask   │
      │   (pose/CRS)    │  (sigmoid heatmap)│
      ├─────────────────┼───────────────────┤
      │  keypoints view │   mask overlay    │
      │  (indexed kp)   │  (transparent)    │
      └─────────────────┴───────────────────┘
    """
    H, W = axes_img.shape[:2]

    cell0 = axes_img.copy()
    cv2.putText(
        cell0, "pose + axes", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3
    )
    cv2.putText(
        cell0, "pose + axes", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
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
    panel = np.concatenate([top, bottom], axis=0)
    return panel


def draw_pose_label(img: np.ndarray, tvec: np.ndarray, rvec: np.ndarray) -> np.ndarray:
    """Overlay translation (mm) and rotation (deg) on the image."""
    t = tvec.ravel()
    r_deg = np.degrees(cv2.Rodrigues(rvec)[0].ravel())
    lines = [
        f"t: x={t[0]:.1f}  y={t[1]:.1f}  z={t[2]:.1f} mm",
        f"r: x={r_deg[0]:.1f}  y={r_deg[1]:.1f}  z={r_deg[2]:.1f} deg",
    ]
    for i, line in enumerate(lines):
        # Black outline for readability
        cv2.putText(
            img, line, (8, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3
        )
        cv2.putText(
            img,
            line,
            (8, 20 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return img


def draw_mask_overlay(
    img: np.ndarray, mask_prob_full: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Overlay the predicted mask probability on top of the real image."""
    mask_u8 = (np.clip(mask_prob_full, 0.0, 1.0) * 255).astype(np.uint8)
    mask_color = cv2.applyColorMap(mask_u8, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(img, 1.0 - alpha, mask_color, alpha, 0.0)

    binary_mask = (mask_prob_full > RANSAC_THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    return overlay


# ---------------------------------------------------------------------------
# Per-frame inference
# ---------------------------------------------------------------------------


def run_inference(
    bgr: np.ndarray,
    pvnet: "PVNet",
    K: np.ndarray,
    dist: np.ndarray,
    keypoints_3d: np.ndarray,
    tfm: PVNetTransform,
    device: torch.device,
    ransac_iter: int,
    smoother: PoseSmoother | None = None,
) -> dict:
    """
    Run the full pipeline on one BGR image.

    Returns a dict:
        result_img   — annotated undistorted BGR image (original resolution)
        success      — bool: solvePnP succeeded
        rvec, tvec   — pose vectors (or None)
        mask_pixels  — number of foreground pixels detected
    """
    tensor, undist_bgr = preprocess_image(bgr, K, dist, tfm)
    tensor = tensor.to(device)

    with torch.no_grad():
        pred_mask, pred_vfield = pvnet(tensor)

    pred_mask = pred_mask.squeeze()
    pred_vfield = pred_vfield.squeeze(0)
    mask_prob_crop = torch.sigmoid(pred_mask).cpu().numpy().astype(np.float32)

    result_img = undist_bgr.copy()
    axes_img = undist_bgr.copy()
    keypoints_img = undist_bgr.copy()
    orig_h, orig_w = result_img.shape[:2]
    mask_prob_full = inverse_transform_mask(mask_prob_crop, orig_h, orig_w, tfm)

    binary_mask = pred_mask.sigmoid() > RANSAC_THRESHOLD
    mask_pixels = int(binary_mask.sum().item())

    if mask_pixels < 10:
        overlay_img = draw_mask_overlay(undist_bgr.copy(), mask_prob_full)
        return dict(
            result_img=result_img,
            success=False,
            rvec=None,
            tvec=None,
            mask_pixels=mask_pixels,
            pred_mask=pred_mask,
            pred_vfield=pred_vfield,
            axes_img=axes_img,
            keypoints_img=keypoints_img,
            overlay_img=overlay_img,
            mask_prob_full=mask_prob_full,
        )

    ransac = PVNetRansac(
        mask=binary_mask.float(), vfield=pred_vfield, num_iter=ransac_iter
    )
    kp_2d_crop = ransac.ransac().cpu().numpy().astype(np.float64)

    kp_2d = inverse_transform_keypoints(kp_2d_crop, orig_h, orig_w, tfm)
    valid_keypoints = np.isfinite(kp_2d).all(axis=1)

    if valid_keypoints.sum() < 4:
        draw_keypoints(result_img, kp_2d)
        draw_keypoints(keypoints_img, kp_2d, radius=8, font_scale=0.7, thickness=2)
        overlay_img = draw_mask_overlay(undist_bgr.copy(), mask_prob_full)
        return dict(
            result_img=result_img,
            success=False,
            rvec=None,
            tvec=None,
            mask_pixels=mask_pixels,
            pred_mask=pred_mask,
            pred_vfield=pred_vfield,
            axes_img=axes_img,
            keypoints_img=keypoints_img,
            overlay_img=overlay_img,
            mask_prob_full=mask_prob_full,
        )

    success, rvec, tvec, _ = cv2.solvePnPRansac(
        keypoints_3d[valid_keypoints].astype(np.float64),
        kp_2d[valid_keypoints].reshape(-1, 1, 2),
        K,
        distCoeffs=np.zeros(4),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    draw_keypoints(result_img, kp_2d)
    draw_keypoints(keypoints_img, kp_2d, radius=8, font_scale=0.7, thickness=2)
    overlay_img = draw_mask_overlay(undist_bgr.copy(), mask_prob_full)

    if success:
        if smoother is not None:
            rvec, tvec = smoother.update(rvec, tvec)
        draw_axes(result_img, rvec, tvec, K)
        draw_axes(axes_img, rvec, tvec, K)
        draw_pose_label(result_img, tvec, rvec)
        draw_pose_label(axes_img, tvec, rvec)

    return dict(
        result_img=result_img,
        success=success,
        rvec=rvec,
        tvec=tvec,
        mask_pixels=mask_pixels,
        pred_mask=pred_mask,
        pred_vfield=pred_vfield,
        axes_img=axes_img,
        keypoints_img=keypoints_img,
        overlay_img=overlay_img,
        mask_prob_full=mask_prob_full,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PVNet inference on a folder of real images."
    )
    p.add_argument("--images", required=True, help="Folder of captured images.")
    p.add_argument(
        "--calib",
        required=True,
        help="Folder with camera_matrix.txt and distortion_coefficients.txt.",
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pth model checkpoint (not the .txt train report!).",
    )
    p.add_argument(
        "--keypoints",
        required=True,
        help="Path to obj_XXXXXX_keypoints.txt (3-D, in mm).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output video path (default: inference_<timestamp>.mp4). Ignored in --debug mode.",
    )
    p.add_argument(
        "--save-debug-panel",
        action="store_true",
        help="Save the same 4-panel visualization used by --debug into the output video.",
    )
    p.add_argument(
        "--ransac-iter", type=int, default=256, help="RANSAC iterations (default: 256)."
    )
    p.add_argument(
        "--smooth-alpha-rvec",
        type=float,
        default=0.5,
        help="Rotation smoothing factor in [0, 1]. Use 1.0 to disable smoothing.",
    )
    p.add_argument(
        "--smooth-alpha-tvec",
        type=float,
        default=0.5,
        help="Translation smoothing factor in [0, 1]. Use 1.0 to disable smoothing.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: show each frame interactively and log estimated pose.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ── Validate checkpoint extension ────────────────────────────────────────
    if not args.checkpoint.endswith(".pth"):
        raise ValueError(
            f"--checkpoint must be a .pth file, got: {args.checkpoint}\n"
            "Did you accidentally pass the .txt train report instead of the .pth checkpoint?"
        )

    # ── Logging ──────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("inference")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Calibration ──────────────────────────────────────────────────────────
    K, dist = load_calibration(args.calib)
    logger.info("Camera matrix:\n%s", K)
    logger.info("Distortion: %s", dist.ravel())

    # ── 3-D keypoints ────────────────────────────────────────────────────────
    keypoints_3d = np.loadtxt(args.keypoints)
    logger.info("Loaded %d keypoints from %s", len(keypoints_3d), args.keypoints)

    # ── Model ────────────────────────────────────────────────────────────────
    pvnet = PVNet()
    # weights_only=False is required because the checkpoint also stores
    # the optimizer state dict (numpy arrays), which torch 2.6 blocks otherwise.
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    pvnet.load_state_dict(ckpt["model_state_dict"])
    pvnet.to(device)
    pvnet.eval()
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    # ── Collect & sort images ─────────────────────────────────────────────────
    img_dir = Path(args.images)
    image_paths = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        logger.error("No images found in %s", img_dir)
        return
    logger.info("Found %d image(s) in %s", len(image_paths), img_dir)

    tfm = PVNetTransform()
    smoother = PoseSmoother(
        alpha_rvec=args.smooth_alpha_rvec,
        alpha_tvec=args.smooth_alpha_tvec,
    )
    logger.info(
        "Pose smoothing  : alpha_rvec=%.2f alpha_tvec=%.2f",
        args.smooth_alpha_rvec,
        args.smooth_alpha_tvec,
    )
    logger.info(
        "Video mode      : %s",
        "4-panel debug" if args.save_debug_panel else "annotated result",
    )

    # ── Video writer (default mode only) ─────────────────────────────────────
    if not args.debug:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_path = args.output or f"inference_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = None
        logger.info("Output video: %s", video_path)
    else:
        video_writer = None

    # ── Frame loop ────────────────────────────────────────────────────────────
    for frame_idx, img_path in enumerate(image_paths):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning("[SKIP] Could not read %s", img_path.name)
            continue

        result = run_inference(
            bgr, pvnet, K, dist, keypoints_3d, tfm, device, args.ransac_iter, smoother
        )

        result_img = result["result_img"]
        success = result["success"]
        mask_pixels = result["mask_pixels"]

        if not success:
            reason = "object not detected" if mask_pixels < 10 else "solvePnP failed"
            logger.warning(
                "Frame %03d (%s): %s (mask_px=%d)",
                frame_idx,
                img_path.name,
                reason,
                mask_pixels,
            )
        elif args.debug:
            tvec = result["tvec"].ravel()
            r_deg = np.degrees(cv2.Rodrigues(result["rvec"])[0].ravel())
            logger.debug(
                "Frame %03d | %-30s | t=[%6.1f, %6.1f, %6.1f] mm | "
                "r=[%6.1f, %6.1f, %6.1f] deg | mask_px=%d",
                frame_idx,
                img_path.name,
                tvec[0],
                tvec[1],
                tvec[2],
                r_deg[0],
                r_deg[1],
                r_deg[2],
                mask_pixels,
            )

        # Default mode — write frame to video
        if not args.debug:
            frame_out = result_img
            if args.save_debug_panel:
                frame_out = build_debug_panel(
                    result["axes_img"],
                    result["keypoints_img"],
                    result["overlay_img"],
                    result["mask_prob_full"],
                )
            if video_writer is None:
                frame_size = (frame_out.shape[1], frame_out.shape[0])
                video_writer = cv2.VideoWriter(
                    video_path, fourcc, VIDEO_FPS, frame_size
                )
                logger.info("Video frame size: %dx%d", frame_size[0], frame_size[1])
            video_writer.write(frame_out)
            if (frame_idx + 1) % 50 == 0 or (frame_idx + 1) == len(image_paths):
                logger.info("  Processed %d / %d", frame_idx + 1, len(image_paths))

        # Debug mode — interactive display, frame by frame
        else:
            status = "OK" if success else "NO DETECTION"
            panel = build_debug_panel(
                result["axes_img"],
                result["keypoints_img"],
                result["overlay_img"],
                result["mask_prob_full"],
            )
            win_title = (
                f"[DEBUG {status}] {img_path.name}  |  any key: next  |  q: quit"
            )
            cv2.imshow(win_title, panel)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                logger.info("Interrupted by user at frame %d.", frame_idx)
                break

    # ── Finalise ─────────────────────────────────────────────────────────────
    if video_writer is not None:
        video_writer.release()
        logger.info("Video saved to: %s", video_path)
        logger.info(
            "Tip — convert to GIF:  ffmpeg -i %s -vf 'fps=%d,scale=960:-1:flags=lanczos' -loop 0 inference.gif",
            video_path,
            VIDEO_FPS,
        )

    cv2.destroyAllWindows()
    logger.info("Done.")


if __name__ == "__main__":
    main()
