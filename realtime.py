import cv2
import torch
import numpy as np
from model import PVNet
from ransac import PVNetRansac
from bop_toolkit.data_transfrom import PVNetTransformV2

RANSAC_THRESHOLD = 0.5
MIN_MASK_PIXELS = 600
AXIS_LENGTH_MM = 30.0


def is_point_in_image(x, y, img_width, img_height):
    return 0 <= x < img_width and 0 <= y < img_height


def draw_axes(img, rvec, tvec, camera_matrix, axis_length=AXIS_LENGTH_MM):
    axes_3d = np.float32(
        [[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length], [0, 0, 0]]
    )
    points_2d, _ = cv2.projectPoints(
        axes_3d, rvec, tvec, camera_matrix, np.zeros(4, dtype=np.float64)
    )
    points_2d = points_2d.reshape(-1, 2).astype(int)

    origin = tuple(points_2d[3])
    cv2.line(img, origin, tuple(points_2d[0]), (0, 0, 255), 2)
    cv2.line(img, origin, tuple(points_2d[1]), (0, 255, 0), 2)
    cv2.line(img, origin, tuple(points_2d[2]), (255, 0, 0), 2)
    return img


CHECKPOINT_PATH = "checkpoints/2026-04-02_14-56-01_obj1_drill_hd+drill_cut+sl_drill2+sl_real/checkpoint.pth"
CAMERA_CALIB_PATH = "dataset/realfootage/drill3/calibration/"
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

while True:
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    display_frame = cv2.undistort(frame, camera_matrix, distortion_coeffs)
    img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img = pvnet_transform.transform(img)

    with torch.no_grad():
        mask, vfield = pvnet.forward(img.unsqueeze(0).to(device))

    mask_proba = torch.sigmoid(mask.squeeze(0).squeeze(0))
    mask_binary = (mask_proba > RANSAC_THRESHOLD).float()

    # If the mask is too small skip RANSAC 
    # and display frame and mask
    print(f"mask pixels: {mask_binary.sum().item():.0f}")
    if int(mask_binary.sum().item()) < MIN_MASK_PIXELS:
        mask_vis = (mask_proba.cpu().numpy() * 255).astype(np.uint8)
        mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_INFERNO)
        mask_vis = cv2.resize(
            mask_vis, (display_frame.shape[1], display_frame.shape[0])
        )
        side_by_side = np.hstack([display_frame, mask_vis])

        cv2.imshow("frame", side_by_side)
        if cv2.waitKey(1) == ord("q"):
            break

        continue

    keypoints = PVNetRansac(mask_binary, vfield.squeeze(0), num_iter=512).ransac()

    orig_keypoints = pvnet_transform.inverse_transform_keypoints(
        keypoints.cpu().numpy(), display_frame.shape[0], display_frame.shape[1]
    )

    valid_keypoints = np.isfinite(orig_keypoints).all(axis=1)
    if valid_keypoints.sum() >= 4:
        success, rvec, tvec = cv2.solvePnP(
            keypoints_3d[valid_keypoints],
            orig_keypoints[valid_keypoints].astype(np.float64).reshape(-1, 1, 2),
            camera_matrix,
            distCoeffs=np.zeros(4, dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success:
            draw_axes(display_frame, rvec, tvec, camera_matrix)

    for x, y in orig_keypoints:
        if is_point_in_image(x, y, display_frame.shape[1], display_frame.shape[0]):
            cv2.circle(
                display_frame,
                (int(x), int(y)),
                radius=3,
                color=(0, 255, 0),
                thickness=-1,
            )

    # Display
    mask_vis = (mask_proba.cpu().numpy() * 255).astype(np.uint8)
    mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_INFERNO)
    mask_vis = cv2.resize(mask_vis, (display_frame.shape[1], display_frame.shape[0]))
    side_by_side = np.hstack([display_frame, mask_vis])

    cv2.imshow("frame", side_by_side)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
