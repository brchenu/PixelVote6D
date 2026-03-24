"""
pose_annotator.py

Overlay a 3D model on a real image and manually adjust the 6-DOF pose until
it aligns. Outputs the pose in BOP format (cam_R_m2c, cam_t_m2c in mm).

Controls
--------
  W / S       translate +Z / -Z  (farther / closer)
  A / D       translate -X / +X  (left / right in image)
  R / F       translate -Y / +Y  (up / down in image)
  I / K       rotate model around world X
  J / L       rotate model around world Y
  U / O       rotate model around world Z
  = / -       double / halve step sizes
  P           print current pose to terminal
  Enter       save pose to JSON and exit
  Esc         exit without saving

The coordinate frame follows BOP convention:
  x_camera = cam_R_m2c @ x_model_mm + cam_t_m2c_mm

GLB models are converted automatically from glTF space to BOP mm space.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
import open3d as o3d
import open3d.visualization.rendering as rendering


# ---------------------------------------------------------------------------
# Camera loading
# ---------------------------------------------------------------------------

def load_camera(camera_path: str, image: np.ndarray | None = None):
    """
    Load camera intrinsics. Supports:
      - BOP camera.json  (keys: fx, fy, cx, cy, width, height)
      - Plain 3x3 K matrix text file  (width/height taken from the background image)
    Returns: K (3x3 float64), width (int), height (int)
    """
    p = Path(camera_path)
    if p.suffix == ".json":
        with open(p) as f:
            cam = json.load(f)
        K = np.array([[cam["fx"], 0.0, cam["cx"]],
                      [0.0, cam["fy"], cam["cy"]],
                      [0.0, 0.0,       1.0      ]], dtype=np.float64)
        return K, int(cam["width"]), int(cam["height"])
    else:
        K = np.loadtxt(camera_path, dtype=np.float64)
        assert K.shape == (3, 3), f"Expected 3x3 matrix in {camera_path}, got {K.shape}"
        if image is not None:
            H, W = image.shape[:2]
        else:
            raise ValueError("Provide the background image to infer width/height from a matrix txt file.")
        return K, W, H


# ---------------------------------------------------------------------------
# Model loading — output: Open3D mesh in BOP mm space
# ---------------------------------------------------------------------------

def _gltf_to_bop_mm(verts_m: np.ndarray) -> np.ndarray:
    """Convert glTF vertices (meters, Y-up Z-back) to BOP/OpenCV mm (Z-forward Y-down)."""
    return (verts_m * 1000.0)[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])


def load_model_bop_mm(model_path: str) -> o3d.geometry.TriangleMesh:
    """Load a GLB or PLY model and return an Open3D mesh with vertices in BOP mm."""
    if model_path.lower().endswith((".glb", ".gltf")):
        loaded = trimesh.load(model_path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
        verts = _gltf_to_bop_mm(np.asarray(loaded.vertices, dtype=np.float64))
        faces = np.asarray(loaded.faces)
        try:
            rgba = loaded.visual.to_color().vertex_colors
            colors = rgba[:, :3].astype(np.float64) / 255.0
        except Exception:
            colors = np.full((len(verts), 3), 0.7)
    else:
        tm = o3d.io.read_triangle_mesh(model_path)
        verts = np.asarray(tm.vertices, dtype=np.float64)
        faces = np.asarray(tm.triangles)
        colors = (np.asarray(tm.vertex_colors, dtype=np.float64)
                  if tm.has_vertex_colors() else np.full((len(verts), 3), 0.7))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# Rotation increments
# ---------------------------------------------------------------------------

def rot_x(a): c, s = np.cos(a), np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(a): c, s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_z(a): c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def make_extrinsic(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 extrinsic [R|t] (world -> camera) from R (3x3) and t (3,) in mm."""
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = t
    return E


# ---------------------------------------------------------------------------
# Main annotator
# ---------------------------------------------------------------------------

def annotate_pose(
    model_path: str,
    image_path: str,
    camera_path: str,
    output_path: str = "pose_out.json",
    initial_R: np.ndarray | None = None,
    initial_t: np.ndarray | None = None,
    overlay_alpha: float = 0.65,
) -> None:
    # ---- background image ----
    bg = cv2.imread(image_path)
    if bg is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ---- camera intrinsics ----
    K, W, H = load_camera(camera_path, image=bg)
    bg = cv2.resize(bg, (W, H))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # ---- model ----
    print("Loading model...")
    mesh = load_model_bop_mm(model_path)
    bb = mesh.get_axis_aligned_bounding_box()
    obj_radius = np.linalg.norm(
        np.asarray(bb.max_bound) - np.asarray(bb.min_bound)) * 0.5
    print(f"Model loaded. Bounding-box radius: {obj_radius:.1f} mm")

    # ---- renderer ----
    renderer = rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"   # show vertex colors, no lighting
    renderer.scene.add_geometry("model", mesh, mat)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    # ---- initial pose (object centred, placed at a reasonable Z depth) ----
    R = np.eye(3, dtype=np.float64)  if initial_R is None else initial_R.copy()
    t = np.array([0.0, 0.0, obj_radius * 4]) if initial_t is None else initial_t.copy()

    t_step = obj_radius * 0.02      # ~2% of object radius in mm
    r_step = np.deg2rad(2.0)

    # ---- render helpers ----
    def render_frame() -> np.ndarray:
        renderer.setup_camera(intrinsic, make_extrinsic(R, t))
        color = np.asarray(renderer.render_to_image())   # (H, W, 3) RGB uint8
        depth = np.asarray(
            renderer.render_to_depth_image(z_in_view_space=True))  # (H, W) float
        mask = np.isfinite(depth) & (depth > 0)
        bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        blended = bg_rgb.copy()
        blended[mask] = (overlay_alpha * color[mask].astype(np.float32) +
                         (1 - overlay_alpha) * bg_rgb[mask].astype(np.float32)
                         ).astype(np.uint8)
        return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

    def print_pose() -> None:
        print("\n--- Current Pose (BOP format) ---")
        print(f"  cam_R_m2c : {R.flatten().tolist()}")
        print(f"  cam_t_m2c : {t.tolist()}  (mm)")
        print(f"  step sizes: t={t_step:.2f} mm  r={np.rad2deg(r_step):.2f} deg\n")

    # ---- UI loop ----
    HELP = [
        "A/Z: closer/farther   Arrows: left/right/up/down",
        "J/K: rotX   H/L: rotY   O/P: rotZ",
        "= / -: step size   P: print   Enter: save   Esc: quit",
    ]
    cv2.namedWindow("Pose Annotator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Annotator", min(W, 1280), min(H, 720))
    print("Pose Annotator ready.")
    print_pose()

    while True:
        frame = render_frame()
        for i, line in enumerate(HELP):
            cv2.putText(frame, line, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA)
        cv2.imshow("Pose Annotator", frame)
        key = cv2.waitKey(30) & 0xFF

        if   key == 27:                    # Esc — quit
            print("Exiting without saving.")
            break
        elif key == 13:                    # Enter — save
            out = {"cam_R_m2c": R.flatten().tolist(),
                   "cam_t_m2c": t.tolist()}
            with open(output_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Pose saved to {output_path}")
            print_pose()
            break
        elif key == ord('P'): print_pose()
        # Translation
        elif key == ord('a'): t[2] += t_step          # A — closer
        elif key == ord('z'): t[2] -= t_step          # Z — farther
        elif key == 81:        t[0] -= t_step          # ← left
        elif key == 83:        t[0] += t_step          # → right
        elif key == 82:        t[1] -= t_step          # ↑ up
        elif key == 84:        t[1] += t_step          # ↓ down
        # Rotation: pre-multiply = rotate model around its own world axes
        elif key == ord('j'): R = rot_x( r_step) @ R  # J — rot +X
        elif key == ord('k'): R = rot_x(-r_step) @ R  # K — rot -X
        elif key == ord('h'): R = rot_y( r_step) @ R  # H — rot +Y
        elif key == ord('l'): R = rot_y(-r_step) @ R  # L — rot -Y
        elif key == ord('o'): R = rot_z( r_step) @ R  # O — rot +Z
        elif key == ord('p'): R = rot_z(-r_step) @ R  # P — rot -Z
        # Step size
        elif key in (ord('='), ord('+')):
            t_step *= 2; r_step *= 2
            print(f"step: t={t_step:.2f} mm  r={np.rad2deg(r_step):.2f} deg")
        elif key == ord('-'):
            t_step /= 2; r_step /= 2
            print(f"step: t={t_step:.2f} mm  r={np.rad2deg(r_step):.2f} deg")

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point — configure paths here
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MODEL_PATH  = "dataset/drill_hd/models/model.glb"
    IMAGE_PATH  = "dataset/realfootage/drill1/frames/frame_0070.png"
    CAMERA_PATH = "dataset/realfootage/drill1/calibration/camera_matrix.txt"
    OUTPUT_PATH = "pose_out.json"

    # To start from an existing BOP GT pose (e.g. from scene_gt.json):
    # INITIAL_R = np.array([...]).reshape(3, 3)
    # INITIAL_T = np.array([tx_mm, ty_mm, tz_mm])
    # annotate_pose(MODEL_PATH, IMAGE_PATH, CAMERA_PATH, OUTPUT_PATH,
    #               initial_R=INITIAL_R, initial_t=INITIAL_T)

    annotate_pose(MODEL_PATH, IMAGE_PATH, CAMERA_PATH, OUTPUT_PATH)
