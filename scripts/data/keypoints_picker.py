"""
Keypoint picker for 3D models.

Samples a set of keypoints from a 3D mesh using Farthest Point Sampling (FPS)
and saves them to a .txt file in BOP format (millimeters).

The workflow is:
  1. The mesh is sampled into a point cloud and displayed for inspection.
  2. FPS selects N well-spread keypoints from the cloud.
  3. Keypoints are converted to mm (if the model is in meters) and axis-corrected
     for glTF/GLB models exported from Blender/BlenderProc.
  4. The final keypoints are visualized as red spheres on the mesh, then saved.

These keypoints are consumed by BOPDirectDataset during training and by the
inference/demo scripts for PnP pose estimation.

Usage:
    python keypoints_picker.py <model_path> <output_path> [--num-keypoints N] [--unit mm|m]
"""
import argparse
import numpy as np
import open3d as o3d


def _keypoint_spheres(
    keypoints: np.ndarray,
    radius_factor: float = 0.005,
    color: tuple = (1.0, 0.1, 0.1),
) -> list[o3d.geometry.TriangleMesh]:
    """Return a list of small spheres placed at each keypoint."""
    extent = np.linalg.norm(keypoints.max(axis=0) - keypoints.min(axis=0))
    radius = extent * radius_factor
    spheres = []
    for pt in keypoints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(pt)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        spheres.append(sphere)
    return spheres


def pick_keypoints(
    model_path: str,
    output_path: str,
    sample_size: int,
    model_unit: str = "mm",
):
    """
    Pick keypoints from a 3D model and save them in millimeters (BOP standard).

    Args:
        model_path: path to the 3D model (.ply, .obj, .glb, .gltf, ...)
        output_path: path where the keypoints .txt will be saved
        sample_size: number of keypoints to pick via FPS
        model_unit: unit of the model coordinates. Either "mm" (default) or "m".
                    BOP PLY models are in mm; BlenderProc GLB models are in m.
                    When "m", keypoints are multiplied by 1000 before saving.
    """
    mesh = o3d.io.read_triangle_mesh(model_path)
    point_cloud = mesh.sample_points_uniformly(number_of_points=1000)

    # Let the user inspect the cloud before committing
    o3d.visualization.draw_geometries_with_editing([point_cloud])

    fps_pcd = point_cloud.farthest_point_down_sample(sample_size)
    keypoints = np.asarray(fps_pcd.points)

    # Convert to millimeters if the model is in meters (BOP standard is mm)
    scale = 1000.0 if model_unit == "m" else 1.0
    keypoints_mm = keypoints * scale

    # GLB/GLTF from Blender/BlenderProc: glTF is Y-up, Z-toward-viewer.
    # Blender imports glTF with an axis swap to its Z-up, Y-forward convention,
    # and BlenderProc records cam_R_m2c relative to that Blender frame.
    # Correct conversion: BOP_X = glTF_X, BOP_Y = -glTF_Z, BOP_Z = glTF_Y
    if model_path.lower().endswith((".glb", ".gltf")):
        keypoints_mm = np.column_stack(
            [
                keypoints_mm[:, 0],
                -keypoints_mm[:, 2],
                keypoints_mm[:, 1],
            ]
        )

    print(keypoints_mm)

    # Show keypoints as red spheres overlaid on the mesh (in model units)
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    spheres = _keypoint_spheres(keypoints)
    o3d.visualization.draw_geometries([mesh] + spheres, window_name="Keypoints")

    np.savetxt(output_path, keypoints_mm)


parser = argparse.ArgumentParser(description="Pick keypoints from a 3D model using FPS")
parser.add_argument("model_path", type=str, help="Path to the 3D model (.ply, .obj, .glb, ...)")
parser.add_argument("output_path", type=str, help="Path where the keypoints .txt will be saved")
parser.add_argument("--num-keypoints", type=int, default=8, help="Number of keypoints to pick (default: 8)")
parser.add_argument("--unit", type=str, default="mm", choices=["mm", "m"], help="Unit of the model coordinates (default: mm)")
args = parser.parse_args()

pick_keypoints(args.model_path, args.output_path, args.num_keypoints, model_unit=args.unit)
