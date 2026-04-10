import open3d as o3d
import numpy as np

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
        keypoints_mm = np.column_stack([
            keypoints_mm[:, 0],
            -keypoints_mm[:, 2],
            keypoints_mm[:, 1],
        ])

    print(keypoints_mm)

    # Show keypoints as red spheres overlaid on the mesh (in model units)
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    spheres = _keypoint_spheres(keypoints)
    o3d.visualization.draw_geometries([mesh] + spheres, window_name="Keypoints")

    np.savetxt(output_path, keypoints_mm)

idx = "000010"

model_root_dir = "/home/brann/projects/bproc/clean_drill"

pick_keypoints(
    f"{model_root_dir}/model.glb",
    f"{model_root_dir}/clean_model_keypoints.txt",
    8,
    model_unit="m",  # BlenderProc GLB models are in meters; saved as mm for BOP
) 