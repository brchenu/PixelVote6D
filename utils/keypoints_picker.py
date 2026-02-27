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


def pick_keypoints(model_path: str, output_path: str, sample_size: int):
    mesh = o3d.io.read_triangle_mesh(model_path)
    point_cloud = mesh.sample_points_uniformly(number_of_points=1000)

    # Let the user inspect the cloud before committing
    o3d.visualization.draw_geometries_with_editing([point_cloud])

    fps_pcd = point_cloud.farthest_point_down_sample(sample_size)
    keypoints = np.asarray(fps_pcd.points)

    print(keypoints)

    # Show keypoints as red spheres overlaid on the mesh
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    spheres = _keypoint_spheres(keypoints)
    o3d.visualization.draw_geometries([mesh] + spheres, window_name="Keypoints")

    np.savetxt(output_path, keypoints)

idx = "000010"

model_root_dir = "/home/brann/projects/bproc/clean_drill"

pick_keypoints(
    f"{model_root_dir}/model.glb",
    f"{model_root_dir}/model_keypoints.txt",
    8,
) 