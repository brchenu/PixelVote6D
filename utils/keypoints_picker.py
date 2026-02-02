import open3d as o3d
import numpy as np


def pick_keypoints(model_path: str, output_path: str, sample_size: int):
    mesh = o3d.io.read_triangle_mesh(model_path)
    point_cloud = mesh.sample_points_uniformly(number_of_points=1000)

    o3d.visualization.draw_geometries_with_editing([mesh])
    # o3d.visualization.draw_geometries_with_editing([point_cloud])

    fps_pcd = point_cloud.farthest_point_down_sample(sample_size)
    keypoints = np.asarray(fps_pcd.points)

    print(keypoints)
    
    np.savetxt(output_path, keypoints)

idx = "000010"

pick_keypoints(
    f"dataset/lm/models/obj_{idx}.ply",
    f"dataset/lm/models/obj_{idx}_keypoints.txt",
    8,
) 