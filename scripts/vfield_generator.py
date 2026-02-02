import os
import argparse
import numpy as np
from bop_toolkit.dataloader import load_data
from utils.vector_field import generate_vector_field
from utils.algebra import project_points


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--dataset", type=str, help="Dataset directory")
    argparser.add_argument("-s", "--scene", type=str, help="Scene folder name")
    argparser.add_argument("-id", "--obj_id", type=int, help="BOP object ID")
    argparser.add_argument("-o", "--output", type=str, help="Output directory")
    args = argparser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for idx, (keypoints, img, mask, K, R, t) in enumerate(
        load_data(args.dataset, args.scene, args.obj_id)
    ):

        project_keypoints = project_points(keypoints, K, R, t)

        vector_field = generate_vector_field(
            img.shape[0], img.shape[1], mask, project_keypoints
        )
        print(f"vector_field shape: {vector_field.shape}")

        # # Create out dir
        curr_dir = os.path.join(args.output, str(idx).zfill(6))
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)

        np.save(
            os.path.join(curr_dir, f"vector_field_{str(idx).zfill(6)}.npy"),
            vector_field,
        )
        np.save(os.path.join(curr_dir, f"image_{str(idx).zfill(6)}.npy"), img)
        np.save(os.path.join(curr_dir, f"mask_{str(idx).zfill(6)}.npy"), mask)
        np.save(
            os.path.join(curr_dir, f"keypoints_{str(idx).zfill(6)}.npy"),
            project_keypoints,
        )
        np.savez(
            os.path.join(curr_dir, f"camera_params_{str(idx).zfill(6)}.npz"),
            K=K,
            R=R,
            t=t,
        )
