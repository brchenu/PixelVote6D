import os
import argparse
import numpy as np
from bop_toolkit.dataloader import load_data
from bop_toolkit.vizualization import show_vector_field
from utils.vector_field import generate_vector_field
from utils.algebra import project_points


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Generate vector fields from BOP dataset for PVNet training"
    )
    argparser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Dataset directory (e.g., dataset/lm)",
    )
    argparser.add_argument(
        "-s",
        "--scene",
        type=str,
        required=True,
        help="Scene folder name (e.g., 000010)",
    )
    argparser.add_argument(
        "-id", "--obj_id", type=int, required=True, help="BOP object ID (e.g., 10)"
    )
    argparser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory for generated data (required unless --debug is set)",
    )
    argparser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with visualizations"
    )
    args = argparser.parse_args()

    # Validate that output is provided when not in debug mode
    if not args.debug and not args.output:
        argparser.error("--output/-o is required when --debug is not set")

    # Create output directory only if not in debug mode
    if args.output and not args.debug:
        if not os.path.exists(args.output):
            os.makedirs(args.output)

    for idx, (keypoints, img, mask, K, R, t) in enumerate(
        load_data(args.dataset, args.scene, args.obj_id)
    ):
        project_keypoints = project_points(keypoints, K, R, t)

        vector_field = generate_vector_field(
            img.shape[0], img.shape[1], mask, project_keypoints
        )

        if args.debug:
            # Visualize vector fields with full displacement vectors
            show_vector_field(
                img, mask, vector_field, project_keypoints, scale_mode="full"
            )
        else:
            # Save generated data only when not in debug mode
            curr_dir = os.path.join(args.output, f"scene_{args.scene}", f"{idx:06d}")
            os.makedirs(curr_dir, exist_ok=True)

            np.save(os.path.join(curr_dir, "vector_field.npy"), vector_field)
            np.save(os.path.join(curr_dir, "image.npy"), img)
            np.save(os.path.join(curr_dir, "mask.npy"), mask)
            np.save(os.path.join(curr_dir, "2d_keypoints.npy"), project_keypoints)
            np.savez(
                os.path.join(curr_dir, "camera_params.npz"),
                K=K, R=R, t=t
            )
