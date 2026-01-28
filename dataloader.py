import os
import re
import numpy as np
import cv2


def load_data(category: str):
    BASE_DIR = os.path.join("dataset", "lm")

    keypoints = np.loadtxt(
        os.path.join(BASE_DIR, "models", f"obj_{category}_keypoints.txt")
    )

    train_dir = os.path.join(BASE_DIR, "train_pbr", category)

    rgb_dir = os.path.join(train_dir, "rgb")
    img_files = sorted(os.listdir(rgb_dir))

    pose_file = os.path.join(BASE_DIR, "scene_gt.json")

    for img_filename in img_files:
        # Extract index from filename 
        idx = int(re.search(r"\d+", img_filename).group())
        img_path = os.path.join(rgb_dir, img_filename)

        img = cv2.imread(img_path)

        yield keypoints, img


# LINEMOD dataset cheat sheet
#
# - mask: H x W, binary mask of the object
# - amodal_mask: H x W, binary mask including occluded parts
# - farthest.txt: 3D coordinates of the 8 keypooints chosen via FPS (farthest point sampling on the 3D model)
# - pose: 4 x 4, ground truth object pose (rotation and translation)
# - JPEGImages: Raw RGB images


egg_box = "000010"

for idx, (keypoints, img) in enumerate(load_data(egg_box)):
    print(f"Sample {idx}\n")

    cv2.imshow("Color image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
