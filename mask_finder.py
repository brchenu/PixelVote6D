import os
import sys
import orjson


def find_masks_for_object(scene_folder: str, obj_id: int, dataset: str = "lm") -> dict:
    """Find all mask files containing a specific object ID in a scene folder.

    Args:
        scene_folder: Scene folder name (e.g., "000010")
        obj_id: Object ID to search for
        dataset: Dataset name ("lm" or "lmo")

    Returns:
        Dictionary mapping scene_id -> mask filename
    """
    scene_path = os.path.join("dataset", dataset, "train_pbr", scene_folder)
    scene_gt_path = os.path.join(scene_path, "scene_gt.json")

    if not os.path.exists(scene_gt_path):
        raise FileNotFoundError(f"scene_gt.json not found at: {scene_gt_path}")

    # Load scene ground truth
    with open(scene_gt_path, "rb") as f:
        scenes = orjson.loads(f.read())

    mask_files = {}

    # Iterate through each scene (frame)
    for scene_id, objects in scenes.items():
        # Find the index of the object with matching obj_id
        for idx, obj in enumerate(objects):
            if obj["obj_id"] == obj_id:
                # Mask filename format: {scene_id:06d}_{index:06d}.png
                mask_filename = f"{int(scene_id):06d}_{idx:06d}.png"
                mask_files[scene_id] = mask_filename
                break

    return mask_files


def print_results(mask_files: dict, obj_id: int, scene_folder: str):
    """Print the results in a readable format."""
    if not mask_files:
        print(f"No masks found for object ID {obj_id} in scene {scene_folder}")
        return

    print(
        f"Found {len(mask_files)} mask(s) for object ID {obj_id} in scene {scene_folder}:\n"
    )

    for scene_id in sorted(mask_files.keys(), key=int):
        print(f"  Scene {scene_id}: {mask_files[scene_id]}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python mask_finder.py <scene_folder> <obj_id> [dataset]")
        print("Example: python mask_finder.py 000010 10")
        print("Example: python mask_finder.py 000001 5 lmo")
        sys.exit(1)

    scene_folder = sys.argv[1]
    obj_id = int(sys.argv[2])
    dataset = sys.argv[3] if len(sys.argv) > 3 else "lm"

    try:
        mask_files = find_masks_for_object(scene_folder, obj_id, dataset)
        # print_results(mask_files, obj_id, scene_folder)
        print(mask_files)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
