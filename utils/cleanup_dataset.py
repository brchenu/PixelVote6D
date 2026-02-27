"""
Cleans up BOP train_pbr scene JSON files by removing frame entries whose
RGB image or any object mask is missing from disk.

Affected files (kept in sync):
  - scene_gt.json
  - scene_gt_info.json   (if present)
  - scene_camera.json    (if present)

Usage:
    python utils/cleanup_dataset.py --dataset dataset/drill
    python utils/cleanup_dataset.py --dataset dataset/drill --dry-run
"""

import argparse
import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def check_frame(scene_dir: Path, frame_id: str, num_objects: int) -> tuple[bool, list[str]]:
    """
    Returns (is_valid, list_of_missing_files).
    A frame is valid when:
      - rgb/{frame_id:06d}.jpg  exists
      - mask/{frame_id:06d}_{obj_idx:06d}.png  exists for every object
    """
    missing = []
    fid = int(frame_id)

    # --- RGB ---
    rgb_path = scene_dir / "rgb" / f"{fid:06d}.jpg"
    if not rgb_path.exists():
        missing.append(str(rgb_path))

    # --- Masks ---
    for obj_idx in range(num_objects):
        mask_path = scene_dir / "mask" / f"{fid:06d}_{obj_idx:06d}.png"
        if not mask_path.exists():
            missing.append(str(mask_path))

    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# Per-scene cleanup
# ---------------------------------------------------------------------------

def cleanup_scene(scene_dir: Path, dry_run: bool) -> None:
    gt_path = scene_dir / "scene_gt.json"
    if not gt_path.exists():
        print(f"  [SKIP] no scene_gt.json in {scene_dir}")
        return

    scene_gt = load_json(gt_path)
    total = len(scene_gt)
    removed_ids = []
    missing_report = []

    for frame_id, objects in scene_gt.items():
        valid, missing = check_frame(scene_dir, frame_id, len(objects))
        if not valid:
            removed_ids.append(frame_id)
            missing_report.extend(missing)

    if not removed_ids:
        print(f"  [OK]   {scene_dir.name}  ({total} frames, nothing to remove)")
        return

    print(f"  [FIX]  {scene_dir.name}  removing {len(removed_ids)}/{total} frames")
    for m in missing_report:
        print(f"         missing: {m}")

    if dry_run:
        return

    # --- Patch scene_gt.json ---
    cleaned_gt = {k: v for k, v in scene_gt.items() if k not in removed_ids}
    save_json(gt_path, cleaned_gt)

    # --- Patch scene_gt_info.json (same keys) ---
    info_path = scene_dir / "scene_gt_info.json"
    if info_path.exists():
        info = load_json(info_path)
        cleaned_info = {k: v for k, v in info.items() if k not in removed_ids}
        save_json(info_path, cleaned_info)

    # --- Patch scene_camera.json (same keys) ---
    cam_path = scene_dir / "scene_camera.json"
    if cam_path.exists():
        cam = load_json(cam_path)
        cleaned_cam = {k: v for k, v in cam.items() if k not in removed_ids}
        save_json(cam_path, cleaned_cam)

    # --- Patch scene_gt_coco.json if present ---
    coco_path = scene_dir / "scene_gt_coco.json"
    if coco_path.exists():
        try:
            coco = load_json(coco_path)
            # COCO format: images list with "id" field, annotations with "image_id"
            if isinstance(coco, dict) and "images" in coco:
                removed_int = {int(i) for i in removed_ids}
                coco["images"] = [img for img in coco["images"] if img["id"] not in removed_int]
                coco["annotations"] = [ann for ann in coco["annotations"]
                                       if ann["image_id"] not in removed_int]
                save_json(coco_path, coco)
        except Exception as e:
            print(f"         [WARN] could not patch scene_gt_coco.json: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Remove orphaned frame entries from BOP scene JSONs.")
    parser.add_argument("--dataset", required=True, help="Root of the dataset, e.g. dataset/drill")
    parser.add_argument("--dry-run", action="store_true", help="Only report issues, do not write files")
    args = parser.parse_args()

    pbr_root = Path(args.dataset) / "train_pbr"
    if not pbr_root.exists():
        print(f"ERROR: {pbr_root} does not exist")
        return

    scene_dirs = sorted(d for d in pbr_root.iterdir() if d.is_dir())
    if not scene_dirs:
        print("No scene directories found.")
        return

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"{mode}Scanning {len(scene_dirs)} scene(s) in {pbr_root}\n")

    for scene_dir in scene_dirs:
        cleanup_scene(scene_dir, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
