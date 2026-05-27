"""
Remove duplicate spiral images across all/ training/ testing using SHA256.
Run before split_spiral_dataset.py or CNN training.
"""
import os

from patient_data import (
    SPIRAL_ALL,
    SPIRAL_TEST,
    SPIRAL_TRAIN,
    deduplicate_spiral_images,
    iter_spiral_image_paths,
)


def main():
    print("[INFO] === Mandatory spiral duplicate removal (SHA256) ===")
    roots = [SPIRAL_ALL, SPIRAL_TRAIN, SPIRAL_TEST]
    before = len(iter_spiral_image_paths(*roots))
    report = deduplicate_spiral_images(roots=roots, delete_duplicates=True)
    after = len(
        {
            h
            for h, a in zip(report.get("sha256", []), report.get("action", []))
            if a == "kept" and h
        }
    )
    removed = int((report["action"] == "removed").sum()) if len(report) else 0
    print(f"[INFO] Files before: {before} | unique kept: {after} | removed: {removed}")
    if removed > 0:
        print("[WARNING] Duplicates were present and removed — re-run split_spiral_dataset.py if needed.")
    else:
        print("[SUCCESS] No duplicate spiral images remain.")


if __name__ == "__main__":
    main()
