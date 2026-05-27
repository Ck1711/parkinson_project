"""
Patient-wise spiral split: SHA256 dedup, robust patient groups, copy to training/testing.
"""
import os
import shutil

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from patient_data import (
    SPIRAL_ALL,
    deduplicate_spiral_images,
    extract_image_patient_id,
    file_sha256,
    iter_spiral_image_paths,
    print_patient_grouping_stats,
)


def gather_images(directory):
    paths = []
    for dirpath, _, files in os.walk(directory):
        for fn in sorted(files):
            if fn.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            ):
                paths.append(os.path.join(dirpath, fn))
    return paths


def reset_dirs(paths):
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def copy_files_unique(file_paths, target_dir):
    """Copy with hash-prefixed names to avoid basename collisions."""
    for source_path in file_paths:
        digest = file_sha256(source_path)[:12]
        base = os.path.basename(source_path)
        destination_path = os.path.join(target_dir, f"{digest}_{base}")
        shutil.copy2(source_path, destination_path)


def build_unique_list(source_dir, class_name, global_hashes):
    image_paths = gather_images(source_dir)
    unique_files = []
    duplicates = 0

    for path in image_paths:
        file_hash = file_sha256(path)
        if file_hash in global_hashes:
            duplicates += 1
            existing_class, existing_path = global_hashes[file_hash]
            if existing_class != class_name:
                print(
                    f"[WARNING] Duplicate across classes ({existing_class}/{class_name}): {path}"
                )
                print(f"           Kept: {existing_path}")
            else:
                print(f"[WARNING] Duplicate skipped ({class_name}): {path}")
            continue
        global_hashes[file_hash] = (class_name, path)
        unique_files.append((path, file_hash))

    return unique_files, duplicates


def verify_no_cross_duplicates(train_dirs, test_dirs):
    train_hashes = {}
    duplicates = []
    for directory in train_dirs:
        for path in gather_images(directory):
            file_hash = file_sha256(path)
            train_hashes.setdefault(file_hash, []).append(path)
    for directory in test_dirs:
        for path in gather_images(directory):
            file_hash = file_sha256(path)
            if file_hash in train_hashes:
                duplicates.append((file_hash, path, train_hashes[file_hash]))
    return duplicates


def main():
    base_source = SPIRAL_ALL
    healthy_source = os.path.join(base_source, "healthy")
    parkinson_source = os.path.join(base_source, "parkinson")

    healthy_train = os.path.join("datasets", "spiral", "training", "healthy")
    healthy_test = os.path.join("datasets", "spiral", "testing", "healthy")
    parkinson_train = os.path.join("datasets", "spiral", "training", "parkinson")
    parkinson_test = os.path.join("datasets", "spiral", "testing", "parkinson")

    print("[INFO] --- Spiral Dataset Splitter (SHA256 + patient-wise) ---")

    for folder in [healthy_source, parkinson_source]:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Required source folder not found: {folder}")

    deduplicate_spiral_images(roots=[base_source], delete_duplicates=True)

    reset_dirs([healthy_train, healthy_test, parkinson_train, parkinson_test])

    global_hashes = {}
    healthy_files, healthy_dup = build_unique_list(healthy_source, "healthy", global_hashes)
    parkinson_files, parkinson_dup = build_unique_list(
        parkinson_source, "parkinson", global_hashes
    )

    print(f"[INFO] Healthy duplicates removed: {healthy_dup}")
    print(f"[INFO] Parkinson duplicates removed: {parkinson_dup}")
    print(f"[INFO] Unique healthy: {len(healthy_files)} | unique parkinson: {len(parkinson_files)}")

    all_paths = [path for path, _ in healthy_files] + [path for path, _ in parkinson_files]
    all_labels = [0] * len(healthy_files) + [1] * len(parkinson_files)
    all_groups = [extract_image_patient_id(p) for p in all_paths]

    from patient_data import SpiralRecord

    preview = [
        SpiralRecord(p, all_labels[i], all_groups[i], "all")
        for i, p in enumerate(all_paths)
    ]
    print_patient_grouping_stats(preview, title="all (pre-split)")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(np.zeros(len(all_paths)), all_labels, all_groups))

    train_paths = [all_paths[i] for i in train_idx]
    test_paths = [all_paths[i] for i in test_idx]

    healthy_set = {path for path, _ in healthy_files}
    healthy_train_paths = [p for p in train_paths if p in healthy_set]
    parkinson_train_paths = [p for p in train_paths if p not in healthy_set]
    healthy_test_paths = [p for p in test_paths if p in healthy_set]
    parkinson_test_paths = [p for p in test_paths if p not in healthy_set]

    train_patients = {all_groups[i] for i in train_idx}
    test_patients = {all_groups[i] for i in test_idx}
    overlap = train_patients & test_patients
    if overlap:
        raise RuntimeError(f"Patient overlap after group split: {list(overlap)[:5]}")
    print(
        f"[SUCCESS] Patient-wise split: {len(train_patients)} train / "
        f"{len(test_patients)} test patients"
    )

    copy_files_unique(healthy_train_paths, healthy_train)
    copy_files_unique(healthy_test_paths, healthy_test)
    copy_files_unique(parkinson_train_paths, parkinson_train)
    copy_files_unique(parkinson_test_paths, parkinson_test)

    print("\n[INFO] Split counts:")
    print(f"  Healthy training: {len(healthy_train_paths)}")
    print(f"  Healthy testing: {len(healthy_test_paths)}")
    print(f"  Parkinson training: {len(parkinson_train_paths)}")
    print(f"  Parkinson testing: {len(parkinson_test_paths)}")

    cross_duplicates = verify_no_cross_duplicates(
        [healthy_train, parkinson_train], [healthy_test, parkinson_test]
    )
    if cross_duplicates:
        print("[ERROR] Duplicate SHA256 across train and test:")
        for file_hash, test_path, train_paths in cross_duplicates[:5]:
            print(f"  {file_hash[:12]}... test={test_path} train={train_paths[0]}")
        raise RuntimeError("Duplicate image leakage detected after split.")

    print("[SUCCESS] Dataset split completed — no cross-split duplicate hashes.")


if __name__ == "__main__":
    main()
