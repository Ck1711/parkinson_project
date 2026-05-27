"""
Patient-wise splits, leakage-safe voice preprocessing, and multimodal pairing.
"""
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from model_utils import IMG_SIZE

VOICE_CSV = os.path.join("datasets", "voice", "current", "pd_speech_features.csv")
SPLITS_JSON = os.path.join("outputs", "patient_splits.json")
VOICE_SPLITS_JSON = os.path.join("outputs", "voice_patient_splits.json")
CNN_TRAIN_RECORDS_JSON = os.path.join("outputs", "cnn_train_records.json")
CNN_VAL_RECORDS_JSON = os.path.join("outputs", "cnn_val_records.json")
CNN_TEST_RECORDS_JSON = os.path.join("outputs", "cnn_test_records.json")
SELECTED_FEATURES_PATH = os.path.join("models", "selected_voice_features.txt")

METADATA_COLUMN_NAMES = {
    "patient_id",
    "image_path",
    "label",
    "_label",
    "id",
    "gender",
    "class",
    "status",
    "diagnosis",
}

SPIRAL_ALL = os.path.join("datasets", "spiral", "all")
SPIRAL_TRAIN = os.path.join("datasets", "spiral", "training")
SPIRAL_TEST = os.path.join("datasets", "spiral", "testing")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

SUSPICIOUS_ACC_THRESHOLD = 0.97


@dataclass
class SpiralRecord:
    path: str
    label: int
    patient_id: str
    split_folder: str


@dataclass
class VoiceSplitResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    patient_col: str
    target_col: str
    feature_cols: List[str]
    scaler: StandardScaler
    selector: SelectKBest
    raw_feature_cols: List[str]
    pruned_feature_cols: Optional[List[str]] = None
    variance_selector: Optional[VarianceThreshold] = None
    best_k: Optional[int] = None


def extract_image_patient_id(filepath: str) -> str:
    """
    Robust patient ID from spiral filename.
    Examples:
      V05HE01.png -> V05
      V11PA02.png -> V11
      healthy05.png / Healthy_05.jpg -> healthy_05
      parkinson12.png -> parkinson_12
    """
    name = os.path.splitext(os.path.basename(filepath))[0]
    upper = name.upper()

    # Clinical codes: V05HE01, V11PA02, V05HE1, etc.
    clinical = re.match(r"^([A-Z]\d{2})[A-Z]{2}\d+", upper)
    if clinical:
        return clinical.group(1)

    # Alternate: leading letter + digits before non-alphanumeric tail
    clinical2 = re.match(r"^([A-Z]\d{2,3})", upper)
    if clinical2 and re.search(r"[A-Z]{2}\d", upper):
        return clinical2.group(1)

    lower = name.lower()
    if lower.startswith("healthy"):
        m = re.search(r"(\d+)", name)
        num = m.group(1).lstrip("0") or "0" if m else name
        return f"healthy_{num}"
    if lower.startswith("parkinson"):
        m = re.search(r"(\d+)", name)
        num = m.group(1).lstrip("0") or "0" if m else name
        return f"parkinson_{num}"

    # Generic numeric subject id in filename
    generic = re.match(r"^([A-Za-z]+[_-]?\d+)", name)
    if generic:
        return generic.group(1).replace("-", "_").lower()

    return f"subject_{name}"


def normalize_patient_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_md5(path: str) -> str:
    """Legacy alias — prefer file_sha256 for duplicate detection."""
    return file_sha256(path)


def list_spiral_images(root: str, label: int, class_name: str) -> List[SpiralRecord]:
    folder = os.path.join(root, class_name)
    records = []
    if not os.path.isdir(folder):
        return records
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(IMAGE_EXTS):
            continue
        path = os.path.join(folder, fn)
        records.append(
            SpiralRecord(
                path=path,
                label=label,
                patient_id=extract_image_patient_id(path),
                split_folder=root,
            )
        )
    return records


def spiral_records_to_json(records: List[SpiralRecord]) -> List[dict]:
    """Serialize records without modification (exact paths/labels/patient_ids)."""
    return [
        {
            "image_path": r.path,
            "label": int(r.label),
            "patient_id": str(r.patient_id),
        }
        for r in records
    ]


def spiral_records_from_json(
    items: List[dict], split_folder: str
) -> List[SpiralRecord]:
    """Restore SpiralRecord list preserving JSON order exactly."""
    return [
        SpiralRecord(
            path=str(item["image_path"]),
            label=int(item["label"]),
            patient_id=str(item["patient_id"]),
            split_folder=split_folder,
        )
        for item in items
    ]


def save_cnn_record_lists(
    train_recs: List[SpiralRecord],
    test_recs: List[SpiralRecord],
    val_recs: Optional[List[SpiralRecord]] = None,
) -> None:
    """Persist exact CNN split records for fusion and evaluation."""
    os.makedirs(os.path.dirname(CNN_TRAIN_RECORDS_JSON) or ".", exist_ok=True)
    with open(CNN_TRAIN_RECORDS_JSON, "w", encoding="utf-8") as f:
        json.dump(spiral_records_to_json(train_recs), f, indent=2)
    if val_recs is not None:
        with open(CNN_VAL_RECORDS_JSON, "w", encoding="utf-8") as f:
            json.dump(spiral_records_to_json(val_recs), f, indent=2)
    with open(CNN_TEST_RECORDS_JSON, "w", encoding="utf-8") as f:
        json.dump(spiral_records_to_json(test_recs), f, indent=2)
    print(f"[SUCCESS] Saved CNN train records: {CNN_TRAIN_RECORDS_JSON} ({len(train_recs)})")
    if val_recs is not None:
        print(f"[SUCCESS] Saved CNN val records:   {CNN_VAL_RECORDS_JSON} ({len(val_recs)})")
    print(f"[SUCCESS] Saved CNN test records:  {CNN_TEST_RECORDS_JSON} ({len(test_recs)})")


def _bootstrap_cnn_records_from_manifest() -> None:
    """Write cnn_*_records.json from patient_splits.json (same CNN run, legacy installs)."""
    if not os.path.isfile(SPLITS_JSON):
        raise FileNotFoundError(
            f"Missing {CNN_TRAIN_RECORDS_JSON} and {SPLITS_JSON}. Run train_cnn_model.py."
        )
    with open(SPLITS_JSON, encoding="utf-8") as f:
        manifest = json.load(f)

    def _items(split: str) -> List[dict]:
        return [
            {
                "image_path": str(row["path"]),
                "label": int(row["label"]),
                "patient_id": str(row["patient_id"]),
            }
            for row in manifest[split]
        ]

    train_recs = spiral_records_from_json(_items("train"), "train")
    test_recs = spiral_records_from_json(_items("test"), "test")
    val_recs = None
    if "val" in manifest:
        val_recs = spiral_records_from_json(_items("val"), "val")
    print(
        f"[WARNING] Bootstrapping {CNN_TRAIN_RECORDS_JSON} / {CNN_TEST_RECORDS_JSON} "
        f"from {SPLITS_JSON} — re-run train_cnn_model.py to refresh."
    )
    save_cnn_record_lists(train_recs, test_recs, val_recs=val_recs)


def load_cnn_record_lists() -> Tuple[List[SpiralRecord], List[SpiralRecord]]:
    """Load exact CNN train/test record lists written by train_cnn_model.py."""
    if not os.path.isfile(CNN_TRAIN_RECORDS_JSON) or not os.path.isfile(
        CNN_TEST_RECORDS_JSON
    ):
        _bootstrap_cnn_records_from_manifest()
    with open(CNN_TRAIN_RECORDS_JSON, encoding="utf-8") as f:
        train_items = json.load(f)
    with open(CNN_TEST_RECORDS_JSON, encoding="utf-8") as f:
        test_items = json.load(f)
    train_recs = spiral_records_from_json(train_items, "train")
    test_recs = spiral_records_from_json(test_items, "test")
    print(f"[INFO] Loaded CNN train records: {len(train_recs)} from {CNN_TRAIN_RECORDS_JSON}")
    print(f"[INFO] Loaded CNN test records:  {len(test_recs)} from {CNN_TEST_RECORDS_JSON}")
    return train_recs, test_recs


def load_cnn_record_lists_with_val() -> Tuple[List[SpiralRecord], List[SpiralRecord], List[SpiralRecord]]:
    """Load exact CNN train/val/test record lists if available."""
    if not os.path.isfile(CNN_TRAIN_RECORDS_JSON) or not os.path.isfile(
        CNN_TEST_RECORDS_JSON
    ):
        _bootstrap_cnn_records_from_manifest()

    with open(CNN_TRAIN_RECORDS_JSON, encoding="utf-8") as f:
        train_items = json.load(f)
    with open(CNN_TEST_RECORDS_JSON, encoding="utf-8") as f:
        test_items = json.load(f)

    val_items: List[dict] = []
    if os.path.isfile(CNN_VAL_RECORDS_JSON):
        with open(CNN_VAL_RECORDS_JSON, encoding="utf-8") as f:
            val_items = json.load(f)
    elif os.path.isfile(SPLITS_JSON):
        with open(SPLITS_JSON, encoding="utf-8") as f:
            manifest = json.load(f)
        if "val" in manifest:
            val_items = [
                {
                    "image_path": str(row["path"]),
                    "label": int(row["label"]),
                    "patient_id": str(row["patient_id"]),
                }
                for row in manifest["val"]
            ]
            save_cnn_record_lists(
                spiral_records_from_json(train_items, "train"),
                spiral_records_from_json(test_items, "test"),
                val_recs=spiral_records_from_json(val_items, "val"),
            )

    train_recs = spiral_records_from_json(train_items, "train")
    val_recs = spiral_records_from_json(val_items, "val") if val_items else []
    test_recs = spiral_records_from_json(test_items, "test")
    print(f"[INFO] Loaded CNN train records: {len(train_recs)} from {CNN_TRAIN_RECORDS_JSON}")
    if val_recs:
        print(f"[INFO] Loaded CNN val records:   {len(val_recs)} from {CNN_VAL_RECORDS_JSON}")
    else:
        print("[INFO] No CNN val records found; proceeding with train/test only.")
    print(f"[INFO] Loaded CNN test records:  {len(test_recs)} from {CNN_TEST_RECORDS_JSON}")
    return train_recs, val_recs, test_recs


def print_cnn_record_audit(records: List[SpiralRecord], title: str) -> None:
    """Print record count, sample paths, class and patient counts for consistency checks."""
    labels = np.array([int(r.label) for r in records], dtype=np.int32)
    patients = [str(r.patient_id) for r in records]
    print(f"\n[INFO] === CNN record audit: {title} ===")
    print(f"[INFO] Record count: {len(records)}")
    print(f"[INFO] Unique patients: {len(set(patients))}")
    print(
        f"[INFO] Class counts — healthy(0): {int((labels == 0).sum())}, "
        f"parkinson(1): {int((labels == 1).sum())}"
    )
    print("[INFO] First 10 image paths (exact order):")
    for i, rec in enumerate(records[:10]):
        print(f"  [{i}] {rec.path} | label={rec.label} | patient_id={rec.patient_id}")


def collect_spiral_records(
    use_all_source: bool = True,
    include_train: bool = True,
    include_test: bool = True,
) -> List[SpiralRecord]:
    records: List[SpiralRecord] = []
    if use_all_source and os.path.isdir(SPIRAL_ALL):
        records.extend(list_spiral_images(SPIRAL_ALL, 0, "healthy"))
        records.extend(list_spiral_images(SPIRAL_ALL, 1, "parkinson"))
    else:
        if include_train and os.path.isdir(SPIRAL_TRAIN):
            records.extend(list_spiral_images(SPIRAL_TRAIN, 0, "healthy"))
            records.extend(list_spiral_images(SPIRAL_TRAIN, 1, "parkinson"))
        if include_test and os.path.isdir(SPIRAL_TEST):
            records.extend(list_spiral_images(SPIRAL_TEST, 0, "healthy"))
            records.extend(list_spiral_images(SPIRAL_TEST, 1, "parkinson"))
    return records


def patient_wise_split(
    records: List[SpiralRecord],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[SpiralRecord], List[SpiralRecord]]:
    if not records:
        return [], []
    groups = np.array([r.patient_id for r in records])
    y = np.array([r.label for r in records])
    n_splits = max(2, int(round(1.0 / test_size)))
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_idx, test_idx = next(sgkf.split(np.zeros(len(records)), y, groups))
    return [records[i] for i in train_idx], [records[i] for i in test_idx]


def patient_wise_split_train_val_test(
    records: List[SpiralRecord],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[SpiralRecord], List[SpiralRecord], List[SpiralRecord]]:
    if not records:
        return [], [], []
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.0")

    patient_groups: Dict[str, List[SpiralRecord]] = {}
    for record in records:
        patient_groups.setdefault(record.patient_id, []).append(record)

    patient_labels = {
        patient_id: int(group[0].label)
        for patient_id, group in patient_groups.items()
    }

    rng = np.random.RandomState(random_state)
    label_to_patients = {0: [], 1: []}
    for patient_id, label in patient_labels.items():
        label_to_patients[label].append(patient_id)

    train_ids, val_ids, test_ids = [], [], []
    for label, patients in label_to_patients.items():
        rng.shuffle(patients)
        n = len(patients)
        n_train = int(round(train_fraction * n))
        n_val = int(round(val_fraction * n))
        n_test = n - n_train - n_val
        train_ids.extend(patients[:n_train])
        val_ids.extend(patients[n_train : n_train + n_val])
        test_ids.extend(patients[n_train + n_val :])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    train_recs = [r for r in records if r.patient_id in set(train_ids)]
    val_recs = [r for r in records if r.patient_id in set(val_ids)]
    test_recs = [r for r in records if r.patient_id in set(test_ids)]
    return train_recs, val_recs, test_recs


def check_patient_overlap(
    train_recs: List[SpiralRecord], test_recs: List[SpiralRecord]
) -> List[str]:
    train_patients = {r.patient_id for r in train_recs}
    test_patients = {r.patient_id for r in test_recs}
    return sorted(train_patients & test_patients)


def check_image_hash_overlap(
    train_recs: List[SpiralRecord], test_recs: List[SpiralRecord]
) -> List[Tuple[str, str, str]]:
    train_hashes = {}
    duplicates = []
    for r in train_recs:
        train_hashes[file_sha256(r.path)] = r.path
    for r in test_recs:
        h = file_sha256(r.path)
        if h in train_hashes:
            duplicates.append((h, train_hashes[h], r.path))
    return duplicates


def records_to_arrays(
    records: List[SpiralRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paths, labels, patient group ids for StratifiedGroupKFold."""
    if not records:
        return np.array([]), np.array([]), np.array([])
    paths = np.array([r.path for r in records])
    labels = np.array([r.label for r in records], dtype=np.int32)
    groups = np.array([r.patient_id for r in records])
    return paths, labels, groups


def print_patient_grouping_stats(records: List[SpiralRecord], title: str = "Spiral"):
    """Images per patient and class distribution after patient ID extraction."""
    if not records:
        print(f"[WARNING] No records for {title} grouping stats.")
        return
    from collections import Counter

    per_patient = Counter(r.patient_id for r in records)
    labels = [r.label for r in records]
    print(f"\n[INFO] === Patient grouping ({title}) ===")
    print(f"[INFO] Unique patients: {len(per_patient)} | Total images: {len(records)}")
    imgs_per = list(per_patient.values())
    print(
        f"[INFO] Images per patient — min: {min(imgs_per)}, max: {max(imgs_per)}, "
        f"mean: {np.mean(imgs_per):.2f}"
    )
    print(
        f"[INFO] Class distribution — healthy: {labels.count(0)}, "
        f"parkinson: {labels.count(1)}"
    )
    multi = sum(1 for c in imgs_per if c > 1)
    print(f"[INFO] Patients with >1 image: {multi} ({100*multi/len(per_patient):.1f}%)")
    sample = list(per_patient.items())[:8]
    print(f"[INFO] Sample patient groups: {sample}")


def iter_spiral_image_paths(*roots: str) -> List[str]:
    """Collect image paths under healthy/ and parkinson/ subfolders (sorted)."""
    paths = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for class_name in ("healthy", "parkinson"):
            folder = os.path.join(root, class_name)
            if not os.path.isdir(folder):
                continue
            for dirpath, _, files in os.walk(folder):
                for fn in sorted(files):
                    if fn.lower().endswith(IMAGE_EXTS):
                        paths.append(os.path.join(dirpath, fn))
    return paths


def deduplicate_spiral_images(
    roots: Optional[List[str]] = None,
    report_path: str = os.path.join("outputs", "duplicate_spiral_report.csv"),
    delete_duplicates: bool = True,
) -> pd.DataFrame:
    """
    SHA256 deduplication across spiral folders. Keeps first seen file per hash;
    optionally deletes duplicate files from disk.
    """
    if roots is None:
        roots = [SPIRAL_ALL, SPIRAL_TRAIN, SPIRAL_TEST]
    roots = [r for r in roots if r and os.path.isdir(r)]

    rows = []
    hash_to_canonical: Dict[str, str] = {}
    removed = 0

    all_paths = iter_spiral_image_paths(*roots)
    print(f"\n[INFO] === Spiral SHA256 deduplication ({len(all_paths)} files scanned) ===")

    for path in all_paths:
        try:
            digest = file_sha256(path)
        except OSError as exc:
            rows.append(
                {
                    "sha256": "",
                    "action": "error",
                    "kept_path": "",
                    "removed_path": path,
                    "reason": str(exc),
                }
            )
            continue

        if digest in hash_to_canonical:
            kept = hash_to_canonical[digest]
            rows.append(
                {
                    "sha256": digest,
                    "action": "removed" if delete_duplicates else "duplicate_found",
                    "kept_path": kept,
                    "removed_path": path,
                    "reason": "duplicate_hash",
                }
            )
            if delete_duplicates and os.path.isfile(path) and path != kept:
                try:
                    os.remove(path)
                    removed += 1
                except OSError as exc:
                    rows.append(
                        {
                            "sha256": digest,
                            "action": "delete_failed",
                            "kept_path": kept,
                            "removed_path": path,
                            "reason": str(exc),
                        }
                    )
        else:
            hash_to_canonical[digest] = path
            rows.append(
                {
                    "sha256": digest,
                    "action": "kept",
                    "kept_path": path,
                    "removed_path": "",
                    "reason": "first_occurrence",
                }
            )

    report = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    report.to_csv(report_path, index=False)
    dup_count = int((report["action"] == "removed").sum()) if len(report) else 0
    remaining = len(hash_to_canonical)

    print(f"[INFO] Duplicates removed: {removed}")
    print(f"[INFO] Unique images remaining: {remaining}")
    print(f"[SUCCESS] Report saved: {report_path}")
    if dup_count > 0:
        print(f"[WARNING] {dup_count} duplicate hash entries found in report.")
        sample = report[report["action"] == "removed"].head(5)
        for _, row in sample.iterrows():
            print(f"  removed: {row['removed_path']} (kept: {row['kept_path']})")
    else:
        print("[SUCCESS] No duplicate SHA256 hashes across scanned spiral folders.")

    return report


def print_split_diagnostics(
    train_recs: List[SpiralRecord],
    test_recs: List[SpiralRecord],
    title: str = "Spiral",
):
    print(f"\n[INFO] === {title} split diagnostics ===")
    print(f"[INFO] Train images: {len(train_recs)} | Test images: {len(test_recs)}")
    print(
        f"[INFO] Unique patients — train: {len({r.patient_id for r in train_recs})} | "
        f"test: {len({r.patient_id for r in test_recs})}"
    )
    tr_labels = [r.label for r in train_recs]
    te_labels = [r.label for r in test_recs]
    print(
        f"[INFO] Train class counts — healthy: {tr_labels.count(0)}, "
        f"parkinson: {tr_labels.count(1)}"
    )
    print(
        f"[INFO] Test class counts — healthy: {te_labels.count(0)}, "
        f"parkinson: {te_labels.count(1)}"
    )
    overlap = check_patient_overlap(train_recs, test_recs)
    if overlap:
        print(f"[ERROR] Patient overlap train/test ({len(overlap)}): {overlap[:10]}...")
    else:
        print("[SUCCESS] No patient overlap between train and test.")
    hash_dup = check_image_hash_overlap(train_recs, test_recs)
    if hash_dup:
        print(f"[ERROR] {len(hash_dup)} duplicate image hash(es) across train/test.")
    else:
        print("[SUCCESS] No duplicate image hashes across train/test.")


def load_voice_table(csv_path: str = VOICE_CSV) -> Tuple[pd.DataFrame, str, str]:
    """Load voice CSV; keep patient id; do NOT impute with global test statistics."""
    df = pd.read_csv(csv_path, header=1)
    df = df.drop_duplicates()

    patient_col = None
    for c in df.columns:
        if str(c).lower() in ("id", "patient_id", "subject_id"):
            patient_col = c
            break
    if patient_col is None:
        raise ValueError("Voice CSV must contain an 'id' (patient) column.")

    target_col = None
    for c in df.columns:
        if str(c).lower() in ("class", "status", "diagnosis"):
            target_col = c
            break
    if target_col is None:
        target_col = df.columns[-1]

    numeric = df.select_dtypes(include=[np.number]).copy()
    if patient_col not in numeric.columns:
        numeric = numeric.copy()
        numeric.loc[:, patient_col] = df[patient_col].values

    numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric = numeric.copy()
    numeric.loc[:, "_label"] = (
        numeric[target_col] == numeric[target_col].max()
    ).astype(int)

    dup_patients = numeric[patient_col].duplicated().sum()
    print(f"[INFO] Voice rows: {len(numeric)} | duplicate patient rows: {dup_patients}")
    return numeric, patient_col, target_col


def get_raw_voice_feature_columns(
    voice_df: pd.DataFrame, patient_col: str, target_col: str
) -> List[str]:
    exclude = set(METADATA_COLUMN_NAMES) | {patient_col, target_col, "_label"}
    cols = [
        c
        for c in voice_df.columns
        if c not in exclude
        and str(c).lower() not in {x.lower() for x in exclude}
        and pd.api.types.is_numeric_dtype(voice_df[c])
    ]
    seen = set()
    return [c for c in cols if not (c in seen or seen.add(c))]


def aggregate_voice_by_patient(
    df: pd.DataFrame, patient_col: str, feature_cols: List[str]
) -> pd.DataFrame:
    return df.groupby(patient_col, as_index=False).agg(
        {**{c: "mean" for c in feature_cols}, "_label": lambda s: int(s.mode().iloc[0])}
    )


def patient_wise_voice_split_df(
    patient_df: pd.DataFrame,
    patient_col: str,
    feature_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    groups = patient_df[patient_col].astype(str).values
    y = patient_df["_label"].values
    n_splits = max(2, int(round(1.0 / test_size)))
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_idx, test_idx = next(sgkf.split(patient_df[feature_cols], y, groups))
    return (
        patient_df.iloc[train_idx].reset_index(drop=True),
        patient_df.iloc[test_idx].reset_index(drop=True),
    )


def impute_with_train_stats(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill NaN using training-set means only (prevents test leakage)."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    means = train_df[feature_cols].mean()
    train_df[feature_cols] = train_df[feature_cols].fillna(means)
    test_df[feature_cols] = test_df[feature_cols].fillna(means)
    return train_df, test_df


def print_class_distribution_and_weights(
    y: np.ndarray,
    label: str = "train",
    print_weights: bool = True,
) -> Tuple[Dict[int, float], float]:
    """Print class counts and return sklearn balanced weights + XGB scale_pos_weight."""
    labels = np.asarray(y).astype(int)
    unique, counts = np.unique(labels, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(unique, counts)}
    total = int(counts.sum())
    print(f"\n[INFO] === Class distribution ({label}, n={total}) ===")
    for cls in sorted(dist):
        name = "healthy" if cls == 0 else "parkinson"
        pct = 100.0 * dist[cls] / total if total else 0.0
        print(f"[INFO]   {name} (label={cls}): {dist[cls]} ({pct:.1f}%)")

    classes = np.array(sorted(dist.keys()))
    sk_weights = compute_class_weight("balanced", classes=classes, y=labels)
    weight_map = {int(c): float(w) for c, w in zip(classes, sk_weights)}
    scale_pos_weight = 1.0
    if 0 in dist and 1 in dist and dist[1] > 0:
        scale_pos_weight = dist[0] / dist[1]

    if print_weights:
        print(f"[INFO] compute_class_weight('balanced'): {weight_map}")
        print(f"[INFO] XGBoost scale_pos_weight (neg/pos): {scale_pos_weight:.4f}")
    return weight_map, scale_pos_weight


def remove_correlated_features(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.95,
) -> Tuple[List[str], List[str]]:
    """Drop one feature from each highly correlated pair (fit on train only)."""
    if len(feature_cols) < 2:
        return list(feature_cols), []
    corr = train_df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = sorted(
        {col for col in upper.columns if (upper[col] > threshold).any()}
    )
    kept = [c for c in feature_cols if c not in to_drop]
    return kept, to_drop


def prune_weak_voice_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    variance_threshold: float = 1e-5,
    correlation_threshold: float = 0.95,
) -> Tuple[List[str], VarianceThreshold, List[str], List[str]]:
    """
    Remove constant / near-constant (VarianceThreshold on train) and
    highly correlated columns (correlation matrix on train).
    """
    vt = VarianceThreshold(threshold=variance_threshold)
    vt.fit(train_df[feature_cols])
    var_kept = [feature_cols[i] for i, ok in enumerate(vt.get_support()) if ok]
    var_dropped = [c for c in feature_cols if c not in var_kept]

    corr_kept, corr_dropped = remove_correlated_features(
        train_df, var_kept, threshold=correlation_threshold
    )
    print(
        f"[INFO] Feature pruning (train-only): {len(feature_cols)} -> "
        f"{len(var_kept)} after variance -> {len(corr_kept)} after correlation"
    )
    if var_dropped:
        print(f"[INFO]   Dropped {len(var_dropped)} low-variance feature(s)")
    if corr_dropped:
        print(f"[INFO]   Dropped {len(corr_dropped)} correlated feature(s)")
    return corr_kept, vt, var_dropped, corr_dropped


def scale_train_transform_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train only; transform test."""
    if scaler is None:
        scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    return X_train, X_test, scaler


def select_k_best_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    pruned_cols: List[str],
    k: int,
) -> Tuple[np.ndarray, np.ndarray, SelectKBest, List[str]]:
    k_eff = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k_eff)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    support = selector.get_support()
    selected = [pruned_cols[i] for i, on in enumerate(support) if on]
    return X_train_sel, X_test_sel, selector, selected


def make_selected_voice_frames(
    X_train_sel: np.ndarray,
    X_test_sel: np.ndarray,
    selected_cols: List[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    patient_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_sel = pd.DataFrame(X_train_sel, columns=selected_cols)
    train_sel[patient_col] = train_df[patient_col].values
    train_sel["_label"] = train_df["_label"].values
    test_sel = pd.DataFrame(X_test_sel, columns=selected_cols)
    test_sel[patient_col] = test_df[patient_col].values
    test_sel["_label"] = test_df["_label"].values
    return train_sel, test_sel


def load_voice_patient_split(
    test_size: float = 0.2,
    random_state: int = 42,
    csv_path: str = VOICE_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str, List[str]]:
    """Patient-level split before any feature engineering (leak-free)."""
    voice_df, patient_col, target_col = load_voice_table(csv_path)
    raw_feature_cols = get_raw_voice_feature_columns(voice_df, patient_col, target_col)
    print(f"[INFO] Raw voice feature candidates: {len(raw_feature_cols)}")
    patient_df = aggregate_voice_by_patient(voice_df, patient_col, raw_feature_cols)
    print(f"[INFO] Unique voice patients: {len(patient_df)}")
    train_df, test_df = patient_wise_voice_split_df(
        patient_df, patient_col, raw_feature_cols, test_size, random_state
    )
    print_voice_diagnostics(train_df, test_df, patient_col)
    return train_df, test_df, patient_col, target_col, raw_feature_cols


def prepare_voice_split_pipeline(
    test_size: float = 0.2,
    random_state: int = 42,
    k_features: int = 150,
    prune_weak: bool = True,
    variance_threshold: float = 1e-5,
    correlation_threshold: float = 0.95,
) -> VoiceSplitResult:
    """
    Leak-free voice preprocessing:
      1) aggregate to patient level
      2) patient-wise split
      3) optional variance / correlation pruning (train-only)
      4) impute with train means
      5) scaler.fit(train) -> transform test
      6) SelectKBest.fit(train_scaled) -> transform test
    """
    train_df, test_df, patient_col, target_col, raw_feature_cols = load_voice_patient_split(
        test_size, random_state
    )

    vt = None
    pruned_cols = list(raw_feature_cols)
    if prune_weak:
        pruned_cols, vt, _, _ = prune_weak_voice_features(
            train_df,
            test_df,
            raw_feature_cols,
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
        )

    train_df, test_df = impute_with_train_stats(train_df, test_df, pruned_cols)
    X_train, X_test, scaler = scale_train_transform_test(train_df, test_df, pruned_cols)
    y_train = train_df["_label"].values

    k_eff = min(k_features, X_train.shape[1])
    X_train_sel, X_test_sel, selector, feature_cols = select_k_best_features(
        X_train, y_train, X_test, pruned_cols, k_eff
    )
    train_sel, test_sel = make_selected_voice_frames(
        X_train_sel, X_test_sel, feature_cols, train_df, test_df, patient_col
    )

    print(f"[INFO] Selected voice features (k={k_eff}): {len(feature_cols)}")
    return VoiceSplitResult(
        train_df=train_sel,
        test_df=test_sel,
        patient_col=patient_col,
        target_col=target_col,
        feature_cols=feature_cols,
        scaler=scaler,
        selector=selector,
        raw_feature_cols=raw_feature_cols,
        pruned_feature_cols=pruned_cols,
        variance_selector=vt,
        best_k=k_eff,
    )


def save_voice_splits(train_df, test_df, patient_col: str, path: str = VOICE_SPLITS_JSON):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "train_patients": train_df[patient_col].astype(str).tolist(),
        "test_patients": test_df[patient_col].astype(str).tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[SUCCESS] Voice patient splits saved: {path}")


def transform_voice_inference(
    df: pd.DataFrame,
    pruned_cols: List[str],
    scaler: StandardScaler,
    selector: SelectKBest,
    impute_means: Optional[pd.Series] = None,
) -> np.ndarray:
    """Leak-free inference: impute with saved train means, scale, then select."""
    X = df[pruned_cols].copy()
    if impute_means is not None:
        X = X.fillna(impute_means)
    else:
        X = X.fillna(X.mean())
    X_scaled = scaler.transform(X.values)
    return selector.transform(X_scaled)


def resolve_voice_feature_columns(
    voice_df: pd.DataFrame,
    patient_col: str,
    target_col: str = "class",
    selected_features_path: str = SELECTED_FEATURES_PATH,
) -> List[str]:
    exclude_cols = set(METADATA_COLUMN_NAMES) | {
        patient_col,
        target_col,
        "_label",
        "patient_id",
        "image_path",
        "label",
    }
    if os.path.isfile(selected_features_path):
        with open(selected_features_path, encoding="utf-8") as f:
            feature_cols = [line.strip() for line in f if line.strip()]
    else:
        feature_cols = [
            col
            for col in voice_df.columns
            if col not in exclude_cols
            and pd.api.types.is_numeric_dtype(voice_df[col])
        ]
    feature_cols = [
        col
        for col in feature_cols
        if col in voice_df.columns
        and col not in exclude_cols
        and str(col).lower() not in {x.lower() for x in exclude_cols}
    ]
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]
    missing = [c for c in feature_cols if c not in voice_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:5]}...")
    print(f"[INFO] Total voice columns: {len(voice_df.columns)} | features: {len(feature_cols)}")
    return feature_cols


def print_voice_diagnostics(train_df, test_df, patient_col: str):
    print("\n[INFO] === Voice split diagnostics ===")
    print(f"[INFO] Train patients: {len(train_df)} | Test patients: {len(test_df)}")
    overlap = set(train_df[patient_col].astype(str)) & set(test_df[patient_col].astype(str))
    if overlap:
        print(f"[ERROR] Voice patient overlap: {sorted(overlap)[:10]}")
    else:
        print("[SUCCESS] No voice patient overlap.")
    print(
        f"[INFO] Train labels — healthy: {(train_df['_label']==0).sum()}, "
        f"parkinson: {(train_df['_label']==1).sum()}"
    )
    print(
        f"[INFO] Test labels — healthy: {(test_df['_label']==0).sum()}, "
        f"parkinson: {(test_df['_label']==1).sum()}"
    )


def load_spiral_image(path: str, normalized: bool = False) -> np.ndarray:
    """RGB float32 [0,255] for EfficientNet; set normalized=True for matplotlib display."""
    from model_utils import load_spiral_rgb_float

    img = load_spiral_rgb_float(path)
    if normalized:
        return img / 255.0
    return img


def pair_voice_to_spiral(
    spiral_records: List[SpiralRecord],
    voice_df: pd.DataFrame,
    feature_cols: List[str],
    patient_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pair each spiral image with a voice vector from the same split and class.
    Prefer patient ID match (normalized); else deterministic hash by spiral patient_id.
    """
    voice_work = voice_df.reset_index(drop=True).copy()
    voice_work["_pid_norm"] = voice_work[patient_col].astype(str).map(normalize_patient_key)

    pools = {
        0: voice_work[voice_work["_label"] == 0],
        1: voice_work[voice_work["_label"] == 1],
    }
    patient_voice_cache: Dict[Tuple[int, str], pd.Series] = {}

    X_voice, X_img, y = [], [], []
    matched = 0
    for rec in spiral_records:
        pool = pools[rec.label]
        if pool.empty:
            continue
        pid_norm = normalize_patient_key(rec.patient_id)
        cache_key = (rec.label, pid_norm)
        if cache_key in patient_voice_cache:
            row = patient_voice_cache[cache_key]
        else:
            exact = pool[pool["_pid_norm"] == pid_norm]
            if not exact.empty:
                row = exact.iloc[0]
                matched += 1
            else:
                idx = int(hashlib.sha256(rec.patient_id.encode()).hexdigest(), 16) % len(pool)
                row = pool.iloc[idx]
            patient_voice_cache[cache_key] = row

        X_voice.append(row[feature_cols].to_numpy(dtype=np.float32))
        X_img.append(load_spiral_image(rec.path))
        y.append(rec.label)

    if y:
        print(
            f"[INFO] Voice-spiral pairing: {matched}/{len(spiral_records)} "
            f"images matched voice patient ID; others use class-pool hash."
        )
    if not y:
        return np.empty((0, len(feature_cols))), np.empty((0, IMG_SIZE, IMG_SIZE, 3)), np.empty(0)
    return np.stack(X_voice), np.stack(X_img), np.array(y, dtype=np.int32)


def pair_late_fusion_features(
    spiral_records: List[SpiralRecord],
    voice_df: pd.DataFrame,
    feature_cols: List[str],
    patient_col: str,
    cnn_cache: Dict[str, dict],
    xgb_model,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Class-aware pseudo pairing for late fusion only (no image tensors).
    CNN features come from precomputed cache keyed by normalized image path.
    """
    voice_work = voice_df.reset_index(drop=True).copy()
    voice_work["_pid_norm"] = voice_work[patient_col].astype(str).map(normalize_patient_key)
    pools = {
        0: voice_work[voice_work["_label"] == 0],
        1: voice_work[voice_work["_label"] == 1],
    }
    patient_voice_cache: Dict[Tuple[int, str], pd.Series] = {}

    X_voice, voice_prob, cnn_embed, cnn_prob, y = [], [], [], [], []
    matched_voice = 0
    for rec in spiral_records:
        key = os.path.normpath(rec.path)
        if key not in cnn_cache:
            continue
        pool = pools[rec.label]
        if pool.empty:
            continue
        pid_norm = normalize_patient_key(rec.patient_id)
        cache_key = (rec.label, pid_norm)
        if cache_key in patient_voice_cache:
            row = patient_voice_cache[cache_key]
        else:
            exact = pool[pool["_pid_norm"] == pid_norm]
            if not exact.empty:
                row = exact.iloc[0]
                matched_voice += 1
            else:
                idx = int(hashlib.sha256(rec.patient_id.encode()).hexdigest(), 16) % len(pool)
                row = pool.iloc[idx]
            patient_voice_cache[cache_key] = row

        vfeat = row[feature_cols].to_numpy(dtype=np.float32)
        vprob = float(xgb_model.predict_proba(vfeat.reshape(1, -1))[0, 1])
        X_voice.append(vfeat)
        voice_prob.append(vprob)
        cnn_embed.append(cnn_cache[key]["embedding"])
        cnn_prob.append(float(cnn_cache[key]["prob"]))
        y.append(rec.label)

    if y:
        print(
            f"[INFO] Late-fusion pairs: {len(y)} | voice ID matched: {matched_voice}/"
            f"{len(spiral_records)}"
        )
    if not y:
        dim = len(feature_cols)
        embed_dim = next(iter(cnn_cache.values()))["embedding"].shape[0] if cnn_cache else 0
        return (
            np.empty((0, dim)),
            np.empty((0, 1)),
            np.empty((0, embed_dim)),
            np.empty((0, 1)),
            np.empty(0),
        )
    return (
        np.stack(X_voice),
        np.array(voice_prob, dtype=np.float32).reshape(-1, 1),
        np.stack(cnn_embed),
        np.array(cnn_prob, dtype=np.float32).reshape(-1, 1),
        np.array(y, dtype=np.int32),
    )


def split_records_train_val(
    records: List[SpiralRecord],
    val_fraction: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[SpiralRecord], List[SpiralRecord]]:
    """Patient-wise train/val split inside training pool (for fusion early stopping)."""
    if not records or val_fraction <= 0:
        return records, []
    groups = np.array([r.patient_id for r in records])
    y = np.array([r.label for r in records])
    n_splits = max(2, int(round(1.0 / val_fraction)))
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_idx, val_idx = next(sgkf.split(np.zeros(len(records)), y, groups))
    return [records[i] for i in train_idx], [records[i] for i in val_idx]


def check_overfitting_gap(
    train_acc: float, val_acc: float, name: str = "Model", threshold: float = 0.08
) -> None:
    gap = train_acc - val_acc
    print(
        f"[INFO] {name} — train acc: {train_acc*100:.2f}% | val acc: {val_acc*100:.2f}% | gap: {gap*100:.2f}%"
    )
    if gap > threshold:
        print(f"[WARNING] {name} overfitting detected (train - val accuracy > {threshold*100:.0f}%)")


def warn_suspicious_accuracy(name: str, accuracy: float, threshold: float = SUSPICIOUS_ACC_THRESHOLD):
    if accuracy >= threshold:
        print(
            f"\n[WARNING] {name} accuracy {accuracy*100:.2f}% >= {threshold*100:.0f}%. "
            "Investigate leakage, duplicate patients, and feature contamination."
        )
        investigate_leakage(name)


def investigate_leakage(context: str = "Fusion"):
    """Automatic leakage investigation when metrics look suspiciously high."""
    print(f"\n[INFO] === Leakage investigation ({context}) ===")
    roots = [SPIRAL_ALL, SPIRAL_TRAIN, SPIRAL_TEST]
    paths = iter_spiral_image_paths(*roots)
    if paths:
        deduplicate_spiral_images(roots=roots, delete_duplicates=False)
    train_recs = collect_spiral_records(
        use_all_source=False, include_train=True, include_test=False
    )
    test_recs = collect_spiral_records(
        use_all_source=False, include_train=False, include_test=True
    )
    if train_recs and test_recs:
        print_split_diagnostics(train_recs, test_recs, title=context)
        dups = check_image_hash_overlap(train_recs, test_recs)
        if dups:
            print(f"[ERROR] {len(dups)} duplicate image hash(es) train/test.")
    else:
        print("[INFO] Could not load train/test spiral folders for overlap check.")
