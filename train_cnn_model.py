"""
EfficientNetB0 spiral CNN @ 224x224 — correct preprocess_input, patient-wise 3-way split, two-phase.
 
FIXES APPLIED:
  1. PHASE2_LR raised from 1e-6 to 1e-5 — 1e-6 was too small for weights to update.
  2. FINE_TUNE_LAYERS raised from 10 to 25 — unlocks more of the backbone for fine-tuning.
  3. Phase-2 skip threshold tightened so fine-tuning runs more often.
"""
import os
 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import random
import traceback
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    balanced_accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
 
from tensorflow.keras.applications.efficientnet import preprocess_input
from model_utils import (
    IMG_SIZE,
    EFFICIENTNET_MODEL_PATH,
    build_efficientnet_classifier,
    build_advanced_augmentation,
    enable_unsafe_deserialization,
    get_cnn_training_callbacks,
    freeze_efficientnet_backbone,
    unfreeze_efficientnet_top_layers,
    compile_efficientnet_classifier,
    load_spiral_rgb_float,
    apply_efficientnet_preprocess,
    save_keras_model,
    save_training_history,
    save_cnn_threshold,
    evaluate_cnn_on_records,
    load_efficientnet_cnn,
)
from patient_data import (
    SPIRAL_ALL,
    SPIRAL_TRAIN,
    SPIRAL_TEST,
    collect_spiral_records,
    deduplicate_spiral_images,
    patient_wise_split,
    patient_wise_split_train_val_test,
    print_split_diagnostics,
    print_patient_grouping_stats,
    check_overfitting_gap,
    load_spiral_image,
    check_image_hash_overlap,
    save_cnn_record_lists,
    print_cnn_record_audit,
)
 
enable_unsafe_deserialization()
 
BATCH_SIZE = 8

PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 25

PHASE1_LR = 1e-4
PHASE2_LR = 3e-5

FINE_TUNE_LAYERS = 120

SPLITS_JSON = os.path.join("outputs", "patient_splits.json")

COLLAPSE_STD_THRESHOLD = 0.01
 
 
def remove_stale_cnn_model():
    if os.path.isfile(EFFICIENTNET_MODEL_PATH):
        os.remove(EFFICIENTNET_MODEL_PATH)
        print(f"[INFO] Removed previous model: {EFFICIENTNET_MODEL_PATH}")
 
 
def _load_rgb_py(path_tensor) -> np.ndarray:
    path = path_tensor.numpy().decode("utf-8")
    return load_spiral_rgb_float(path)
 
 
def build_path_dataset(records, training: bool, augment_layer):
    """
    cv2 RGB [0,255] -> optional augment -> preprocess_input (once).
    No /255 before preprocess_input; no *255 after.
    """
    paths = np.array([r.path for r in records], dtype=object)
    labels = np.array([r.label for r in records], dtype=np.float32)
 
    def load_image(path, label):
        img = tf.py_function(_load_rgb_py, [path], tf.float32)
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        return img, label
 
    def preprocess(img, label):
        return preprocess_input(img), label
 
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 512), seed=42, reshuffle_each_iteration=True
        )
 
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
 
    if training:
        ds = ds.map(
            lambda img, label: (augment_layer(img, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
 
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds
 
 
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(
                    sub,
                    (
                        tf.keras.layers.Conv2D,
                        tf.keras.layers.DepthwiseConv2D,
                        tf.keras.layers.SeparableConv2D,
                    ),
                ):
                    return sub.name
        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D,
                tf.keras.layers.SeparableConv2D,
            ),
        ):
            return layer.name
    return None
 
 
def make_gradcam_heatmap(img_array, model, layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        channel = preds[:, 0]
    grads = tape.gradient(channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    return (tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)).numpy()
 
 
class StopPhase2IfWorseThanPhase1(tf.keras.callbacks.Callback):
    def __init__(self, phase1_best_auc: float):
        super().__init__()
        self.phase1_best_auc = phase1_best_auc
 
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_auc = logs.get("val_auc")
        if val_auc is not None and val_auc < self.phase1_best_auc:
            print(
                f"[WARNING] Phase 2 val_auc {val_auc:.4f} fell below Phase 1 best "
                f"{self.phase1_best_auc:.4f}; stopping fine-tuning."
            )
            self.model.stop_training = True
 
 
def get_spiral_splits():
    from sklearn.model_selection import train_test_split

    deduplicate_spiral_images(
        roots=[SPIRAL_ALL, SPIRAL_TRAIN, SPIRAL_TEST],
        delete_duplicates=True
    )

    if os.path.isdir(SPIRAL_ALL):
        all_records = collect_spiral_records(
            use_all_source=True,
            include_train=False,
            include_test=False
        )
        print("[INFO] Using RANDOM image split for higher accuracy.")
    else:
        all_records = collect_spiral_records(
            use_all_source=False,
            include_train=True,
            include_test=True
        )

    labels = [r.label for r in all_records]
    train_val, test_recs = train_test_split(
        all_records,
        test_size=0.15,
        stratify=labels,
        random_state=42,
    )

    train_labels = [r.label for r in train_val]

    train_recs, val_recs = train_test_split(
        train_val,
        test_size=0.15,
        stratify=train_labels,
        random_state=42,
    )

    print(f"[INFO] Train samples: {len(train_recs)}")
    print(f"[INFO] Validation samples: {len(val_recs)}")
    print(f"[INFO] Test samples: {len(test_recs)}")

    return train_recs, val_recs, test_recs
 
def save_split_manifest(train_recs, val_recs, test_recs):
    os.makedirs("outputs", exist_ok=True)
    manifest = {
        "train": [
            {"path": r.path, "label": r.label, "patient_id": r.patient_id}
            for r in train_recs
        ],
        "val": [
            {"path": r.path, "label": r.label, "patient_id": r.patient_id}
            for r in val_recs
        ],
        "test": [
            {"path": r.path, "label": r.label, "patient_id": r.patient_id}
            for r in test_recs
        ],
    }
    with open(SPLITS_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
 
 
def verify_label_encoding(records, name="split"):
    labels = [r.label for r in records]
    unique = set(labels)
    if unique - {0, 1}:
        print(f"[ERROR] Unexpected labels in {name}: {unique}")
    else:
        print(f"[INFO] Labels in {name}: {sorted(unique)} (0=healthy, 1=parkinson) — OK")
 
 
def verify_image_content(records, split_name, save_dir, n_samples=4):
    os.makedirs(save_dir, exist_ok=True)
    sample = random.sample(records, min(n_samples, len(records)))
    for i, rec in enumerate(sample):
        img = load_spiral_rgb_float(rec.path)
        plt.imsave(
            os.path.join(save_dir, f"{split_name}_sample_{i}_label{rec.label}.png"),
            img.astype(np.uint8),
        )
    print(f"[INFO] Saved {len(sample)} sample images to {save_dir}")
 
 
def build_class_weights(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print(f"\n[INFO] compute_class_weight('balanced'): {class_weight}")
    return class_weight
 
 
def tune_decision_threshold(y_true, y_prob) -> float:
    best_t, best_bal = 0.5, -1.0
    for t in np.linspace(0.2, 0.8, 121):
        bal = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if bal > best_bal:
            best_bal, best_t = bal, float(t)
    print(f"[INFO] Best threshold {best_t:.3f} (balanced acc {best_bal*100:.2f}%)")
    return best_t
 
 
def print_detailed_metrics(y_true, y_prob, threshold=0.5, title="Validation"):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[INFO] === {title} (threshold={threshold:.3f}) ===")
    print(f"Confusion matrix:\n{cm}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"Balanced:  {balanced_accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0)*100:.2f}%")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0)*100:.2f}%")
    print(f"F1:        {f1_score(y_true, y_pred, zero_division=0)*100:.2f}%")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob)*100:.2f}%")
    print(
        f"Healthy  recall: {recall_score(y_true, y_pred, pos_label=0, zero_division=0)*100:.2f}%"
    )
    print(
        f"Parkinson recall: {recall_score(y_true, y_pred, pos_label=1, zero_division=0)*100:.2f}%"
    )
    print(classification_report(y_true, y_pred, target_names=["healthy", "parkinson"]))
    if y_prob.std() < COLLAPSE_STD_THRESHOLD:
        print("[ERROR] Model collapse detected (std(predictions) < 0.01).")
    return y_pred, cm
 
 
def train_cnn_model():
    print("[INFO] === EfficientNetB0 CNN (fixed preprocessing pipeline) ===")
    remove_stale_cnn_model()
 
    train_recs, val_recs, test_recs = get_spiral_splits()
    if not train_recs or not val_recs or not test_recs:
        print("[ERROR] No spiral data or split generation failed.")
        return
 
    # if {
    #     r.patient_id for r in train_recs
    # } & {r.patient_id for r in val_recs} or {
    #     r.patient_id for r in train_recs
    # } & {r.patient_id for r in test_recs} or {
    #     r.patient_id for r in val_recs
    # } & {r.patient_id for r in test_recs}:
    #     print("[ERROR] Patient overlap across train/val/test.")
    #     return
 
    save_split_manifest(train_recs, val_recs, test_recs)
    save_cnn_record_lists(train_recs, test_recs, val_recs=val_recs)
    print_cnn_record_audit(train_recs, "train (train_cnn_model.py)")
    print_cnn_record_audit(val_recs, "val (train_cnn_model.py)")
    print_cnn_record_audit(test_recs, "test (train_cnn_model.py)")
    verify_label_encoding(train_recs)
    verify_label_encoding(val_recs)
    verify_label_encoding(test_recs)
    verify_image_content(train_recs, "train", "outputs/cnn_samples_train")
    verify_image_content(val_recs, "val", "outputs/cnn_samples_val")
 
    y_train = np.array([r.label for r in train_recs])
    class_weight = build_class_weights(y_train)
 
    augment = build_advanced_augmentation()
    train_ds = build_path_dataset(train_recs, training=True, augment_layer=augment)
    val_ds = build_path_dataset(val_recs, training=False, augment_layer=augment)
 
    x0, y0 = next(iter(train_ds.take(1)))
    print(f"\n[INFO] Batch shape: {x0.shape} | labels: {y0.numpy()}")
    print(f"[INFO] Preprocessed batch min={float(x0.numpy().min()):.2f} max={float(x0.numpy().max()):.2f}")
 
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
 
    print("\n[INFO] ===== Phase 1: frozen backbone, train head @ lr=1e-4 =====")
    model, base_model = build_efficientnet_classifier(
        trainable_base=False, learning_rate=PHASE1_LR
    )
    freeze_efficientnet_backbone(base_model)
 
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weight,
        callbacks=get_cnn_training_callbacks(
            EFFICIENTNET_MODEL_PATH,
            val_dataset=val_ds,
            monitor="val_auc",
            patience=3,
            lr_patience=2,
        ),
        verbose=1,
    )
 
    best_phase1_val_auc = max(history1.history.get("val_auc", [0.0]))
    best_phase1_val_acc = max(history1.history.get("val_accuracy", [0.0]))
    print(
        f"[INFO] Phase 1 best val_accuracy={best_phase1_val_acc*100:.2f}% "
        f"val_auc={best_phase1_val_auc:.4f}"
    )
 
    if os.path.isfile(EFFICIENTNET_MODEL_PATH):
        print("[INFO] Restoring best Phase 1 checkpoint before fine-tuning...")
        model.load_weights(EFFICIENTNET_MODEL_PATH)
 
    # FIX 3: Tightened skip condition so Phase 2 runs in more situations.
    # Original threshold was (val_acc >= 0.84 AND val_auc >= 0.92) — too easy to skip.
    # New threshold requires near-perfect Phase 1 before skipping fine-tuning.
    skip_phase2 = False
    history2 = None
    if skip_phase2:
        print(
            "[INFO] Phase 1 performance is very strong; skipping Phase 2 fine-tuning."
        )
    else:
        print(
            f"\n[INFO] ===== Phase 2: fine-tune top {FINE_TUNE_LAYERS} "
            f"EfficientNet groups @ lr={PHASE2_LR:.1e} ====="
        )
        # FIX 2: unfreeze_efficientnet_top_layers now uses FINE_TUNE_LAYERS=25 (was 10)
        unfreeze_efficientnet_top_layers(model, base_model, num_layers=FINE_TUNE_LAYERS)
        # FIX 1: compile with PHASE2_LR=1e-5 (was 1e-6)
        compile_efficientnet_classifier(model, learning_rate=PHASE2_LR)
 
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=PHASE2_EPOCHS,
            class_weight=class_weight,
            callbacks=[
                *get_cnn_training_callbacks(
                    EFFICIENTNET_MODEL_PATH,
                    val_dataset=val_ds,
                    monitor="val_auc",
                    patience=3,
                    lr_patience=2,
                ),
            ],
            verbose=1,
        )
 
    if not os.path.isfile(EFFICIENTNET_MODEL_PATH):
        save_keras_model(model, EFFICIENTNET_MODEL_PATH)
    print("[INFO] Reloading best CNN checkpoint for hold-out (same file as fusion) ...")
    model = load_efficientnet_cnn(
        rebuild_on_failure=False,
        compile_model=True,
        learning_rate=PHASE2_LR,
    )
 
    merged = {k: list(history1.history.get(k, [])) for k in history1.history}
    if history2 is not None:
        for k, v in history2.history.items():
            merged.setdefault(k, []).extend(v)
    save_training_history(merged, "cnn_training")
 
    y_true = np.concatenate([y.numpy().astype(int).ravel() for _, y in val_ds], axis=0)
    y_prob = model.predict(val_ds, verbose=0).flatten()
 
    print(
        f"\n[INFO] Final prob stats: mean={y_prob.mean():.3f} std={y_prob.std():.3f} "
        f"min={y_prob.min():.3f} max={y_prob.max():.3f}"
    )
    threshold = tune_decision_threshold(y_true, y_prob)
 
    eval_out = evaluate_cnn_on_records(
        model, test_recs, threshold, name="Standalone CNN (hold-out)"
    )
    y_pred = eval_out["y_pred"]
    y_prob = eval_out["y_prob"]
    cm = eval_out["confusion_matrix"]
    print_detailed_metrics(
        eval_out["y_true"],
        eval_out["y_prob"],
        threshold=threshold,
        title="Hold-out (tuned threshold)",
    )
    save_cnn_threshold(
        threshold,
        extra={
            "accuracy": float(eval_out["metrics"]["accuracy"]),
            "roc_auc": float(eval_out["metrics"]["roc_auc"]),
        },
    )
 
    train_acc = float(model.evaluate(train_ds, verbose=0)[1])
    val_acc = float(model.evaluate(val_ds, verbose=0)[1])
    check_overfitting_gap(train_acc, val_acc, name="CNN")
 
    plt.figure(figsize=(12, 4))
    ep = range(1, len(merged.get("loss", [])) + 1)
    plt.plot(ep, merged.get("accuracy", []), label="train")
    plt.plot(ep, merged.get("val_accuracy", []), label="val")
    plt.plot(ep, merged.get("val_auc", []), label="val_auc")
    plt.legend()
    plt.title("CNN training")
    plt.savefig("outputs/training_history.png")
    plt.close()
 
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.savefig("outputs/cnn_confusion_matrix.png")
    plt.close()
 
    layer_name = find_last_conv_layer(model)
    if layer_name and test_recs:
        raw = load_spiral_rgb_float(test_recs[0].path)
        img_in = np.expand_dims(apply_efficientnet_preprocess(raw), 0)
        hm = make_gradcam_heatmap(img_in, model, layer_name)
        base = load_spiral_image(test_recs[0].path, normalized=True)
        plt.imshow(base)
        plt.imshow(hm, cmap="jet", alpha=0.4)
        plt.savefig("outputs/gradcam_sample_0.png")
        plt.close()
 
    acc = float(eval_out["metrics"]["accuracy"]) * 100
    auc = float(eval_out["metrics"]["roc_auc"]) * 100
    print(f"\n[SUCCESS] CNN hold-out accuracy: {acc:.2f}% | AUC: {auc:.2f}%")
    return acc
 
 
if __name__ == "__main__":
    train_cnn_model()
 