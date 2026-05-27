"""
True late fusion: train only a fusion head on pretrained voice + CNN outputs.
Standalone voice/CNN metrics are computed independently (not on pseudo-pairs).
 
FIXES APPLIED:
  1. FUSION_EPOCHS increased from 35 to 50 — more room to converge.
  2. get_standard_callbacks patience increased from 6 to 12 — was stopping too early.
  3. Learning rate is now inherited from model_utils.build_late_fusion_model (1e-4),
     which was corrected from 5e-5. No change needed here — model_utils fix covers it.
"""
import os
 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import traceback
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight
 
from model_utils import (
    EFFICIENTNET_MODEL_PATH,
    FUSION_MODEL_PATH,
    enable_unsafe_deserialization,
    load_efficientnet_cnn,
    build_late_fusion_model,
    extract_cnn_cache,
    get_cnn_embedding_dim,
    load_cnn_threshold,
    print_cnn_model_info,
    evaluate_cnn_on_records,
    apply_efficientnet_preprocess,
    load_spiral_rgb_float,
    save_keras_model,
    save_training_history,
    get_standard_callbacks,
)
from patient_data import (
    CNN_TEST_RECORDS_JSON,
    CNN_TRAIN_RECORDS_JSON,
    CNN_VAL_RECORDS_JSON,
    resolve_voice_feature_columns,
    print_voice_diagnostics,
    pair_late_fusion_features,
    warn_suspicious_accuracy,
    load_spiral_image,
    split_records_train_val,
    check_overfitting_gap,
    load_cnn_record_lists_with_val,
    print_cnn_record_audit,
)
 
enable_unsafe_deserialization()
 
selector_path = os.path.join("models", "feature_selector.pkl")
xgb_model_path = os.path.join("models", "voice_xgb_model.pkl")
voice_threshold_path = os.path.join("models", "voice_decision_threshold.json")
 
# FIX 1: Increased from 35 to 50 — gives the fusion head more room to converge.
FUSION_EPOCHS = 50
BATCH_SIZE = 16
CNN_MIN_ACCURACY = 0.80
FUSION_VAL_FRACTION = 0.15
 
 
def find_last_conv_layer_name(model):
    for layer in model.layers:
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None
 
 
def make_gradcam_heatmap(img_array, model, layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv, preds = grad_model(img_array)
        channel = preds[:, 0]
    grads = tape.gradient(channel, conv)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv[0] * pooled, axis=-1)
    return (tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)).numpy()
 
 
def load_voice_threshold(default: float = 0.5) -> float:
    if os.path.isfile(voice_threshold_path):
        with open(voice_threshold_path, encoding="utf-8") as f:
            return float(json.load(f).get("threshold", default))
    return default
 
 
def evaluate_standalone_voice(xgb_model, voice_test_df, feature_cols, threshold: float):
    """Patient-level voice hold-out — unchanged from train_dl_model.py."""
    X = voice_test_df[feature_cols].values
    y = voice_test_df["_label"].values.astype(int)
    prob = xgb_model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, prob)
    print("\n[INFO] === Standalone Voice (patient hold-out, NOT pseudo-paired) ===")
    print(f"[SUCCESS] Accuracy: {acc*100:.2f}% | ROC-AUC: {auc*100:.2f}%")
    print(f"[INFO] Threshold: {threshold:.4f}")
    print(f"[INFO] Confusion matrix:\n{confusion_matrix(y, pred)}")
    warn_suspicious_accuracy("Standalone Voice", acc)
    return {"accuracy": acc, "roc_auc": auc, "y_true": y, "y_prob": prob, "y_pred": pred}
 
 
def evaluate_standalone_cnn(cnn_model, test_recs, threshold: float):
    """Image-level CNN hold-out on outputs/cnn_test_records.json only."""
    out = evaluate_cnn_on_records(
        cnn_model,
        test_recs,
        threshold,
        name=f"Standalone CNN ({CNN_TEST_RECORDS_JSON})",
    )
    warn_suspicious_accuracy("Standalone CNN", out["metrics"]["accuracy"])
    return out
 
 
def train_and_evaluate_fusion():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
 
    print("[INFO] === TRUE Late Fusion (pretrained voice + CNN features only) ===")
 
    voice_train_path = os.path.join("models", "voice_train_features.pkl")
    voice_test_path = os.path.join("models", "voice_test_features.pkl")
    if not all(
        os.path.exists(p)
        for p in [selector_path, xgb_model_path, voice_train_path, voice_test_path]
    ):
        print("[ERROR] Train voice first: python train_dl_model.py")
        return
 
    voice_train_df = joblib.load(voice_train_path)
    voice_test_df = joblib.load(voice_test_path)
    patient_col = next(
        (c for c in voice_train_df.columns if str(c).lower() in ("id", "patient_id")),
        "id",
    )
    feature_cols = resolve_voice_feature_columns(
        voice_train_df, patient_col, target_col="_label"
    )
    print_voice_diagnostics(voice_train_df, voice_test_df, patient_col)
 
    xgb_model = joblib.load(xgb_model_path)
    voice_threshold = load_voice_threshold()
    cnn_threshold = load_cnn_threshold()
 
    try:
        train_recs, val_recs, test_recs = load_cnn_record_lists_with_val()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return
 
    print_cnn_record_audit(train_recs, "train (train_fusion_model.py)")
    if val_recs:
        print_cnn_record_audit(val_recs, "val (train_fusion_model.py)")
    print_cnn_record_audit(test_recs, "test (train_fusion_model.py)")
 
    if {
        r.patient_id for r in train_recs
    } & {r.patient_id for r in val_recs} or {
        r.patient_id for r in train_recs
    } & {r.patient_id for r in test_recs} or {
        r.patient_id for r in val_recs
    } & {r.patient_id for r in test_recs}:
        print("[ERROR] Patient overlap in saved CNN records — aborting.")
        return
 
    print_cnn_model_info(EFFICIENTNET_MODEL_PATH)
    try:
        cnn_model = load_efficientnet_cnn(rebuild_on_failure=False)
    except Exception as e:
        print(f"[ERROR] Load CNN: {e}")
        return
 
    # --- Standalone CNN FIRST (exact cnn_test_records.json only, before pairing) ---
    print(f"\n[INFO] Standalone CNN eval uses ONLY: {CNN_TEST_RECORDS_JSON}")
    cnn_standalone = evaluate_standalone_cnn(cnn_model, test_recs, cnn_threshold)
    if cnn_standalone["metrics"]["accuracy"] < CNN_MIN_ACCURACY:
        print(
            f"[ERROR] Standalone CNN {cnn_standalone['metrics']['accuracy']*100:.1f}% "
            f"< {CNN_MIN_ACCURACY*100:.0f}% — fix CNN before fusion."
        )
        return
 
    if val_recs:
        print(f"\n[INFO] Standalone CNN monitoring uses ONLY: {CNN_VAL_RECORDS_JSON}")
        evaluate_cnn_on_records(
            cnn_model,
            val_recs,
            cnn_threshold,
            name=f"Standalone CNN ({CNN_VAL_RECORDS_JSON})",
        )
 
    voice_standalone = evaluate_standalone_voice(
        xgb_model, voice_test_df, feature_cols, voice_threshold
    )
 
    # --- Precompute CNN cache on exact saved train/val/test record lists ---
    print("\n[INFO] Extracting CNN embeddings/probabilities (frozen backbone) ...")
    all_recs = train_recs + val_recs + test_recs
    cnn_cache = extract_cnn_cache(cnn_model, all_recs)
    cnn_embed_dim = get_cnn_embedding_dim(cnn_model)
 
    fit_recs, fusion_val_recs = split_records_train_val(train_recs, val_fraction=FUSION_VAL_FRACTION)
    print(
        f"[INFO] Fusion train spirals: {len(fit_recs)} | internal val: {len(fusion_val_recs)} | "
        f"test spirals: {len(test_recs)}"
    )
 
    print("\n[INFO] Building late-fusion tensors (pseudo-pairing for alignment only) ...")
    X_v_tr, v_prob_tr, cnn_emb_tr, cnn_prob_tr, y_tr = pair_late_fusion_features(
        fit_recs, voice_train_df, feature_cols, patient_col, cnn_cache, xgb_model
    )
    X_v_va, v_prob_va, cnn_emb_va, cnn_prob_va, y_va = pair_late_fusion_features(
        fusion_val_recs, voice_train_df, feature_cols, patient_col, cnn_cache, xgb_model
    )
    X_v_te, v_prob_te, cnn_emb_te, cnn_prob_te, y_te = pair_late_fusion_features(
        test_recs, voice_test_df, feature_cols, patient_col, cnn_cache, xgb_model
    )
 
    if len(y_tr) == 0 or len(y_te) == 0:
        print("[ERROR] Empty fusion dataset.")
        return
 
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}
 
    fusion_model, attention_model = build_late_fusion_model(
        voice_feature_dim=X_v_tr.shape[1],
        cnn_embed_dim=cnn_embed_dim,
    )
 
    val_data = (
        ([X_v_va, v_prob_va, cnn_emb_va, cnn_prob_va], y_va) if len(y_va) else None
    )
 
    print("\n[INFO] Training late fusion head only (CNN frozen, not retrained) ...")
    try:
        # FIX 2: patience increased from 6 to 12 — original value stopped training
        # before the fusion head had time to converge, especially with a small dataset.
        history = fusion_model.fit(
            [X_v_tr, v_prob_tr, cnn_emb_tr, cnn_prob_tr],
            y_tr,
            validation_data=val_data,
            epochs=FUSION_EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=get_standard_callbacks(FUSION_MODEL_PATH, patience=12),
            verbose=0,
        )
        save_training_history(history, "fusion_training")
    except Exception:
        traceback.print_exc()
        return
 
    save_keras_model(fusion_model, FUSION_MODEL_PATH)
 
    train_acc = max(history.history.get("accuracy", [0]))
    val_acc = max(history.history.get("val_accuracy", [0]))
    check_overfitting_gap(train_acc, val_acc, name="Late fusion (internal val)")
 
    # --- Fusion evaluation on paired test samples ---
    y_prob_fusion = fusion_model.predict(
        [X_v_te, v_prob_te, cnn_emb_te, cnn_prob_te], verbose=0
    ).flatten()
    y_pred_fusion = (y_prob_fusion >= 0.5).astype(int)
    fusion_acc = accuracy_score(y_te, y_pred_fusion)
    fusion_auc = roc_auc_score(y_te, y_prob_fusion)
 
    attn_te = attention_model.predict(
        [X_v_te, v_prob_te, cnn_emb_te, cnn_prob_te], verbose=0
    )
 
    print("\n[INFO] === Fusion test (pseudo-paired multimodal samples) ===")
    print(f"[SUCCESS] Fusion Accuracy: {fusion_acc*100:.2f}% | ROC-AUC: {fusion_auc*100:.2f}%")
    print(f"[INFO] Confusion matrix:\n{confusion_matrix(y_te, y_pred_fusion)}")
    print(classification_report(y_te, y_pred_fusion, target_names=["healthy", "parkinson"]))
 
    # Optional: branch metrics on SAME paired test (for comparison only, not standalone)
    cnn_prob_paired = cnn_prob_te.flatten()
    v_prob_paired = v_prob_te.flatten()
    print("\n[INFO] --- On paired test only (comparison, NOT standalone CNN) ---")
    print(
        f"[INFO] CNN prob from cache on pairs: "
        f"{accuracy_score(y_te, (cnn_prob_paired>=cnn_threshold).astype(int))*100:.2f}%"
    )
    print(
        f"[INFO] Voice prob on pairs: "
        f"{accuracy_score(y_te, (v_prob_paired>=voice_threshold).astype(int))*100:.2f}%"
    )
 
    # --- Final report ---
    print("\n[INFO] " + "=" * 58)
    print("[INFO] FINAL REPORT (standalone vs late fusion)")
    print("[INFO] " + "=" * 58)
    print(
        f"[SUCCESS] Standalone Voice Accuracy:  "
        f"{voice_standalone['accuracy']*100:.2f}%"
    )
    print(
        f"[SUCCESS] Standalone CNN Accuracy:    "
        f"{cnn_standalone['metrics']['accuracy']*100:.2f}%"
    )
    print(f"[SUCCESS] Fusion Accuracy:            {fusion_acc*100:.2f}%")
    print("[INFO] " + "=" * 58)
 
    payload = {
        "standalone_voice_accuracy": voice_standalone["accuracy"],
        "standalone_cnn_accuracy": cnn_standalone["metrics"]["accuracy"],
        "fusion_accuracy": fusion_acc,
        "fusion_roc_auc": fusion_auc,
    }
    with open(os.path.join("outputs", "fusion_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
 
    # --- Plots ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.legend()
    plt.title("Late fusion accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.legend()
    plt.title("Late fusion loss")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "fusion_training_history.png"))
    plt.close()
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (title, pred) in zip(
        axes,
        [
            ("Voice (paired)", (v_prob_paired >= voice_threshold).astype(int)),
            ("CNN (paired cache)", (cnn_prob_paired >= cnn_threshold).astype(int)),
            ("Fusion", y_pred_fusion),
        ],
    ):
        cm = confusion_matrix(y_te, pred)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "confusion_matrices.png"))
    plt.close()
 
    plt.figure(figsize=(7, 6))
    for name, prob in [
        ("Voice (paired)", v_prob_paired),
        ("CNN (cached)", cnn_prob_paired),
        ("Fusion", y_prob_fusion),
    ]:
        fpr, tpr, _ = roc_curve(y_te, prob)
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y_te, prob):.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    plt.savefig(os.path.join("outputs", "roc_curves.png"))
    plt.close()
 
    # Attention plot
    voice_w, spiral_w = attn_te[:, 0], attn_te[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(voice_w, bins=20, alpha=0.8, label="Voice", color="#ff9999")
    axes[0].hist(spiral_w, bins=20, alpha=0.8, label="Spiral", color="#66b3ff")
    axes[0].legend()
    axes[1].pie(
        [voice_w.mean(), spiral_w.mean()],
        labels=["Voice", "Spiral"],
        autopct="%1.1f%%",
        colors=["#ff9999", "#66b3ff"],
    )
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "modality_attention_weights.png"))
    plt.close()
 
    # Grad-CAM on frozen CNN (optional interpretability)
    try:
        layer = find_last_conv_layer_name(cnn_model)
        if layer and test_recs:
            sample = np.expand_dims(
                apply_efficientnet_preprocess(load_spiral_rgb_float(test_recs[0].path)),
                0,
            )
            hm = make_gradcam_heatmap(sample, cnn_model, layer)
            base = (load_spiral_image(test_recs[0].path, normalized=True) * 255).astype(
                np.uint8
            )
            hm = cv2.resize(hm, (base.shape[1], base.shape[0]))
            hm = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join("outputs", "gradcam_spiral.png"), hm * 0.4 + base)
    except Exception as ex:
        print(f"[WARNING] Grad-CAM skipped: {ex}")
 
    return fusion_acc
 
 
if __name__ == "__main__":
    train_and_evaluate_fusion()
 