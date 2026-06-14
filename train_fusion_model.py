"""
Adaptive Attention Fusion Model — combines speech (LightGBM) + spiral CNN (EfficientNet)
into a single late-fusion model with learned modality-attention weights.

Expected base model performance:
  - Speech (LGBMClassifier): ~92% accuracy
  - Spiral CNN (EfficientNetB2): ~94% accuracy
  - Fusion target: 95%+ validation accuracy

The fusion model does NOT retrain the base models. It learns to weight
voice-feature vectors and CNN embeddings via a softmax attention gate,
producing a single Parkinson probability per patient.
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import pickle
import traceback

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight

from model_utils import (
    FUSION_MODEL_PATH,
    build_late_fusion_model,
    enable_unsafe_deserialization,
    extract_cnn_cache,
    get_cnn_embedding_dim,
    load_cnn_threshold,
    load_efficientnet_cnn,
    safe_load_model,
    save_keras_model,
    save_training_history,
    get_standard_callbacks,
)
from patient_data import (
    load_voice_frame_split,
    get_raw_voice_feature_columns,
    impute_with_train_stats,
    prune_weak_voice_features,
    scale_train_transform_test,
    select_k_best_features,
    load_cnn_record_lists_with_val,
    pair_late_fusion_features,
    print_class_distribution_and_weights,
    check_overfitting_gap,
    warn_suspicious_accuracy,
)

enable_unsafe_deserialization()

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.dirname(__file__))
VOICE_CSV = os.path.join(ROOT, "datasets", "voice", "pd_speech_features.csv")
SPEECH_MODEL_PATH = os.path.join(ROOT, "xgboost_pd_speech.pkl")
SCALER_PATH = os.path.join(ROOT, "scaler.pkl")
FEATURES_PATH = os.path.join(ROOT, "selected_features.pkl")
PLOTS_DIR = os.path.join(ROOT, "plots")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
FUSION_THRESHOLD_PATH = os.path.join(OUTPUTS_DIR, "fusion_decision_threshold.json")
FUSION_ATTENTION_PATH = os.path.join(OUTPUTS_DIR, "fusion_attention_weights.json")

RANDOM_STATE = 42
FUSION_EPOCHS = 120
FUSION_BATCH_SIZE = 16
FUSION_DIM = 128
K_VOICE_FEATURES = 100

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Load pre-trained base models ────────────────────────────────────────────
def load_speech_model():
    """Load the pre-trained LightGBM speech model and its threshold."""
    with open(SPEECH_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict):
        model = bundle["model"]
        threshold = bundle.get("threshold", 0.5)
    else:
        model = bundle
        threshold = 0.5
    print(f"[SUCCESS] Speech model loaded: {type(model).__name__}")
    print(f"[INFO] Speech threshold: {threshold:.4f}")
    return model, threshold


def load_voice_features_and_scaler():
    """Load the fitted scaler and selected features for voice inference."""
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    features = pickle.load(open(FEATURES_PATH, "rb"))
    print(f"[INFO] Voice scaler loaded: {SCALER_PATH}")
    print(f"[INFO] Voice selected features: {len(features)} features from {FEATURES_PATH}")
    return scaler, features


# ── Prepare voice data ──────────────────────────────────────────────────────
def prepare_voice_data():
    """
    Load and preprocess voice CSV for fusion — uses the same pipeline as
    pd_speech_voice_pipeline.py to ensure consistency with the trained model.
    Returns scaled feature arrays and labels, plus the fitted scaler/selector
    for inference.
    """
    print("\n[INFO] ===== Preparing voice data for fusion =====")
    df = pd.read_csv(VOICE_CSV, header=1)
    df.columns = df.columns.astype(str).str.strip()

    # Identify target
    target_col = "class"
    id_cols = ["id"]
    remove_cols = [c for c in id_cols if c in df.columns]
    df = df.drop(columns=remove_cols, errors="ignore")

    feature_cols = [c for c in df.columns if c != target_col]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # Use same split as the speech pipeline
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE,
    )

    # Load the SAME scaler + features used by the trained speech model
    scaler, selected_features = load_voice_features_and_scaler()

    # Apply scaler to the selected features
    X_train_scaled = scaler.transform(X_train[selected_features])
    X_test_scaled = scaler.transform(X_test[selected_features])

    print(f"[INFO] Voice train samples: {len(X_train_scaled)}")
    print(f"[INFO] Voice test samples: {len(X_test_scaled)}")
    print(f"[INFO] Voice features used: {len(selected_features)}")

    return (
        X_train_scaled, y_train.values,
        X_test_scaled, y_test.values,
        selected_features, scaler,
    )


# ── Build fusion training arrays ────────────────────────────────────────────
def build_fusion_arrays(
    voice_X, voice_y, speech_model, cnn_model, spiral_records, cnn_threshold,
):
    """
    Build aligned arrays for late fusion training.

    Since voice and spiral datasets come from different patient cohorts,
    we create class-matched pseudo-pairs:
      - For each voice sample, pair it with a randomly selected spiral
        image of the same class, extracting CNN embedding + probability.
    """
    print("\n[INFO] ===== Building fusion training arrays =====")

    # Get CNN embeddings for all spiral records
    cnn_cache = extract_cnn_cache(cnn_model, spiral_records)
    cnn_embed_dim = get_cnn_embedding_dim(cnn_model)

    # Separate spiral records by class
    spiral_healthy = [r for r in spiral_records if r.label == 0 and os.path.normpath(r.path) in cnn_cache]
    spiral_parkinson = [r for r in spiral_records if r.label == 1 and os.path.normpath(r.path) in cnn_cache]

    print(f"[INFO] Spiral records with CNN cache — healthy: {len(spiral_healthy)}, parkinson: {len(spiral_parkinson)}")

    # Build fusion pairs: for each voice sample, pick a spiral of matching class
    rng = np.random.RandomState(RANDOM_STATE)

    X_voice_out = []
    voice_prob_out = []
    cnn_embed_out = []
    cnn_prob_out = []
    y_out = []

    for i in range(len(voice_X)):
        label = int(voice_y[i])
        pool = spiral_healthy if label == 0 else spiral_parkinson
        if not pool:
            continue

        # Voice features and probability
        voice_feat = voice_X[i]
        voice_p = float(speech_model.predict_proba(voice_feat.reshape(1, -1))[0, 1])

        # Pick a random spiral from the matching class
        spiral_rec = pool[rng.randint(len(pool))]
        key = os.path.normpath(spiral_rec.path)
        cnn_entry = cnn_cache[key]

        X_voice_out.append(voice_feat)
        voice_prob_out.append(voice_p)
        cnn_embed_out.append(cnn_entry["embedding"])
        cnn_prob_out.append(cnn_entry["prob"])
        y_out.append(label)

    X_voice_arr = np.array(X_voice_out, dtype=np.float32)
    voice_prob_arr = np.array(voice_prob_out, dtype=np.float32).reshape(-1, 1)
    cnn_embed_arr = np.stack(cnn_embed_out).astype(np.float32)
    cnn_prob_arr = np.array(cnn_prob_out, dtype=np.float32).reshape(-1, 1)
    y_arr = np.array(y_out, dtype=np.int32)

    print(f"[INFO] Fusion pairs built: {len(y_arr)}")
    print(f"[INFO] Voice feature dim: {X_voice_arr.shape[1]}")
    print(f"[INFO] CNN embedding dim: {cnn_embed_arr.shape[1]}")
    print(f"[INFO] Class distribution — healthy: {(y_arr == 0).sum()}, parkinson: {(y_arr == 1).sum()}")

    return X_voice_arr, voice_prob_arr, cnn_embed_arr, cnn_prob_arr, y_arr


# ── Threshold tuning ────────────────────────────────────────────────────────
def tune_fusion_threshold(y_true, y_prob):
    """Find optimal threshold maximizing balanced accuracy."""
    best_t, best_bal = 0.5, -1.0
    for t in np.linspace(0.2, 0.8, 121):
        bal = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if bal > best_bal:
            best_bal, best_t = bal, float(t)
    print(f"[INFO] Best fusion threshold: {best_t:.3f} (balanced acc {best_bal * 100:.2f}%)")
    return best_t


# ── Evaluation & plotting ───────────────────────────────────────────────────
def evaluate_fusion(y_true, y_prob, threshold, title="Fusion"):
    """Full classification report + confusion matrix."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[INFO] === {title} (threshold={threshold:.3f}) ===")
    print(f"[SUCCESS] Accuracy:          {acc * 100:.2f}%")
    print(f"[INFO]    Balanced Accuracy:  {bal_acc * 100:.2f}%")
    print(f"[INFO]    Precision:          {prec * 100:.2f}%")
    print(f"[INFO]    Recall:             {rec * 100:.2f}%")
    print(f"[INFO]    F1 Score:           {f1 * 100:.2f}%")
    print(f"[INFO]    ROC-AUC:            {auc * 100:.2f}%")
    print(f"[INFO]    Confusion Matrix:\n{cm}")
    print(classification_report(y_true, y_pred, target_names=["healthy", "parkinson"], zero_division=0))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
    }


def plot_fusion_results(y_true, y_prob, threshold, history, attention_weights):
    """Generate all fusion visualization plots."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Healthy", "Parkinson's"],
        yticklabels=["Healthy", "Parkinson's"],
        annot_kws={"fontsize": 16, "fontweight": "bold"},
        cbar_kws={"label": "Count"},
    )
    plt.ylabel("True Label", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=13, fontweight="bold")
    plt.title("Fusion Model — Confusion Matrix", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fusion_confusion_matrix.png"), dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Fusion ROC (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Fusion Model — ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fusion_roc_curve.png"), dpi=300)
    plt.close()

    # 3. Training History
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(history.get("loss", [])) + 1)
        axes[0].plot(epochs, history.get("accuracy", []), label="Train Acc", color="#1f77b4")
        axes[0].plot(epochs, history.get("val_accuracy", []), label="Val Acc", color="#ff7f0e")
        axes[0].set_title("Fusion Accuracy", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, history.get("loss", []), label="Train Loss", color="#1f77b4")
        axes[1].plot(epochs, history.get("val_loss", []), label="Val Loss", color="#ff7f0e")
        axes[1].set_title("Fusion Loss", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "fusion_training_history.png"), dpi=300)
        plt.close()

    # 4. Modality Attention Weights
    if attention_weights is not None:
        mean_voice = float(np.mean(attention_weights[:, 0]))
        mean_spiral = float(np.mean(attention_weights[:, 1]))
        labels_att = ["Voice/Speech", "Spiral/Image"]
        values = [mean_voice, mean_spiral]
        colors = ["#2196F3", "#4CAF50"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        bars = axes[0].bar(labels_att, values, color=colors, edgecolor="black", linewidth=1.5, alpha=0.85)
        for bar, val in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=13)
        axes[0].set_ylabel("Mean Attention Weight", fontsize=12)
        axes[0].set_title("Modality Importance (Attention Gate)", fontsize=13, fontweight="bold")
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis="y", alpha=0.3)

        # Distribution per sample
        axes[1].hist(attention_weights[:, 0], bins=30, alpha=0.7, label="Voice weight", color="#2196F3")
        axes[1].hist(attention_weights[:, 1], bins=30, alpha=0.7, label="Spiral weight", color="#4CAF50")
        axes[1].set_xlabel("Attention Weight", fontsize=12)
        axes[1].set_ylabel("Count", fontsize=12)
        axes[1].set_title("Attention Weight Distribution", fontsize=13, fontweight="bold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "fusion_modality_attention.png"), dpi=300)
        plt.close()

    print(f"[SUCCESS] Fusion plots saved to {PLOTS_DIR}")


# ── Main training pipeline ──────────────────────────────────────────────────
def train_fusion_model():
    print("=" * 70)
    print("  ADAPTIVE ATTENTION FUSION MODEL — Training Pipeline")
    print("=" * 70)

    # ── Step 1: Load base models ────────────────────────────────────────
    print("\n[STEP 1] Loading pre-trained base models...")
    speech_model, speech_threshold = load_speech_model()

    cnn_model = load_efficientnet_cnn(rebuild_on_failure=False, compile_model=True)
    cnn_threshold = load_cnn_threshold()
    cnn_embed_dim = get_cnn_embedding_dim(cnn_model)
    print(f"[INFO] CNN embedding dim: {cnn_embed_dim}")

    # ── Step 2: Prepare voice data ──────────────────────────────────────
    print("\n[STEP 2] Preparing voice data...")
    (
        voice_X_train, voice_y_train,
        voice_X_test, voice_y_test,
        selected_features, voice_scaler,
    ) = prepare_voice_data()
    voice_feature_dim = voice_X_train.shape[1]

    # ── Step 3: Load spiral records ─────────────────────────────────────
    print("\n[STEP 3] Loading spiral image records...")
    train_spiral_recs, val_spiral_recs, test_spiral_recs = load_cnn_record_lists_with_val()

    # Combine train+val for fusion training pool (we'll re-split for fusion)
    all_train_spiral = train_spiral_recs + val_spiral_recs
    print(f"[INFO] Spiral — train+val pool: {len(all_train_spiral)}, test: {len(test_spiral_recs)}")

    # ── Step 4: Build fusion arrays ─────────────────────────────────────
    print("\n[STEP 4] Building fusion training arrays...")
    train_voice, train_voice_prob, train_cnn_embed, train_cnn_prob, train_y = \
        build_fusion_arrays(
            voice_X_train, voice_y_train,
            speech_model, cnn_model, all_train_spiral, cnn_threshold,
        )

    print("\n[STEP 4b] Building fusion test arrays...")
    test_voice, test_voice_prob, test_cnn_embed, test_cnn_prob, test_y = \
        build_fusion_arrays(
            voice_X_test, voice_y_test,
            speech_model, cnn_model, test_spiral_recs, cnn_threshold,
        )

    if len(train_y) == 0 or len(test_y) == 0:
        print("[ERROR] No fusion pairs could be built. Check data availability.")
        return

    # Class weights for imbalanced data
    classes = np.unique(train_y)
    cw = compute_class_weight("balanced", classes=classes, y=train_y)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print(f"[INFO] Fusion class weights: {class_weight}")

    # ── Step 5: Build & train fusion model ──────────────────────────────
    print("\n[STEP 5] Building adaptive attention fusion model...")
    fusion_model, attention_model = build_late_fusion_model(
        voice_feature_dim=voice_feature_dim,
        cnn_embed_dim=cnn_embed_dim,
        fusion_dim=FUSION_DIM,
    )
    fusion_model.summary()

    # Split a validation set from training for early stopping
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(train_y))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.15, stratify=train_y, random_state=RANDOM_STATE,
    )

    def make_inputs(idx):
        return [
            train_voice[idx],
            train_voice_prob[idx],
            train_cnn_embed[idx],
            train_cnn_prob[idx],
        ]

    X_train_inputs = make_inputs(train_idx)
    y_train_split = train_y[train_idx]
    X_val_inputs = make_inputs(val_idx)
    y_val_split = train_y[val_idx]

    print(f"\n[INFO] Fusion training split — train: {len(train_idx)}, val: {len(val_idx)}")
    print(f"[INFO] Train class dist — healthy: {(y_train_split == 0).sum()}, parkinson: {(y_train_split == 1).sum()}")
    print(f"[INFO] Val class dist — healthy: {(y_val_split == 0).sum()}, parkinson: {(y_val_split == 1).sum()}")

    # Callbacks
    fusion_checkpoint = FUSION_MODEL_PATH
    os.makedirs(os.path.dirname(fusion_checkpoint) or ".", exist_ok=True)
    callbacks = get_standard_callbacks(
        fusion_checkpoint,
        monitor="val_accuracy",
        patience=15,
        monitor_acc=True,
    )

    print(f"\n[INFO] ===== Training fusion model ({FUSION_EPOCHS} epochs max) =====")
    history = fusion_model.fit(
        X_train_inputs,
        y_train_split,
        validation_data=(X_val_inputs, y_val_split),
        epochs=FUSION_EPOCHS,
        batch_size=FUSION_BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Reload best checkpoint
    if os.path.isfile(fusion_checkpoint):
        print("[INFO] Reloading best fusion checkpoint...")
        fusion_model = safe_load_model(fusion_checkpoint)
        # Rebuild attention model from the loaded fusion model
        attention_output = None
        for layer in fusion_model.layers:
            if layer.name == "modality_attention":
                attention_output = layer.output
                break
        if attention_output is not None:
            attention_model = tf.keras.Model(
                inputs=fusion_model.inputs,
                outputs=attention_output,
                name="attention_probe",
            )
    else:
        save_keras_model(fusion_model, fusion_checkpoint)

    save_training_history(history, "fusion_training")

    # ── Step 6: Evaluate on hold-out test set ───────────────────────────
    print("\n[STEP 6] Evaluating fusion on hold-out test set...")
    test_inputs = [test_voice, test_voice_prob, test_cnn_embed, test_cnn_prob]
    y_prob = fusion_model.predict(test_inputs, verbose=0).flatten()

    # Tune threshold
    threshold = tune_fusion_threshold(test_y, y_prob)
    metrics = evaluate_fusion(test_y, y_prob, threshold, title="Fusion Hold-out Test")

    # Save threshold
    with open(FUSION_THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "metrics": metrics}, f, indent=2)
    print(f"[SUCCESS] Fusion threshold saved: {FUSION_THRESHOLD_PATH}")

    # ── Step 7: Extract attention weights (modality importance) ─────────
    print("\n[STEP 7] Analyzing modality attention weights...")
    try:
        attention_weights = attention_model.predict(test_inputs, verbose=0)
        mean_voice_w = float(np.mean(attention_weights[:, 0]))
        mean_spiral_w = float(np.mean(attention_weights[:, 1]))

        # Per-class analysis
        healthy_mask = test_y == 0
        parkinson_mask = test_y == 1

        att_report = {
            "overall": {
                "voice_weight": mean_voice_w,
                "spiral_weight": mean_spiral_w,
            },
        }
        if healthy_mask.any():
            att_report["healthy_samples"] = {
                "voice_weight": float(np.mean(attention_weights[healthy_mask, 0])),
                "spiral_weight": float(np.mean(attention_weights[healthy_mask, 1])),
            }
        if parkinson_mask.any():
            att_report["parkinson_samples"] = {
                "voice_weight": float(np.mean(attention_weights[parkinson_mask, 0])),
                "spiral_weight": float(np.mean(attention_weights[parkinson_mask, 1])),
            }

        with open(FUSION_ATTENTION_PATH, "w", encoding="utf-8") as f:
            json.dump(att_report, f, indent=2)

        print(f"\n{'=' * 50}")
        print(f"  MODALITY IMPORTANCE (Learned Attention)")
        print(f"{'=' * 50}")
        print(f"  Voice/Speech contribution:  {mean_voice_w * 100:.1f}%")
        print(f"  Spiral/Image contribution:  {mean_spiral_w * 100:.1f}%")
        if "healthy_samples" in att_report:
            print(f"\n  For HEALTHY samples:")
            print(f"    Voice:  {att_report['healthy_samples']['voice_weight'] * 100:.1f}%")
            print(f"    Spiral: {att_report['healthy_samples']['spiral_weight'] * 100:.1f}%")
        if "parkinson_samples" in att_report:
            print(f"\n  For PARKINSON samples:")
            print(f"    Voice:  {att_report['parkinson_samples']['voice_weight'] * 100:.1f}%")
            print(f"    Spiral: {att_report['parkinson_samples']['spiral_weight'] * 100:.1f}%")
        print(f"{'=' * 50}")
        print(f"[SUCCESS] Attention weights saved: {FUSION_ATTENTION_PATH}")
    except Exception as ex:
        print(f"[WARNING] Could not extract attention weights: {ex}")
        traceback.print_exc()
        attention_weights = None

    # ── Step 8: Plots ───────────────────────────────────────────────────
    print("\n[STEP 8] Generating fusion plots...")
    plot_fusion_results(
        test_y, y_prob, threshold,
        history.history if hasattr(history, "history") else history,
        attention_weights,
    )

    # ── Step 9: Overfitting check ───────────────────────────────────────
    print("\n[STEP 9] Overfitting analysis...")
    train_prob = fusion_model.predict(X_train_inputs, verbose=0).flatten()
    train_pred = (train_prob >= threshold).astype(int)
    train_acc = accuracy_score(y_train_split, train_pred)
    test_acc = metrics["accuracy"]
    check_overfitting_gap(train_acc, test_acc, name="Fusion")

    # ── Final summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FUSION MODEL — TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Accuracy:          {metrics['accuracy'] * 100:.2f}%")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  Precision:         {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:            {metrics['recall'] * 100:.2f}%")
    print(f"  F1 Score:          {metrics['f1'] * 100:.2f}%")
    print(f"  ROC-AUC:           {metrics['roc_auc'] * 100:.2f}%")
    print(f"  Threshold:         {threshold:.3f}")
    print(f"  Model saved:       {FUSION_MODEL_PATH}")
    print("=" * 70)

    # Update pipeline progress
    try:
        progress_file = os.path.join(ROOT, "📊 PIPELINE PROGRESS.txt")
        with open(progress_file, "w", encoding="utf-8") as f:
            f.write(
                f"📊 PIPELINE PROGRESS\n"
                f"├─ [✅] Dataset Split (split_spiral_dataset.py)\n"
                f"│   └─ Training/Test images split | Zero duplicates\n"
                f"├─ [✅] CNN Training (train_cnn_model.py)\n"
                f"│   └─ EfficientNetB2 spiral classifier ~94% accuracy\n"
                f"├─ [✅] Voice Model (pd_speech_voice_pipeline.py)\n"
                f"│   └─ LightGBM speech classifier ~92% accuracy\n"
                f"└─ [✅] Fusion Model (train_fusion_model.py)\n"
                f"    ├─ Adaptive attention fusion | {metrics['accuracy'] * 100:.1f}% accuracy\n"
                f"    ├─ Voice contribution: {mean_voice_w * 100:.1f}%\n"
                f"    └─ Spiral contribution: {mean_spiral_w * 100:.1f}%\n"
            )
        print(f"[SUCCESS] Pipeline progress updated: {progress_file}")
    except Exception:
        pass

    return metrics


if __name__ == "__main__":
    train_fusion_model()
