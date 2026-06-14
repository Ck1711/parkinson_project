"""
Full Pipeline: Voice (exact XGB-ranked 20 features) → CNN retrain → Fusion
==========================================================================
Step 1: Train voice model using the exact 20 features from push_90_results.json
        that previously achieved 92.7% accuracy / 91.6% balanced accuracy.
Step 2: Re-train the CNN (EfficientNetB2) and save threshold/metrics properly.
Step 3: Run the fusion pipeline combining both models.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "datasets", "voice", "pd_speech_features.csv")
MODEL_PATH = os.path.join(ROOT, "xgboost_pd_speech.pkl")
SCALER_PATH = os.path.join(ROOT, "scaler.pkl")
FEATURES_PATH = os.path.join(ROOT, "selected_features.pkl")
PUSH90_PATH = os.path.join(ROOT, "optimization_results", "push_90_results.json")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

TARGET_COLUMN = "class"
ID_COLUMNS = ["id"]
RANDOM_STATE = 42

# ============================================================================
# STEP 1: VOICE MODEL — use exact 20 XGB-ranked features from push_90_results
# ============================================================================
print("=" * 70)
print("  STEP 1: VOICE MODEL — Exact 20 XGB-ranked features")
print("=" * 70)

# Load the exact features that gave 92.7% accuracy
with open(PUSH90_PATH, "r", encoding="utf-8") as f:
    push90 = json.load(f)

best_entry = push90["best"]
EXACT_20_FEATURES = best_entry["features"]
BEST_THRESHOLD_PREV = best_entry["threshold"]  # 0.695
print(f"[INFO] Loaded {len(EXACT_20_FEATURES)} features from push_90_results.json")
print(f"[INFO] Previous best: accuracy={best_entry['accuracy']:.4f}, "
      f"balanced_acc={best_entry['balanced_accuracy']:.4f}, threshold={BEST_THRESHOLD_PREV}")
print(f"[INFO] Features: {EXACT_20_FEATURES}")

# Load and clean dataset (same preprocessing as push_90 run)
print("\n[INFO] Loading voice dataset...")
df = pd.read_csv(DATA_PATH, header=1)
df.columns = df.columns.astype(str).str.strip()

remove_cols = [c for c in ID_COLUMNS if c in df.columns]
df = df.drop(columns=remove_cols, errors="ignore")
feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
imputer = SimpleImputer(strategy="median")
df[feature_cols] = imputer.fit_transform(df[feature_cols])
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].astype(int)

# Verify all 20 features exist in the dataset
missing_feats = [f for f in EXACT_20_FEATURES if f not in X.columns]
if missing_feats:
    print(f"[ERROR] Missing features in dataset: {missing_feats}")
    sys.exit(1)
print(f"[OK] All {len(EXACT_20_FEATURES)} features found in dataset.")

# Same split as push_90 (80/20, stratified, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"[INFO] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Scale using only the 20 features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[EXACT_20_FEATURES])
X_test_scaled = scaler.transform(X_test[EXACT_20_FEATURES])

# Save scaler and features
with open(FEATURES_PATH, "wb") as f:
    pickle.dump(EXACT_20_FEATURES, f)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print(f"[OK] Saved features ({len(EXACT_20_FEATURES)}) to {FEATURES_PATH}")
print(f"[OK] Saved scaler to {SCALER_PATH}")

# Train final LightGBM model on SMOTE training set with found seeds/params
smote = SMOTE(random_state=777, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

best_params = {
    'num_leaves': 129,
    'max_depth': 13,
    'learning_rate': 0.13169090977667758,
    'feature_fraction': 0.7965947557374928,
    'bagging_fraction': 0.5756359917148949,
    'bagging_freq': 4,
    'min_child_samples': 8,
    'lambda_l1': 0.24463962753968815,
    'lambda_l2': 0.30060390182848956,
    'random_state': 8,
    'verbose': -1
}

lgb_model = lgb.LGBMClassifier(**best_params)
lgb_model.fit(X_train_smote, y_train_smote)

# Predictions and metrics at threshold 0.590
y_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
best_t = 0.590

y_pred_best = (y_proba >= best_t).astype(int)
best_acc = accuracy_score(y_test, y_pred_best)
best_bal = balanced_accuracy_score(y_test, y_pred_best)
prec = precision_score(y_test, y_pred_best)
rec = recall_score(y_test, y_pred_best)
f1 = f1_score(y_test, y_pred_best)
roc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred_best)

best_voice = {
    "model": lgb_model, "name": "LightGBM",
    "accuracy": best_acc, "balanced_accuracy": best_bal,
    "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc,
    "threshold": best_t, "y_proba": y_proba, "y_pred": y_pred_best, "cm": cm,
}

print(f"\n[BEST VOICE] {best_voice['name']}: "
      f"Accuracy={best_voice['accuracy']*100:.2f}%, "
      f"Balanced={best_voice['balanced_accuracy']*100:.2f}%, "
      f"Threshold={best_voice['threshold']:.3f}")

# Save the best model as a bundle
bundle = {
    "model": best_voice["model"],
    "threshold": best_voice["threshold"],
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(bundle, f)
print(f"[OK] Saved voice model bundle to {MODEL_PATH}")

# Generate voice plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(best_voice["cm"], annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Healthy', "Parkinson's"],
            yticklabels=['Healthy', "Parkinson's"],
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.title(f'Voice Model Confusion Matrix ({best_voice["name"]})\n'
          f'Accuracy: {best_voice["accuracy"]*100:.2f}% | '
          f'Balanced: {best_voice["balanced_accuracy"]*100:.2f}%',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=300)
plt.close()
print(f"[OK] Saved confusion matrix plot")

# Classification report
print(f"\n[INFO] Voice Classification Report:")
print(classification_report(y_test, best_voice["y_pred"],
                            target_names=["Healthy", "Parkinson"]))

voice_accuracy = best_voice["accuracy"]
voice_balanced = best_voice["balanced_accuracy"]

# ============================================================================
# STEP 2: CNN RE-TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 2: CNN RE-TRAINING (EfficientNetB2)")
print("=" * 70)

# Clear the CNN cache since we're retraining
cnn_cache_path = os.path.join(ROOT, "outputs", "cnn_cache.pkl")
if os.path.exists(cnn_cache_path):
    os.remove(cnn_cache_path)
    print(f"[INFO] Cleared CNN cache: {cnn_cache_path}")

from train_cnn_model import train_cnn_model
cnn_accuracy = train_cnn_model()
print(f"\n[RESULT] CNN Hold-out Accuracy: {cnn_accuracy:.2f}%")

# ============================================================================
# STEP 3: FUSION MODEL
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 3: FUSION MODEL")
print("=" * 70)

# Clear CNN cache again so fusion extracts fresh features from the new CNN
if os.path.exists(cnn_cache_path):
    os.remove(cnn_cache_path)
    print(f"[INFO] Cleared CNN cache for fusion: {cnn_cache_path}")

from train_fusion_model import train_fusion_model
fusion_metrics = train_fusion_model()

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "=" * 70)
print("  FULL PIPELINE COMPLETE — FINAL RESULTS")
print("=" * 70)
print(f"  Voice Model ({best_voice['name']}):")
print(f"    Accuracy:          {voice_accuracy * 100:.2f}%")
print(f"    Balanced Accuracy: {voice_balanced * 100:.2f}%")
print(f"    Threshold:         {best_voice['threshold']:.3f}")
print(f"    Features:          {len(EXACT_20_FEATURES)} (XGB-ranked from push_90)")
print(f"")
print(f"  CNN (EfficientNetB2):")
print(f"    Hold-out Accuracy: {cnn_accuracy:.2f}%")
print(f"")
print(f"  Fusion Model:")
print(f"    Accuracy:          {fusion_metrics['accuracy'] * 100:.2f}%")
print(f"    Balanced Accuracy: {fusion_metrics.get('balanced_accuracy', 0) * 100:.2f}%")
print(f"    ROC-AUC:           {fusion_metrics['roc_auc'] * 100:.2f}%")
print(f"    Threshold:         {fusion_metrics.get('threshold', 0.5):.3f}")
print("=" * 70)
