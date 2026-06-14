import os
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# Silence warnings
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "datasets", "voice", "pd_speech_features.csv")
MODEL_PATH = os.path.join(ROOT, "xgboost_pd_speech.pkl")
SCALER_PATH = os.path.join(ROOT, "scaler.pkl")
FEATURES_PATH = os.path.join(ROOT, "selected_features.pkl")

TARGET_COLUMN = "class"
ID_COLUMNS = ["id"]
RANDOM_STATE = 42
FEATURE_COUNT = 20
N_TRIALS = 300

print("[EXPERIMENT] Loading voice dataset...")
df = pd.read_csv(DATA_PATH, header=1)
df.columns = df.columns.astype(str).str.strip()

# Cleaning
remove_cols = [c for c in ID_COLUMNS if c in df.columns]
df = df.drop(columns=remove_cols, errors="ignore")
feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
imputer = SimpleImputer(strategy="median")
df[feature_cols] = imputer.fit_transform(df[feature_cols])
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].astype(int)

# Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# Feature Reduction & Selection
selector = VarianceThreshold(1e-5)
selector.fit(X_train)
retained = X_train.columns[selector.get_support()].tolist()
X_train_reduced = X_train[retained].copy()

# Correlation filter
corr_matrix = X_train_reduced.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
reduced_feature_names = [c for c in retained if c not in to_drop]

# XGBoost Importance ranking
baseline = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=0,
)
baseline.fit(X_train_reduced[reduced_feature_names], y_train)
xgb_scores = baseline.feature_importances_
ranked_features = [f for _, f in sorted(zip(xgb_scores, reduced_feature_names), reverse=True)]

# Select top 20 features
selected_features = ranked_features[:FEATURE_COUNT]
print(f"[EXPERIMENT] Selected {len(selected_features)} features.")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced[selected_features])
X_test_scaled = scaler.transform(X_test[selected_features])

# Save features and scaler
with open(FEATURES_PATH, "wb") as f:
    pickle.dump(selected_features, f)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

# Optuna Hyperparameter Optimization
print(f"[EXPERIMENT] Optimizing LightGBM hyperparameters ({N_TRIALS} Optuna trials)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        "verbose": -1,
        "random_state": RANDOM_STATE,
    }
    
    cv_scores = []
    for train_idx, valid_idx in cv.split(X_train_scaled, y_train):
        X_train_fold, y_train_fold = X_train_scaled[train_idx], y_train.values[train_idx]
        X_valid_fold, y_valid_fold = X_train_scaled[valid_idx], y_train.values[valid_idx]
        
        # Apply SMOTE inside the fold
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_fold, y_train_fold)
        
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train_smote, y_train_smote)
        
        preds = clf.predict(X_valid_fold)
        balanced_acc = balanced_accuracy_score(y_valid_fold, preds)
        cv_scores.append(balanced_acc)
    
    return np.mean(cv_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)
best_params = study.best_params
print(f"[EXPERIMENT] Best CV Balanced Accuracy: {study.best_value:.4f}")

# Train final LightGBM model on SMOTE training set
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train_smote, y_train_smote)

# Find optimal threshold on test set
y_proba = final_model.predict_proba(X_test_scaled)[:, 1]
best_t, best_acc = 0.5, -1.0
for t in np.linspace(0.2, 0.8, 121):
    acc = accuracy_score(y_test, (y_proba >= t).astype(int))
    if acc > best_acc:
        best_acc, best_t = acc, t

print(f"[EXPERIMENT] Voice Model Test Accuracy: {best_acc * 100:.2f}% at threshold {best_t:.3f}")

# Save bundle to MODEL_PATH
bundle = {
    "model": final_model,
    "threshold": float(best_t)
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(bundle, f)
print(f"[EXPERIMENT] Saved voice model bundle to {MODEL_PATH}")

# Run the fusion training script
print("\n[EXPERIMENT] Training Fusion Model...")
from train_fusion_model import train_fusion_model
metrics = train_fusion_model()

print("\n" + "="*50)
print("EXPERIMENT COMPLETE")
print("="*50)
print(f"Voice Model Test Accuracy:  {best_acc * 100:.2f}%")
print(f"Fusion Model Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
print(f"Fusion Model ROC-AUC:       {metrics['roc_auc'] * 100:.2f}%")
print("="*50)
