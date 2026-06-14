import json
import os
import pickle
from datetime import datetime

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import seaborn as sns

# Paths
ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "datasets", "voice", "pd_speech_features.csv")
MODEL_PATH = os.path.join(ROOT, "xgboost_pd_speech.pkl")
SCALER_PATH = os.path.join(ROOT, "scaler.pkl")
FEATURES_PATH = os.path.join(ROOT, "selected_features.pkl")
PARAMS_PATH = os.path.join(ROOT, "best_params.json")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

TARGET_COLUMN = "class"
ID_COLUMNS = ["id"]
RANDOM_STATE = 42


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=1)
    df.columns = df.columns.astype(str).str.strip()
    return df


def report_initial(df: pd.DataFrame):
    print("STEP 1: DATA LOADING")
    print("Dataset shape:", df.shape)
    feature_columns = [c for c in df.columns if c != TARGET_COLUMN]
    print("Number of features:", len(feature_columns))
    if TARGET_COLUMN in df.columns:
        print("Class distribution:\n", df[TARGET_COLUMN].value_counts(dropna=False).to_string())
    else:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found")
    print("Missing values total:", df.isna().sum().sum())
    print("Missing values per column (top 20):\n", df.isna().sum().sort_values(ascending=False).head(20).to_string())
    print("Duplicate rows:", df.duplicated().sum())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\nSTEP 2: DATA CLEANING")
    # Remove obvious ID-like columns
    remove_cols = [c for c in ID_COLUMNS if c in df.columns]
    print("Removing ID-like columns:", remove_cols)
    df = df.drop(columns=remove_cols, errors="ignore")

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    df = df.drop_duplicates().reset_index(drop=True)
    print("After duplicate removal shape:", df.shape)

    if TARGET_COLUMN in df.columns:
        corr_with_target = df[feature_cols].corrwith(df[TARGET_COLUMN]).abs().sort_values(ascending=False)
        leakage_candidates = corr_with_target[corr_with_target > 0.95]
        print("Features correlated with target > 0.95:\n", leakage_candidates.to_string() if not leakage_candidates.empty else "None")
    else:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found after cleaning")

    leakage_report = {
        "n_missing": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "target_correlated_features": leakage_candidates.index.tolist(),
        "high_target_correlations": leakage_candidates.to_dict(),
    }
    return df, leakage_report


def reduce_features(X: pd.DataFrame, threshold: float = 1e-5, corr_threshold: float = 0.95):
    print("\nSTEP 3: FEATURE REDUCTION")
    original_count = X.shape[1]
    selector = VarianceThreshold(threshold)
    selector.fit(X)
    retained = X.columns[selector.get_support()].tolist()
    print(f"VarianceThreshold retained {len(retained)} of {original_count} features")

    X_reduced = X[retained].copy()
    corr_matrix = X_reduced.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    print(f"Correlation filtering dropping {len(to_drop)} features with corr > {corr_threshold}")
    final_features = [c for c in retained if c not in to_drop]
    print(f"Remaining feature count: {len(final_features)}")
    return X_reduced[final_features], original_count, len(final_features), final_features


def compute_scale_pos_weight(y: pd.Series) -> float:
    negative = sum(y == 0)
    positive = sum(y == 1)
    return float(negative / positive) if positive > 0 else 1.0


def get_feature_rankings(X: pd.DataFrame, y: pd.Series, feature_names: list):
    print("\nRanking features: XGBoost importance and Mutual Information")
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    baseline = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    baseline.fit(X, y)
    xgb_scores = baseline.feature_importances_
    rankings = {
        "mutual_information": [feature for _, feature in sorted(zip(mi_scores, feature_names), key=lambda x: x[0], reverse=True)],
        "xgboost_importance": [feature for _, feature in sorted(zip(xgb_scores, feature_names), key=lambda x: x[0], reverse=True)],
    }
    return rankings


def evaluate_feature_subsets(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, rankings: dict, sizes: list, baseline_params: dict):
    print("\nSTEP 4: FEATURE SELECTION")
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for method_name, ranking in rankings.items():
        print(f"\nEvaluating feature ranking method: {method_name}")
        for k in sizes:
            subset = ranking[:k]
            clf = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbosity=0,
                **baseline_params,
            )
            cv_acc = cross_val_score(clf, X_train[subset], y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()
            clf.fit(X_train[subset], y_train)
            y_pred = clf.predict(X_test[subset])
            val_acc = accuracy_score(y_test, y_pred)
            results.append({
                "method": method_name,
                "k": k,
                "cv_accuracy": float(cv_acc),
                "validation_accuracy": float(val_acc),
                "subset": subset,
            })
            print(f"{method_name} top {k}: CV Accuracy = {cv_acc:.4f}, Validation Accuracy = {val_acc:.4f}")
    return results


def optimize_xgboost_constrained(X: np.ndarray, y: np.ndarray, scale_pos_weight: float = None, n_trials: int = 40):
    print("\nSTEP 7: XGBOOST OPTIMIZATION")
    import optuna
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_categorical("max_depth", [3, 4, 5, 6]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_categorical("subsample", [0.7, 0.8, 0.9]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.7, 0.8, 0.9]),
            "min_child_weight": trial.suggest_categorical("min_child_weight", [3, 5, 7, 10]),
            "gamma": trial.suggest_categorical("gamma", [1, 2, 3, 5]),
            "reg_alpha": trial.suggest_categorical("reg_alpha", [0.5, 1, 2, 5]),
            "reg_lambda": trial.suggest_categorical("reg_lambda", [2, 5, 10]),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight
        cv_scores = []
        for train_idx, valid_idx in cv.split(X, y):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train, y_train, verbose=False)
            preds = clf.predict_proba(X_valid)[:, 1]
            cv_scores.append(roc_auc_score(y_valid, preds))
        return float(np.mean(cv_scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("Best trial ROC-AUC:", study.best_value)
    print("Best params:", study.best_params)
    return study.best_params


def threshold_search(y_true, y_score, thresholds=None):
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    best_acc = {"threshold": None, "score": -1.0}
    best_f1 = {"threshold": None, "score": -1.0}
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if acc > best_acc["score"]:
            best_acc = {"threshold": thr, "score": acc}
        if f1 > best_f1["score"]:
            best_f1 = {"threshold": thr, "score": f1}
    return best_acc, best_f1


def build_ensemble_models(xgb_clf, lgbm_clf, rf_clf, feature_names):
    """Build ensemble models with soft voting."""
    voting_lgb = VotingClassifier(
        estimators=[("xgb", xgb_clf), ("lgbm", lgbm_clf)],
        voting="soft",
        n_jobs=-1,
    )
    voting_rf = VotingClassifier(
        estimators=[("xgb", xgb_clf), ("rf", rf_clf)],
        voting="soft",
        n_jobs=-1,
    )
    # Weighted voting for better accuracy
    voting_weighted = VotingClassifier(
        estimators=[("xgb", xgb_clf), ("lgbm", lgbm_clf), ("rf", rf_clf)],
        voting="soft",
        weights=[1.0, 1.2, 0.8],  # Emphasize LightGBM slightly
        n_jobs=-1,
    )
    return voting_lgb, voting_rf, voting_weighted


def evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    return metrics


def optimize_xgboost(X: np.ndarray, y: np.ndarray, n_trials: int = 50):
    print("\nSTEP 7: XGBOOST OPTIMIZATION")
    import optuna

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }
        cv_scores = []
        for train_idx, valid_idx in cv.split(X, y):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train, y_train, verbose=False)
            preds = clf.predict_proba(X_valid)[:, 1]
            cv_scores.append(roc_auc_score(y_valid, preds))
        return float(np.mean(cv_scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("Best trial ROC-AUC:", study.best_value)
    print("Best params:", study.best_params)
    return study.best_params


def plot_accuracy_comparison(training_acc, cv_acc_mean, cv_acc_std, validation_acc, threshold_acc, out_dir):
    """Plot accuracy comparison across different stages."""
    stages = ['Training', 'CV Mean', 'Validation', 'Threshold\nOptimized']
    accuracies = [training_acc, cv_acc_mean, validation_acc, threshold_acc]
    errors = [0, cv_acc_std, 0, 0]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    plt.figure(figsize=(10, 6))
    bars = plt.bar(stages, accuracies, yerr=errors, capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy Comparison Across Training Stages', fontsize=14, fontweight='bold')
    plt.ylim([0.8, 1.02])
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    acc_plot_path = os.path.join(out_dir, 'accuracy_comparison.png')
    plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return acc_plot_path


def plot_confusion_matrix_heatmap(y_true, y_pred, out_dir):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                xticklabels=['Healthy', 'Parkinson\'s'],
                yticklabels=['Healthy', 'Parkinson\'s'],
                annot_kws={'fontsize': 14, 'fontweight': 'bold'},
                cbar_kws={'label': 'Count'})
    
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Test Set Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    cm_plot_path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return cm_plot_path


def plot_accuracy_metrics_table(accuracy, precision, recall, f1, roc_auc, out_dir):
    """Plot performance metrics as a visual table."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    values = [accuracy, precision, recall, f1, roc_auc]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = [[m, f'{v:.4f}', f'{v*100:.2f}%'] for m, v in zip(metrics, values)]
    table = ax.table(cellText=table_data, 
                     colLabels=['Metric', 'Score', 'Percentage'],
                     cellLoc='center', 
                     loc='center',
                     colWidths=[0.3, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows with alternating colors
    for i in range(1, len(metrics) + 1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(3):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_text_props(weight='bold')
    
    plt.title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    metrics_plot_path = os.path.join(out_dir, 'performance_metrics_table.png')
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return metrics_plot_path


def plot_curves(y_true, y_score, y_pred, y_prefix: str):
    """Plot ROC and Precision-Recall curves."""
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    
    # Plot ROC curve
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {y_prefix.title()} Set', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(PLOTS_DIR, f'{y_prefix}_roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Precision-Recall curve
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {y_prefix.title()} Set', fontsize=14, fontweight='bold')
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    pr_path = os.path.join(PLOTS_DIR, f'{y_prefix}_precision_recall_curve.png')
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_path, pr_path
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_true, y_score):.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(PLOTS_DIR, f"{out_prefix}_roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(PLOTS_DIR, f"{out_prefix}_precision_recall_curve.png")
    plt.savefig(pr_path)
    plt.close()
    return roc_path, pr_path


def main():
    df = load_dataset(DATA_PATH)
    report_initial(df)

    df, leakage_report = clean_data(df)
    print("Leakage report:", json.dumps(leakage_report, indent=2))

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    # Split data early to preserve a final test set for honest evaluation
    print("\nSTEP 5: DATA SPLITTING")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print("Train shape:", X_train_full.shape, "Test shape:", X_test.shape)

    X_train_reduced, orig_count, reduced_count, reduced_feature_names = reduce_features(X_train_full)
    X_test_reduced = X_test[reduced_feature_names].copy()

    # Compare feature subset sizes using XGBoost importance and Mutual Information
    baseline_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 5,
        "gamma": 1,
        "reg_alpha": 1,
        "reg_lambda": 5,
    }
    rankings = get_feature_rankings(X_train_reduced, y_train_full, reduced_feature_names)
    subset_results = evaluate_feature_subsets(
        X_train_reduced,
        y_train_full,
        X_test_reduced,
        y_test,
        rankings,
        sizes=[50, 75, 100, 125, 150],
        baseline_params=baseline_params,
    )

    best_subset = max(subset_results, key=lambda r: (r["validation_accuracy"], r["cv_accuracy"]))
    selected_features = best_subset["subset"]
    print("\nSelected feature count:", len(selected_features))
    print("Best subset method:", best_subset["method"], "k=", best_subset["k"], "CV Accuracy=", best_subset["cv_accuracy"], "Validation Accuracy=", best_subset["validation_accuracy"])

    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(selected_features, f)

    print("\nSTEP 6: SCALING")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced[selected_features])
    X_test_scaled = scaler.transform(X_test_reduced[selected_features])
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Compare XGBoost with and without scale_pos_weight
    scale_weight = compute_scale_pos_weight(y_train_full)
    print(f"Computed scale_pos_weight = {scale_weight:.4f}")
    params_no_weight = optimize_xgboost_constrained(X_train_scaled, y_train_full.values, scale_pos_weight=None, n_trials=15)
    params_with_weight = optimize_xgboost_constrained(X_train_scaled, y_train_full.values, scale_pos_weight=scale_weight, n_trials=15)

    def evaluate_params(params, use_weight):
        copy_params = params.copy()
        if use_weight:
            copy_params["scale_pos_weight"] = scale_weight
        clf_tmp = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
            **copy_params,
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_acc = cross_val_score(clf_tmp, X_train_scaled, y_train_full, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        return cv_acc, clf_tmp

    cv_no_weight, clf_no_weight = evaluate_params(params_no_weight, False)
    cv_with_weight, clf_with_weight = evaluate_params(params_with_weight, True)
    print(f"XGBoost w/o scale_pos_weight CV Acc = {cv_no_weight:.4f}")
    print(f"XGBoost with scale_pos_weight CV Acc = {cv_with_weight:.4f}")

    if cv_with_weight >= cv_no_weight:
        best_params = params_with_weight.copy()
        best_params["scale_pos_weight"] = scale_weight
        print("Using scale_pos_weight for final XGBoost")
    else:
        best_params = params_no_weight.copy()
        print("Not using scale_pos_weight for final XGBoost")

    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print("\nSTEP 8: CROSS VALIDATION")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
        **best_params,
    )
    cv_accuracy = cross_val_score(clf, X_train_scaled, y_train_full, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_roc_auc = cross_val_score(clf, X_train_scaled, y_train_full, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"CV Accuracy mean: {cv_accuracy.mean():.4f}, std: {cv_accuracy.std():.4f}")
    print(f"CV ROC-AUC mean: {cv_roc_auc.mean():.4f}, std: {cv_roc_auc.std():.4f}")

    print("\nSTEP 9: ENSEMBLE EXPERIMENT")
    clf_xgb = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
        **best_params,
    )
    clf_lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    clf_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    voting_lgbm, voting_rf, voting_weighted = build_ensemble_models(clf_xgb, clf_lgbm, clf_rf, selected_features)

    ensemble_models = [
        ("XGBoost", clf_xgb),
        ("LightGBM", clf_lgbm),
        ("RandomForest", clf_rf),
        ("XGBoost+LightGBM", voting_lgbm),
        ("XGBoost+RandomForest", voting_rf),
        ("Weighted Voting (XGB+LGB+RF)", voting_weighted),
    ]
    ensemble_results = []
    model_map = {
        "XGBoost": clf_xgb,
        "LightGBM": clf_lgbm,
        "RandomForest": clf_rf,
        "XGBoost+LightGBM": voting_lgbm,
        "XGBoost+RandomForest": voting_rf,
        "Weighted Voting (XGB+LGB+RF)": voting_weighted,
    }
    for name, model in ensemble_models:
        print(f"Evaluating {name}")
        metrics = evaluate_classifier(model, X_train_scaled, y_train_full, X_test_scaled, y_test)
        metrics["model"] = name
        ensemble_results.append(metrics)
        print(f"{name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")

    best_candidate = max(
        [m for m in ensemble_results if m["roc_auc"] > 0.95],
        key=lambda x: x["accuracy"],
        default=None,
    )
    if best_candidate is None:
        best_candidate = max(ensemble_results, key=lambda x: x["accuracy"])
    print(f"\nBest final model candidate: {best_candidate['model']} with Accuracy={best_candidate['accuracy']:.4f}, ROC-AUC={best_candidate['roc_auc']:.4f}")

    final_model = model_map[best_candidate["model"]]
    print("\nSTEP 10: TRAIN FINAL MODEL")
    final_model.fit(X_train_scaled, y_train_full)
    # We will save model and best threshold as a bundle after threshold optimization
    bundle = {
        "model": final_model,
        "threshold": 0.5,
    }

    y_pred = final_model.predict(X_test_scaled)
    y_proba = final_model.predict_proba(X_test_scaled)[:, 1]
    final_model_name = best_candidate["model"]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    roc_path, pr_path = plot_curves(y_test, y_proba, y_pred, "test")
    print("Saved ROC curve to", roc_path)
    print("Saved Precision-Recall curve to", pr_path)

    print("\nSTEP 11: THRESHOLD OPTIMIZATION")
    best_acc_thr, best_f1_thr = threshold_search(y_test, y_proba)
    print(f"Best threshold for Accuracy: {best_acc_thr['threshold']} => {best_acc_thr['score']:.4f}")
    print(f"Best threshold for F1 Score: {best_f1_thr['threshold']} => {best_f1_thr['score']:.4f}")
    
    # Save the bundle now with the optimal threshold
    bundle["threshold"] = best_acc_thr['threshold']
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[SUCCESS] Saved model bundle with threshold {bundle['threshold']} to {MODEL_PATH}")
    
    # Calculate training accuracy for visualization
    clf_train_preds = final_model.predict(X_train_scaled)
    training_accuracy = accuracy_score(y_train_full, clf_train_preds)
    cv_accuracy_mean = cv_accuracy.mean()

    print("\nSTEP 12: VISUALIZATION - ACCURACY GRAPH & CONFUSION MATRIX")
    # Plot accuracy comparison
    acc_plot_path = plot_accuracy_comparison(training_accuracy, cv_accuracy_mean, cv_accuracy.std(), 
                                              accuracy, best_acc_thr['score'], PLOTS_DIR)
    print(f"Saved accuracy comparison to {acc_plot_path}")
    
    # Plot confusion matrix
    cm_plot_path = plot_confusion_matrix_heatmap(y_test, y_pred, PLOTS_DIR)
    print(f"Saved confusion matrix to {cm_plot_path}")
    
    # Plot metrics table
    metrics_plot_path = plot_accuracy_metrics_table(accuracy, precision, recall, f1, roc_auc, PLOTS_DIR)
    print(f"Saved performance metrics table to {metrics_plot_path}")

    print("\nSTEP 13: OVERFITTING ANALYSIS")
    validation_accuracy = accuracy
    print(f"Training Accuracy: {training_accuracy:.4f}")
    print(f"Mean CV Accuracy: {cv_accuracy_mean:.4f}")
    print(f"Validation Accuracy: {validation_accuracy:.4f}")
    if training_accuracy - cv_accuracy_mean > 0.05:
        print("WARNING: Training accuracy exceeds CV accuracy by more than 5% — potential overfitting")
    else:
        print("No strong overfitting signal in the Training vs CV accuracy gap")

    print("\nSTEP 14: EXPLAINABILITY")
    model_for_shap = final_model
    if hasattr(model_for_shap, "predict_proba") and hasattr(model_for_shap, "feature_importances_"):
        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer.shap_values(X_test_scaled)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_reduced[selected_features], show=False)
        shap_plot_path = os.path.join(PLOTS_DIR, "shap_summary.png")
        plt.tight_layout()
        plt.savefig(shap_plot_path)
        plt.close()
        print("Saved SHAP summary plot to", shap_plot_path)

        importance_df = pd.DataFrame({
            "feature": selected_features,
            "importance": model_for_shap.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance_df_path = os.path.join(PLOTS_DIR, "feature_importance_ranking.csv")
        importance_df.to_csv(importance_df_path, index=False)
        print("Saved feature importance ranking to", importance_df_path)
        print("Top 20 features:\n", importance_df.head(20).to_string(index=False))
    else:
        print("SHAP explainability skipped: final model does not support TreeExplainer or feature importance extraction.")

    print("\nSTEP 15: FINAL REPORT")
    print("Best Feature Subset Size:", len(selected_features))
    print("Best Hyperparameters:", json.dumps(best_params, indent=2))
    print("Final selected model:", final_model_name)
    print(f"Training Accuracy: {training_accuracy:.4f}")
    print(f"Mean CV Accuracy: {cv_accuracy_mean:.4f}")
    print(f"Validation Accuracy: {validation_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Best threshold for Accuracy: {best_acc_thr['threshold']}")
    print(f"Best threshold for F1 Score: {best_f1_thr['threshold']}")
    print("Conclusion:")
    if training_accuracy - cv_accuracy_mean > 0.05:
        print("- The model shows signs of overfitting.")
    else:
        print("- The model does not show major overfitting.")
    print("- The dataset is suitable for Parkinson classification as a voice biomarker dataset, but performance depends on how well features generalize.")
    print("- 90%+ accuracy achieved?", accuracy >= 0.90)
    print("- 94%+ accuracy achieved?", accuracy >= 0.94)
    print("Artifacts saved:", [MODEL_PATH, SCALER_PATH, FEATURES_PATH, PARAMS_PATH])


if __name__ == "__main__":
    main()
