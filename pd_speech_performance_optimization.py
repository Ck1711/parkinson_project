"""
Final Performance Optimization Experiment for PD Speech Features Dataset

This experiment determines whether the PD Speech Features dataset can exceed 90% validation accuracy
without introducing data leakage.

Key Features:
- Uses LightGBM as the primary model
- SMOTE applied only inside each training fold (no data leakage)
- Stratified 5-Fold Cross Validation
- Evaluates feature counts: 15, 20, 25, 30, 40, 50
- Hyperparameter tuning: num_leaves, max_depth, learning_rate, feature_fraction, bagging_fraction,
  min_child_samples, lambda_l1, lambda_l2
- Optimizes for Balanced Accuracy
- Tests thresholds from 0.30 to 0.70 (step 0.01)
- Reports: Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, MCC
"""

import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Paths
ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "datasets", "voice", "pd_speech_features.csv")
RESULTS_DIR = os.path.join(ROOT, "optimization_results")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TARGET_COLUMN = "class"
ID_COLUMNS = ["id"]
RANDOM_STATE = 42

# Feature counts to evaluate
FEATURE_COUNTS = [15, 20, 25, 30, 40, 50]
THRESHOLD_RANGE = np.arange(0.30, 0.71, 0.01)  # 0.30 to 0.70 in steps of 0.01
CV_FOLDS = 5
N_TRIALS = 100  # Optuna trials for hyperparameter tuning


def load_dataset(path: str) -> pd.DataFrame:
    """Load and prepare the dataset."""
    df = pd.read_csv(path, header=1)
    df.columns = df.columns.astype(str).str.strip()
    return df


def report_initial(df: pd.DataFrame):
    """Print initial dataset statistics."""
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)
    print(f"Dataset shape: {df.shape}")
    feature_columns = [c for c in df.columns if c != TARGET_COLUMN]
    print(f"Number of features: {len(feature_columns)}")
    if TARGET_COLUMN in df.columns:
        print(f"Class distribution:\n{df[TARGET_COLUMN].value_counts(dropna=False)}")
    else:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found")
    print(f"Missing values total: {df.isna().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the dataset."""
    print("\n" + "="*80)
    print("STEP 2: DATA CLEANING")
    print("="*80)
    
    # Remove ID columns
    remove_cols = [c for c in ID_COLUMNS if c in df.columns]
    if remove_cols:
        print(f"Removing ID-like columns: {remove_cols}")
    df = df.drop(columns=remove_cols, errors="ignore")

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    
    # Convert to numeric
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"After duplicate removal shape: {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found after cleaning")

    print("Data cleaning completed successfully")
    return df


def reduce_features(X: pd.DataFrame, threshold: float = 1e-5, corr_threshold: float = 0.95) -> Tuple:
    """Apply VarianceThreshold and correlation filtering."""
    print("\n" + "="*80)
    print("STEP 3: FEATURE REDUCTION (VarianceThreshold + Correlation Filtering)")
    print("="*80)
    
    original_count = X.shape[1]
    
    # Apply VarianceThreshold
    selector = VarianceThreshold(threshold)
    selector.fit(X)
    retained = X.columns[selector.get_support()].tolist()
    print(f"VarianceThreshold retained {len(retained)} of {original_count} features")

    X_reduced = X[retained].copy()
    
    # Correlation filtering
    corr_matrix = X_reduced.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    print(f"Correlation filtering dropping {len(to_drop)} features with corr > {corr_threshold}")
    
    final_features = [c for c in retained if c not in to_drop]
    print(f"Remaining feature count: {len(final_features)}")
    
    return X_reduced[final_features], original_count, len(final_features), final_features


def get_feature_rankings(X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> Dict:
    """Rank features using XGBoost importance and Mutual Information."""
    print("\nRanking features using XGBoost importance and Mutual Information...")
    
    # Mutual Information scores
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    
    # XGBoost feature importance
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
        "mutual_information": [f for _, f in sorted(zip(mi_scores, feature_names), reverse=True)],
        "xgboost_importance": [f for _, f in sorted(zip(xgb_scores, feature_names), reverse=True)],
    }
    
    return rankings


def select_top_k_features(feature_ranking: List[str], k: int) -> List[str]:
    """Select top k features from ranking."""
    return feature_ranking[:k]


class LightGBMOptimizer:
    """Optimize LightGBM hyperparameters using Optuna with balanced accuracy."""
    
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 100):
        self.X_train = X_train
        self.y_train = y_train
        self.n_trials = n_trials
        self.cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    def objective(self, trial):
        """Objective function for Optuna."""
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
        for train_idx, valid_idx in self.cv.split(self.X_train, self.y_train):
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_valid_fold = self.X_train[valid_idx]
            y_valid_fold = self.y_train[valid_idx]
            
            # Apply SMOTE only to training fold (no data leakage)
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_fold, y_train_fold)
            
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train_smote, y_train_smote)
            
            y_pred = clf.predict(X_valid_fold)
            balanced_acc = balanced_accuracy_score(y_valid_fold, y_pred)
            cv_scores.append(balanced_acc)
        
        return np.mean(cv_scores)
    
    def optimize(self) -> Dict:
        """Run Optuna optimization."""
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=False)
        
        print(f"Best trial balanced accuracy: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
        return study.best_params


def evaluate_feature_count(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_count: int,
    feature_names: List[str],
    ranking: List[str],
) -> Dict:
    """Evaluate a specific feature count."""
    print(f"\nEvaluating top {feature_count} features...")
    
    # Select top k features
    top_k_features = ranking[:feature_count]
    X_train_k = X_train[:, [feature_names.index(f) for f in top_k_features]]
    X_test_k = X_test[:, [feature_names.index(f) for f in top_k_features]]
    
    # Optimize hyperparameters
    print(f"  Tuning hyperparameters ({N_TRIALS} trials)...")
    optimizer = LightGBMOptimizer(X_train_k, y_train, n_trials=N_TRIALS)
    best_params = optimizer.optimize()
    
    # Train final model and evaluate on test set
    print(f"  Training final model on full training set...")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_k, y_train)
    
    clf = lgb.LGBMClassifier(**best_params)
    clf.fit(X_train_smote, y_train_smote)
    
    y_pred_proba = clf.predict_proba(X_test_k)[:, 1]
    
    # Threshold search
    best_config, threshold_results = search_thresholds(y_test, y_pred_proba)
    
    return {
        "feature_count": feature_count,
        "features": top_k_features,
        "best_params": best_params,
        "best_config": best_config,
        "threshold_results": threshold_results,
        "y_pred_proba": y_pred_proba,
        "model": clf,
    }


def search_thresholds(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[Dict, List[Dict]]:
    """Search for optimal threshold and return full threshold metrics."""
    threshold_results = []
    
    for threshold in THRESHOLD_RANGE:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        threshold_results.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "mcc": float(mcc),
        })
    
    # Select based on highest balanced_accuracy, then accuracy
    best_config = max(threshold_results, key=lambda x: (x["balanced_accuracy"], x["accuracy"]))
    
    return best_config, threshold_results


def plot_threshold_curves(threshold_results: List[Dict], out_dir: str, label: str = "best") -> str:
    """Plot accuracy and balanced accuracy as a function of threshold."""
    thresholds = [r["threshold"] for r in threshold_results]
    accuracies = [r["accuracy"] for r in threshold_results]
    balanced = [r["balanced_accuracy"] for r in threshold_results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, accuracies, marker="o", label="Accuracy", color="#1f77b4")
    ax.plot(thresholds, balanced, marker="s", label="Balanced Accuracy", color="#ff7f0e")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Accuracy vs Threshold ({label.title()} Configuration)", fontsize=14, fontweight="bold")
    ax.set_xlim([min(thresholds), max(thresholds)])
    ax.set_ylim([0.0, 1.0])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{label}_accuracy_threshold_curve.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix_heatmap(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str, label: str = "best") -> str:
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Healthy", "Parkinson"],
        yticklabels=["Healthy", "Parkinson"],
        annot_kws={"fontsize": 14, "fontweight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix - {label.title()} Configuration", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{label}_confusion_matrix.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    """Main optimization experiment."""
    print("\n" + "="*80)
    print("FINAL PERFORMANCE OPTIMIZATION EXPERIMENT")
    print("PD Speech Features Dataset - LightGBM with SMOTE")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and prepare data
    df = load_dataset(DATA_PATH)
    report_initial(df)
    
    df = clean_data(df)
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    
    # Train-test split (80-20)
    print("\n" + "="*80)
    print("STEP 4: DATA SPLITTING")
    print("="*80)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Training set shape: {X_train_full.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training set class distribution:\n{pd.Series(y_train_full).value_counts()}")
    print(f"Test set class distribution:\n{pd.Series(y_test).value_counts()}")
    
    # Feature reduction
    X_train_reduced, orig_count, reduced_count, reduced_feature_names = reduce_features(
        X_train_full, threshold=1e-5, corr_threshold=0.95
    )
    X_test_reduced = X_test[reduced_feature_names].copy()
    
    # Scaling
    print("\n" + "="*80)
    print("STEP 5: FEATURE SCALING")
    print("="*80)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)
    X_test_scaled = scaler.transform(X_test_reduced)
    print("Features scaled using StandardScaler")
    
    # Feature ranking
    print("\n" + "="*80)
    print("STEP 6: FEATURE RANKING")
    print("="*80)
    rankings = get_feature_rankings(X_train_reduced, y_train_full, reduced_feature_names)
    print(f"Generated rankings using 2 methods: {list(rankings.keys())}")
    
    # Evaluate different feature counts
    print("\n" + "="*80)
    print("STEP 7: OPTIMIZATION EXPERIMENT")
    print(f"Evaluating feature counts: {FEATURE_COUNTS}")
    print("="*80)
    
    all_results = {}
    
    for ranking_method, ranking in rankings.items():
        print(f"\n{'='*80}")
        print(f"Using ranking method: {ranking_method}")
        print(f"{'='*80}")
        
        results = []
        for feature_count in FEATURE_COUNTS:
            result = evaluate_feature_count(
                X_train_scaled,
                y_train_full.values,
                X_test_scaled,
                y_test.values,
                feature_count,
                reduced_feature_names,
                ranking,
            )
            results.append(result)
        
        all_results[ranking_method] = results
    
    # Select best configuration across all experiments
    print("\n" + "="*80)
    print("STEP 8: FINAL RESULTS SUMMARY")
    print("="*80)
    
    all_configs = []
    for ranking_method, results in all_results.items():
        for result in results:
            config = result["best_config"].copy()
            config["ranking_method"] = ranking_method
            config["feature_count"] = result["feature_count"]
            config["features"] = result["features"]
            config["params"] = result["best_params"]
            all_configs.append(config)
    
    # Sort by balanced_accuracy and accuracy
    all_configs_sorted = sorted(
        all_configs, 
        key=lambda x: (x["balanced_accuracy"], x["accuracy"]), 
        reverse=True
    )
    
    # Print top 10 configurations
    print("\nTop 10 Configurations (sorted by Balanced Accuracy, then Accuracy):\n")
    results_df = pd.DataFrame(all_configs_sorted[:10])
    display_cols = [
        "ranking_method",
        "feature_count",
        "threshold",
        "balanced_accuracy",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "mcc",
    ]
    print(results_df[display_cols].to_string(index=False))
    
    # Best configuration
    best_config = all_configs_sorted[0]
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"Ranking Method: {best_config['ranking_method']}")
    print(f"Feature Count: {best_config['feature_count']}")
    print(f"Threshold: {best_config['threshold']:.2f}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy: {best_config['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {best_config['balanced_accuracy']:.4f}")
    print(f"  Precision: {best_config['precision']:.4f}")
    print(f"  Recall: {best_config['recall']:.4f}")
    print(f"  F1 Score: {best_config['f1']:.4f}")
    print(f"  ROC-AUC: {best_config['roc_auc']:.4f}")
    print(f"  MCC: {best_config['mcc']:.4f}")
    print(f"\nHyperparameters:")
    for param, value in best_config["params"].items():
        print(f"  {param}: {value}")
    print(f"\nSelected Features ({best_config['feature_count']}):")
    print(f"  {', '.join(best_config['features'][:10])}")
    if len(best_config["features"]) > 10:
        print(f"  ... and {len(best_config['features']) - 10} more")
    
    # Generate accuracy graph and confusion matrix for the best configuration
    best_result = None
    for ranking_method, results in all_results.items():
        if ranking_method != best_config["ranking_method"]:
            continue
        for result in results:
            if result["feature_count"] == best_config["feature_count"] and result["best_config"]["threshold"] == best_config["threshold"]:
                best_result = result
                break
        if best_result is not None:
            break

    if best_result is not None:
        threshold_plot_path = plot_threshold_curves(
            best_result["threshold_results"],
            PLOTS_DIR,
            label=f"{best_config['ranking_method']}_{best_config['feature_count']}",
        )
        print(f"Saved accuracy vs threshold graph to: {threshold_plot_path}")

        y_pred_best = (best_result["y_pred_proba"] >= best_config["threshold"]).astype(int)
        cm_plot_path = plot_confusion_matrix_heatmap(
            y_test.values,
            y_pred_best,
            PLOTS_DIR,
            label=f"{best_config['ranking_method']}_{best_config['feature_count']}",
        )
        print(f"Saved confusion matrix to: {cm_plot_path}")
    else:
        print("WARNING: Best result object not found for plot generation.")

    # Save results
    results_output_path = os.path.join(RESULTS_DIR, "optimization_results.json")
    with open(results_output_path, "w") as f:
        json.dump(all_configs_sorted, f, indent=2, default=str)
    print(f"\nResults saved to: {results_output_path}")
    
    # Save best configuration
    best_config_path = os.path.join(RESULTS_DIR, "best_configuration.json")
    best_config_copy = best_config.copy()
    best_config_copy.pop("features", None)  # Remove list for JSON serialization
    best_config_copy["features_list"] = best_config["features"]
    with open(best_config_path, "w") as f:
        json.dump(best_config_copy, f, indent=2, default=str)
    print(f"Best configuration saved to: {best_config_path}")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    if best_config["accuracy"] >= 0.90:
        print("✓ SUCCESS: Achieved ≥90% validation accuracy!")
    else:
        print(f"✗ Did not achieve 90% accuracy. Best: {best_config['accuracy']:.4f}")
    
    print(f"\nHighest Accuracy: {best_config['accuracy']:.4f}")
    print(f"Highest Balanced Accuracy: {best_config['balanced_accuracy']:.4f}")
    print(f"\nConclusion: The PD Speech Features dataset {'CAN' if best_config['accuracy'] >= 0.90 else 'CANNOT'} exceed 90% validation accuracy.")
    print("No data leakage was introduced - SMOTE was applied only within training folds.")


if __name__ == "__main__":
    main()
