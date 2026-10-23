#!/usr/bin/env python3
"""
Model Training & Evaluation Pipeline for Polygraph Lie Detection
================================================================
Trains, benchmarks, and exports Edge ML models using physiological features
extracted from Pulse (HRV/BPM) and Galvanic Skin Response (GSR) signals.

Models evaluated:
  - Random Forest Classifier (Optimized for Raspberry Pi 4)
  - Gradient Boosting Classifier
  - Support Vector Machine (RBF Kernel + StandardScaler)
  - Logistic Regression (Baseline)

Evaluates using Cross-Validation and exports the best model artifact to:
  `ml/models/best_polygraph_model.pkl`
"""

import os
import sys
import json
import joblib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score

from ml.feature_extraction import process_raw_session_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "bpm_mean", "bpm_max", "bpm_min", "bpm_std", "bpm_delta", "bpm_max_delta",
    "hrv_sdnn", "hrv_rmssd", "hrv_pnn50",
    "pulse_mean", "pulse_std", "pulse_amplitude",
    "gsr_mean", "gsr_max", "gsr_min", "gsr_std",
    "gsr_delta", "gsr_max_delta", "gsr_ratio", "gsr_max_ratio",
    "gsr_max_slope", "gsr_scr_peaks", "gsr_auc", "stress_index"
]


def load_or_aggregate_dataset() -> pd.DataFrame:
    """Loads existing processed features or aggregates all raw CSV sessions."""
    processed_file = PROCESSED_DIR / "polygraph_features.csv"
    dfs = []

    # 1. Load processed dataset if available
    if processed_file.exists():
        df_processed = pd.read_csv(processed_file)
        dfs.append(df_processed)

    # 2. Check for any newly recorded raw sessions that are not yet aggregated
    raw_files = list(RAW_DIR.glob("*.csv"))
    if raw_files:
        print(f"Found {len(raw_files)} raw session file(s). Checking for updates...")
        for rf in raw_files:
            try:
                raw_df = pd.read_csv(rf)
                if {"pulse_raw", "gsr_raw", "ground_truth_label"}.issubset(raw_df.columns):
                    extracted = process_raw_session_dataframe(raw_df)
                    dfs.append(extracted)
            except Exception as e:
                print(f"[Warning] Failed to process {rf.name}: {e}")

    if not dfs:
        raise FileNotFoundError(
            "No physiological training data found! Please run `python -m ml.generate_synthetic_data` "
            "or record sessions with `python -m ml.data_collector` first."
        )

    combined_df = pd.concat(dfs, ignore_index=True)
    # Deduplicate rows by (subject_id, session_id, question_id) if present
    if {"subject_id", "session_id", "question_id"}.issubset(combined_df.columns):
        combined_df = combined_df.drop_duplicates(subset=["subject_id", "session_id", "question_id"])

    print(f"Total dataset size: {len(combined_df)} samples")
    print(f"Class counts: Truth (0) = {sum(combined_df['label'] == 0)}, Lie (1) = {sum(combined_df['label'] == 1)}")
    return combined_df


def train_and_evaluate():
    """Trains candidate models, selects the best, and saves the production pipeline."""
    print("=" * 65)
    print("      PHYSIOLOGICAL LIE DETECTOR - ML TRAINING PIPELINE")
    print("=" * 65)

    df = load_or_aggregate_dataset()

    # Verify required feature columns
    available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if len(available_cols) < 5:
        raise ValueError(f"Too few feature columns found in dataset: {available_cols}")

    X = df[available_cols].values
    y = df["label"].values.astype(int)

    # Candidate models tailored for fast, reliable Raspberry Pi inference
    candidates = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42))
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42))
        ])
    }

    print(f"\nEvaluating candidate models using 5-Fold Stratified Cross-Validation:")
    print("-" * 65)
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for name, pipeline in candidates.items():
        cv_scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
        acc = np.mean(cv_scores["test_accuracy"])
        f1 = np.mean(cv_scores["test_f1"])
        auc = np.mean(cv_scores["test_roc_auc"])
        results[name] = {
            "pipeline": pipeline,
            "accuracy": acc,
            "f1": f1,
            "roc_auc": auc
        }
        print(f"  * {name:<20} | Acc: {acc*100:5.2f}% | F1: {f1:0.3f} | ROC-AUC: {auc:0.3f}")

    # Select best model based on F1-score & ROC-AUC
    best_name = max(results.keys(), key=lambda k: results[k]["f1"] + results[k]["roc_auc"])
    best_info = results[best_name]
    print("-" * 65)
    print(f"--> BEST PERFORMING MODEL: {best_name} (Acc: {best_info['accuracy']*100:.1f}%, F1: {best_info['f1']:.3f})")

    # Fit best model on the complete dataset
    best_pipeline = best_info["pipeline"]
    best_pipeline.fit(X, y)

    # Feature Importance analysis (if tree-based)
    clf = best_pipeline.named_steps["clf"]
    feature_importances = {}
    if hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
        sorted_indices = np.argsort(imps)[::-1]
        print(f"\nTop Physiological Feature Importances ({best_name}):")
        for rank, idx in enumerate(sorted_indices[:8], 1):
            feat_name = available_cols[idx]
            feature_importances[feat_name] = round(float(imps[idx]), 4)
            bar = "#" * int(imps[idx] * 50)
            print(f"  {rank}. {feat_name:<18} [{imps[idx]:.4f}] {bar}")

    # Export production model artifact
    model_artifact_path = MODELS_DIR / "best_polygraph_model.pkl"
    artifact = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "feature_names": available_cols,
        "metrics": {
            "cv_accuracy": round(float(best_info["accuracy"]), 4),
            "cv_f1": round(float(best_info["f1"]), 4),
            "cv_roc_auc": round(float(best_info["roc_auc"]), 4)
        },
        "feature_importances": feature_importances,
        "trained_at": datetime.now().isoformat(),
        "num_samples": len(df)
    }

    joblib.dump(artifact, model_artifact_path)

    # Also save metadata JSON
    metadata = {
        "model_name": best_name,
        "feature_names": available_cols,
        "metrics": artifact["metrics"],
        "feature_importances": feature_importances,
        "trained_at": artifact["trained_at"],
        "num_samples": len(df)
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 65)
    print("SUCCESS! Model exported successfully:")
    print(f"  * Model binary    : {model_artifact_path}")
    print(f"  * Model metadata  : {MODELS_DIR / 'model_metadata.json'}")
    print("=" * 65)
    return model_artifact_path


if __name__ == "__main__":
    train_and_evaluate()
