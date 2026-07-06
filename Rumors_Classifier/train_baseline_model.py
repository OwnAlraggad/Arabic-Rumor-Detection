"""
train_baseline_model.py
=======================
Reproducible training script for the Arabic-rumour baseline XGBoost model.

Usage
-----
Run from the project root (arabic_rumor_scanner/):

    python -m Rumors_Classifier.train_baseline_model [OPTIONS]

Options
-------
  --tweets      Path to Tweets.txt            (default: ArCOV19-Rumors/tweet_verification/Tweets.txt)
  --replies     Path to replies prop. file    (default: ArCOV19-Rumors/.../replies)
  --retweets    Path to retweets prop. file   (default: ArCOV19-Rumors/.../retweets)
  --output-dir  Directory for saved model/logs (default: Rumors_Classifier/notebooks)
  --seed        Global random seed            (default: 42)
  --test-size   Fraction held out for test    (default: 0.2)
  --val-size    Fraction held out for val     (default: 0.2)
  --n-estimators Max XGBoost trees            (default: 2000)
  --verbose-xgb   Flag: print XGBoost per-round eval log

Pipeline
--------
  1. Feature engineering (ArabicTextPreprocessor + propagation counts)
  2. Stratified 60 / 20 / 20 train / val / test split
  3. ColumnTransformer: StandardScaler (handcrafted) + TF-IDF (text)
  4. XGBClassifier with class-imbalance weight + early stopping on val
  5. Evaluation: Accuracy, F1, ROC-AUC, PR-AUC, confusion matrix
  6. Persist full pipeline + metrics to --output-dir
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Project-level imports (run as  python -m Rumors_Classifier.train_baseline_model)
from Rumors_Classifier.feature_pipeline import (
    ArabicTextPreprocessor,
    prepare_data,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HANDCRAFTED_COLS = [
    "char_len",
    "avg_word_length",
    "num_hashtags",
    "num_mentions",
    "num_urls",
    "num_emojis",
    "num_exclamations",
    "num_replies",
    "num_retweets",
]

XGB_FIXED_PARAMS = dict(
    objective="binary:logistic",
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="logloss",
    early_stopping_rounds=20,
)

TFIDF_PARAMS = dict(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)


# ---------------------------------------------------------------------------
# Helper: build ColumnTransformer
# ---------------------------------------------------------------------------

def _build_preprocessor(handcrafted_cols, text_col="normalized_text"):
    """Return a ColumnTransformer that scales handcrafted cols + TF-IDF text."""
    return ColumnTransformer(
        [
            ("handcrafted", StandardScaler(), handcrafted_cols),
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS), text_col),
        ]
    )


# ---------------------------------------------------------------------------
# Helper: compute scale_pos_weight for imbalanced binary labels
# ---------------------------------------------------------------------------

def _scale_pos_weight(y: pd.Series) -> float:
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return float(neg / pos) if pos > 0 else 1.0


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Full training pipeline, driven by parsed CLI arguments."""

    seed = args.seed
    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    models_dir = output_dir / "models"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Feature engineering
    # ------------------------------------------------------------------
    log.info("Loading and engineering features …")
    tweets_df = prepare_data(
        tweets_path=args.tweets,
        replies_path=args.replies,
        retweets_path=args.retweets,
        text_preprocessor=ArabicTextPreprocessor(
            remove_urls=True,
            remove_mentions=False,
            remove_emojis=False,
            remove_numbers=True,
            remove_punctuation=True,
            reduce_repeated_chars=True,
        ),
    )
    log.info("Dataset shape after feature engineering: %s", tweets_df.shape)

    # ------------------------------------------------------------------
    # 2. Train / val / test split  (60 / 20 / 20, stratified)
    # ------------------------------------------------------------------
    log.info("Splitting dataset (test=%.0f%%, val=%.0f%%) …",
             args.test_size * 100, args.val_size * 100)

    X_temp, X_test, y_temp, y_test = train_test_split(
        tweets_df,
        tweets_df["label"].astype(int),
        test_size=args.test_size,
        stratify=tweets_df["label"],
        random_state=seed,
    )
    # val fraction relative to the temporary (non-test) pool
    val_frac = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_frac,
        stratify=y_temp,
        random_state=seed,
    )

    log.info(
        "Split sizes → train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    log.info(
        "Label ratios → train: %.3f | val: %.3f | test: %.3f",
        y_train.mean(), y_val.mean(), y_test.mean(),
    )


    # ------------------------------------------------------------------
    # 3. Build ColumnTransformer + XGBoost
    # ------------------------------------------------------------------
    log.info("Building column transformer …")
    col_transformer = _build_preprocessor(HANDCRAFTED_COLS)

    # Fit on train, transform val for early stopping
    X_train_t = col_transformer.fit_transform(X_train)
    X_val_t = col_transformer.transform(X_val)

    spw = _scale_pos_weight(y_train)
    log.info("Class imbalance → scale_pos_weight = %.4f", spw)

    xgb_model = XGBClassifier(
        scale_pos_weight=spw,
        n_estimators=args.n_estimators,
        random_state=seed,
        **XGB_FIXED_PARAMS,
    )

    log.info("Training XGBoost (max %d estimators, early stopping %d rounds) …",
             args.n_estimators, XGB_FIXED_PARAMS["early_stopping_rounds"])
    xgb_model.fit(
        X_train_t,
        y_train,
        eval_set=[(X_val_t, y_val)],
        verbose=args.verbose_xgb,
    )
    log.info(
        "Training done. Best iteration: %d | Best val logloss: %.6f",
        xgb_model.best_iteration,
        xgb_model.best_score,
    )

    # ------------------------------------------------------------------
    # 4. Evaluate on test set
    # ------------------------------------------------------------------
    X_test_t = col_transformer.transform(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test_t)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Baseline XGBoost — Test-Set Results ===")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["False", "True"]))

    # ------------------------------------------------------------------
    # 6. Persist metrics
    # ------------------------------------------------------------------
    metrics = {
        "accuracy": round(float(accuracy), 6),
        "f1_binary": round(float(f1), 6),
        "roc_auc": round(float(roc_auc), 6),
        "pr_auc": round(float(pr_auc), 6),
        "confusion_matrix": cm.tolist(),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "scale_pos_weight": round(float(spw), 6),
        "best_iteration": int(xgb_model.best_iteration),
        "best_val_logloss": round(float(xgb_model.best_score), 6),
        "seed": seed,
        "test_size_frac": args.test_size,
        "val_size_frac": args.val_size,
        "n_estimators_max": args.n_estimators,
    }

    metrics_path = logs_dir / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Metrics saved → %s", metrics_path)

    # ------------------------------------------------------------------
    # 7. Persist model artifacts
    # ------------------------------------------------------------------
    # Wrap fitted col_transformer + fitted xgb_model into a Pipeline
    # NOTE: xgb_model already has internal early-stopping state; wrapping
    #       it in Pipeline is for clean inference — predict() calls transform
    #       then predict automatically.
    final_pipeline = Pipeline(
        [
            ("preprocessor", col_transformer),
            ("classifier", xgb_model),
        ]
    )

    pipeline_path = models_dir / "xgb_combined_pipeline.pkl"
    preprocessor_path = models_dir / "preprocessor.pkl"

    joblib.dump(final_pipeline, pipeline_path)
    log.info("Full pipeline saved → %s", pipeline_path)

    joblib.dump(col_transformer, preprocessor_path)
    log.info("Preprocessor saved → %s", preprocessor_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Arabic rumour baseline XGBoost classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data paths ---
    _DATA = "../ArCOV19-Rumors/tweet_verification"
    parser.add_argument(
        "--tweets",
        default=f"{_DATA}/Tweets.txt",
        help="Path to the Tweets.txt TSV file.",
    )
    parser.add_argument(
        "--replies",
        default=f"{_DATA}/propagation_networks/replies",
        help="Path to the replies propagation file.",
    )
    parser.add_argument(
        "--retweets",
        default=f"{_DATA}/propagation_networks/retweets",
        help="Path to the retweets propagation file.",
    )

    # --- Output ---
    parser.add_argument(
        "--output-dir",
        default="Rumors_Classifier",
        help="Directory where logs/ and models/ sub-folders will be created.",
    )

    # --- Reproducibility ---
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")

    # --- Split sizes ---
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of dataset reserved for final test evaluation.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of dataset reserved for validation (early stopping).",
    )

    # --- Model ---
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=2000,
        help="Maximum number of XGBoost boosting rounds.",
    )

    parser.add_argument(
        "--verbose-xgb",
        action="store_true",
        help="Print XGBoost per-round evaluation log.",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    train(args)