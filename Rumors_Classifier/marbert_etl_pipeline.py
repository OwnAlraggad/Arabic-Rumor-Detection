"""
marbert_etl_pipeline.py
=======================
Feature / ETL pipeline for the MARBERT Arabic-rumour classifier.

This module handles all data loading and preprocessing specific to the
MARBERT model (i.e. minimal whitespace normalisation rather than the heavy
Arabic NLP pipeline used by the XGBoost baseline).

Design decisions (from notebook conclusions)
--------------------------------------------
* ``use_propagation=False`` is the best-performing config: propagation
  features (num_replies, num_retweets) added noise rather than signal.
  They are still loaded and exposed so experiments can easily re-enable them.
* Deduplication is done by groupby-agg (keeping first label / text, mean
  propagation counts) to faithfully replicate the notebook behaviour.
* Text preprocessing is intentionally minimal (whitespace collapse only)
  so MARBERT can leverage its own sub-word tokenisation on raw Arabic.

Usage
-----
    from Rumors_Classifier.marbert_etl_pipeline import load_and_prepare_marbert_data

    df = load_and_prepare_marbert_data(
        tweets_path  = "ArCOV19-Rumors/tweet_verification/Tweets.txt",
        replies_path = "ArCOV19-Rumors/tweet_verification/propagation_networks/replies",
        retweets_path= "ArCOV19-Rumors/tweet_verification/propagation_networks/retweets",
    )

CLI
---
    python -m Rumors_Classifier.marbert_etl_pipeline \\
        --tweets   ArCOV19-Rumors/tweet_verification/Tweets.txt \\
        --replies  ArCOV19-Rumors/tweet_verification/propagation_networks/replies \\
        --retweets ArCOV19-Rumors/tweet_verification/propagation_networks/retweets \\
        --output   Rumors_Classifier/data/marbert_features.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from Rumors_Classifier.feature_pipeline import load_tweets, load_propagation_counts

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def minimal_preprocess(text: str) -> str:
    """Collapse multiple whitespace characters to a single space.

    Intentionally minimal: MARBERT is a dialect-aware Arabic BERT model
    trained on raw social-media text.  Heavy normalisation (diacritic
    removal, Alef unification, etc.) degrades its sub-word representations.
    """
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Main ETL function
# ---------------------------------------------------------------------------

def load_and_prepare_marbert_data(
    tweets_path: Union[str, Path],
    replies_path: Union[str, Path],
    retweets_path: Union[str, Path],
) -> pd.DataFrame:
    """Load, deduplicate, and feature-engineer the dataset for MARBERT.

    Steps
    -----
    1. Load ``Tweets.txt`` via the shared ``load_tweets`` helper.
    2. Map propagation counts (replies / retweets) onto each tweet.
    3. Select relevant columns, apply ``minimal_preprocess`` on tweet text.
    4. Cast ``label`` to ``int`` (0 / 1).
    5. Deduplicate by original tweet text via groupby-agg:
       * ``label``       → first occurrence (all duplicates should share the same label)
       * ``num_replies`` → mean across duplicate rows
       * ``num_retweets``→ mean across duplicate rows
       * ``text``        → first preprocessed text
    6. Return a clean DataFrame with columns:
       ``text``, ``label``, ``num_replies``, ``num_retweets``.

    Parameters
    ----------
    tweets_path : str or Path
        Path to ``Tweets.txt`` (tab-separated, tweetID / tweetText / label).
    replies_path : str or Path
        Path to the propagation replies file parsed by ``parse_propagation_file``.
    retweets_path : str or Path
        Path to the propagation retweets file.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame ready for train/val/test splitting.

    Notes
    -----
    * ``num_replies`` and ``num_retweets`` are included but the best-performing
      MARBERT model uses ``use_propagation=False`` (text only).
    * Log-transform and StandardScaler are applied *inside the training loop*
      (fold-by-fold) to avoid data leakage.
    """
    tweets_path = Path(tweets_path)
    replies_path = Path(replies_path)
    retweets_path = Path(retweets_path)

    log.info("Loading tweets from %s", tweets_path)
    tweets_df = load_tweets(tweets_path)

    # --- propagation counts -------------------------------------------------
    log.info("Loading propagation counts …")
    reply_counts = load_propagation_counts(replies_path)
    retweet_counts = load_propagation_counts(retweets_path)

    tweets_df["num_replies"] = (
        tweets_df["tweetID"].map(reply_counts).fillna(0).astype(int)
    )
    tweets_df["num_retweets"] = (
        tweets_df["tweetID"].map(retweet_counts).fillna(0).astype(int)
    )

    # --- select + preprocess ------------------------------------------------
    df = tweets_df[["tweetText", "label", "num_replies", "num_retweets"]].copy()
    df["text"] = df["tweetText"].apply(minimal_preprocess)

    # NOTE: load_tweets casts label to bool; we need int for CrossEntropyLoss.
    df["label"] = df["label"].astype(int)

    # --- deduplicate --------------------------------------------------------
    # Group by the *original* tweetText so that duplicates with different IDs
    # are collapsed.  We keep the first label, mean propagation counts, and
    # first preprocessed text.
    agg_funcs = {
        "label": "first",
        "num_replies": "mean",
        "num_retweets": "mean",
        "text": "first",
    }
    df = df.groupby("tweetText", as_index=False).agg(agg_funcs)

    # After groupby-agg the label column may be float due to mixed types;
    # cast back to int explicitly.
    df["label"] = df["label"].astype(int)
    # Propagation counts may be fractional after mean-agg; keep as float for
    # log-transform downstream (log1p handles 0.0 correctly).

    df = df.drop(columns=["tweetText"]).reset_index(drop=True)

    log.info(
        "Dataset ready: %d rows | label distribution: %s",
        len(df),
        df["label"].value_counts().to_dict(),
    )
    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    _DATA = "ArCOV19-Rumors/tweet_verification"
    parser = argparse.ArgumentParser(
        description="Run the MARBERT ETL pipeline and save the output CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tweets",
        default=f"{_DATA}/Tweets.txt",
        help="Path to Tweets.txt",
    )
    parser.add_argument(
        "--replies",
        default=f"{_DATA}/propagation_networks/replies",
        help="Path to the propagation replies file.",
    )
    parser.add_argument(
        "--retweets",
        default=f"{_DATA}/propagation_networks/retweets",
        help="Path to the propagation retweets file.",
    )
    parser.add_argument(
        "--output",
        default="Rumors_Classifier/data/marbert_features.csv",
        help="Destination CSV path.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_marbert_data(
        tweets_path=args.tweets,
        replies_path=args.replies,
        retweets_path=args.retweets,
    )
    df.to_csv(output_path, index=False)
    log.info("Saved %d rows → %s", len(df), output_path)
