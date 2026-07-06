"""
feature_pipeline.py
Reusable feature engineering pipeline for Arabic rumor classification.

Produces a feature DataFrame containing:
  - normalized_text : Arabic-normalised tweet text (used by TF-IDF)
  - char_len        : character count
  - num_hashtags    : hashtag count
  - num_mentions    : mention count
  - num_urls        : URL count
  - num_emojis      : emoji count
  - num_exclamations: exclamation-mark count
  - avg_word_length : char_len / word_count
  - num_replies     : propagation reply count
  - num_retweets    : propagation retweet count
  - label           : bool rumor label
"""

import re
import unicodedata
from pathlib import Path
from typing import Dict, Union

import emoji
import numpy as np
import pandas as pd
import preprocessor as p
from sklearn.base import BaseEstimator, TransformerMixin

from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)

from Rumors_Classifier.utils import parse_propagation_file


# ---------------------------------------------------------------------------
# Arabic text normaliser (sklearn-compatible transformer)
# ---------------------------------------------------------------------------

class ArabicTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Arabic text normalisation for NLP classification tasks.
    Suitable for rumour / fake-news detection.

    Parameters
    ----------
    remove_urls : bool
        Replace HTTP/www URLs with a space.
    remove_mentions : bool
        Replace @mentions with a space.
    remove_emojis : bool
        Strip Unicode emoji characters.
    remove_numbers : bool
        Strip digit sequences.
    remove_punctuation : bool
        Strip non-Arabic, non-word characters.
    reduce_repeated_chars : bool
        Collapse runs of 3+ identical characters to 2 (e.g. aaaa → aa).
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = False,
        remove_emojis: bool = False,
        remove_numbers: bool = True,
        remove_punctuation: bool = True,
        reduce_repeated_chars: bool = True,
    ):
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_emojis = remove_emojis
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.reduce_repeated_chars = reduce_repeated_chars

        # --- compile patterns once at construction time ---
        # Arabic diacritics / harakat
        self.diacritics_pattern = re.compile(
            r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]"
        )
        # Arabic tatweel (kashida)  \u0640
        self.tatweel_pattern = re.compile(r"\u0640")
        # URLs
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        # @mentions
        self.mention_pattern = re.compile(r"@\w+")
        # Hashtag symbol only (keep the word)
        self.hashtag_pattern = re.compile(r"#")
        # Repeated chars (3 or more)
        self.repeat_pattern = re.compile(r"(.)\1{2,}")
        # Digits
        self.number_pattern = re.compile(r"\d+")
        # Punctuation / non-Arabic (keep Arabic range + whitespace)
        self.punct_pattern = re.compile(r"[^\w\s\u0600-\u06FF]")
        # Emoji unicode ranges
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002700-\U000027BF"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        return [self._normalize_text(text) for text in X]

    # ------------------------------------------------------------------
    # Internal normalisation logic
    # ------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        # Unicode normalisation
        text = unicodedata.normalize("NFKC", text)

        if self.remove_urls:
            text = self.url_pattern.sub(" ", text)

        if self.remove_mentions:
            text = self.mention_pattern.sub(" ", text)

        # Strip # but keep the hashtag word
        text = self.hashtag_pattern.sub("", text)

        # Remove Arabic diacritics and tatweel
        text = self.diacritics_pattern.sub("", text)
        text = self.tatweel_pattern.sub("", text)

        # CAMeL Tools orthographic normalisation
        text = normalize_alef_ar(text)
        text = normalize_alef_maksura_ar(text)
        text = normalize_teh_marbuta_ar(text)

        if self.reduce_repeated_chars:
            text = self.repeat_pattern.sub(r"\1\1", text)

        if self.remove_emojis:
            text = self.emoji_pattern.sub("", text)

        if self.remove_numbers:
            text = self.number_pattern.sub("", text)

        if self.remove_punctuation:
            text = self.punct_pattern.sub(" ", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


# ---------------------------------------------------------------------------
# Low-level text feature extraction
# ---------------------------------------------------------------------------

def extract_text_features(text: str) -> pd.Series:
    """
    Extract low-level surface features from a *raw* tweet string.

    Note: features are extracted from the raw text (before normalisation)
    so counts of hashtags, mentions, URLs, and emojis are accurate.

    Returns
    -------
    pd.Series with index:
        char_len, word_len, num_hashtags, num_mentions,
        num_urls, num_emojis, num_exclamations
    """
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.EMOJI)
    parsed = p.parse(text)

    char_len = len(text)
    word_len = len(text.split())
    hashtags = len(parsed.hashtags) if parsed.hashtags else 0
    mentions = len(parsed.mentions) if parsed.mentions else 0
    urls = len(parsed.urls) if parsed.urls else 0
    emojis = sum(1 for ch in text if emoji.is_emoji(ch))
    exclamations = text.count("!")

    return pd.Series(
        [char_len, word_len, hashtags, mentions, urls, emojis, exclamations],
        index=[
            "char_len", "word_len", "num_hashtags", "num_mentions",
            "num_urls", "num_emojis", "num_exclamations",
        ],
    )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_tweets(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load raw tweets from a tab-separated file."""
    df = pd.read_csv(file_path, sep="\t")
    df["tweetID"] = df["tweetID"].astype(str)
    df["label"] = df["label"].astype(bool)
    return df


def load_propagation_counts(file_path: Union[str, Path]) -> Dict[str, int]:
    """Parse a propagation file (replies **or** retweets) into {tweetID: count}."""
    return parse_propagation_file(str(file_path))


# ---------------------------------------------------------------------------
# Full feature-engineering pipeline
# ---------------------------------------------------------------------------

def prepare_data(
    tweets_path: Union[str, Path],
    replies_path: Union[str, Path],
    retweets_path: Union[str, Path],
    text_preprocessor: ArabicTextPreprocessor = None,
) -> pd.DataFrame:
    """
    Steps
    -----
    1. Load tweets TSV.
    2. Track duplicate count (text_dup_count) then deduplicate on tweetText.
    3. Normalise Arabic text → ``normalized_text`` column.
    4. Extract surface text features (char_len, hashtags, …) from normalised text.
    5. Load propagation files; add num_replies, num_retweets.
    6. Compute avg_word_length = char_len / word_len (NaN-safe).
    7. Drop raw tweetText and intermediate word_len.

    Parameters
    ----------
    tweets_path   : Path to Tweets.txt (tab-separated).
    replies_path  : Path to propagation replies file.
    retweets_path : Path to propagation retweets file.
    text_preprocessor : Optional pre-built ArabicTextPreprocessor.
                        Defaults to ArabicTextPreprocessor() with default params.

    Returns
    -------
    pd.DataFrame ready for train/val/test splitting and modelling.
    """
    if text_preprocessor is None:
        text_preprocessor = ArabicTextPreprocessor()

    # 1. Load
    tweets = load_tweets(tweets_path)

    # 2. Duplicate count + deduplicate
    tweets["text_dup_count"] = tweets.groupby("tweetText")["tweetText"].transform("count")
    tweets = tweets.drop_duplicates(subset=["tweetText"], keep="first").copy()

    # 3. Normalise text
    tweets["normalized_text"] = text_preprocessor.transform(tweets["tweetText"].tolist())

    # 4. Surface features (extracted from normalised text to match notebook behaviour)
    feat_cols = ["char_len", "word_len", "num_hashtags", "num_mentions",
                 "num_urls", "num_emojis", "num_exclamations"]
    tweets[feat_cols] = tweets["normalized_text"].apply(extract_text_features)

    # 5. Propagation
    reply_counts = load_propagation_counts(replies_path)
    retweet_counts = load_propagation_counts(retweets_path)
    tweets["num_replies"] = tweets["tweetID"].map(reply_counts).fillna(0).astype(int)
    tweets["num_retweets"] = tweets["tweetID"].map(retweet_counts).fillna(0).astype(int)

    # 6. Average word length (avoid division-by-zero)
    tweets["avg_word_length"] = tweets["char_len"] / tweets["word_len"].replace(0, np.nan)

    # 7. Drop raw text and intermediate word_len
    tweets = tweets.drop(columns=["tweetText", "word_len"], errors="ignore")

    return tweets


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Arabic rumour feature pipeline and save output CSV."
    )
    parser.add_argument("--tweets", required=True, help="Path to Tweets.txt")
    parser.add_argument("--replies", required=True, help="Path to replies propagation file")
    parser.add_argument("--retweets", required=True, help="Path to retweets propagation file")
    parser.add_argument("--output", required=True, help="Destination CSV path")
    args = parser.parse_args()

    df = prepare_data(args.tweets, args.replies, args.retweets)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows → {args.output}")