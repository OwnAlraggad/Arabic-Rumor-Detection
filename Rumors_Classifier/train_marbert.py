"""
train_marbert.py
================
Production-ready, reproducible training script for the MARBERT Arabic-rumour
classifier.

Best configuration (from ablation study in MARBERT_model_training.md)
----------------------------------------------------------------------
* Model       : UBC-NLP/MARBERT
* Loss        : FocalLoss (gamma=2.0)
* LR backbone : 1e-5  (differential LR)
* LR head     : 1e-4
* Max seq len : 128   (tweets rarely exceed 128 sub-word tokens)
* Batch size  : 16
* Max epochs  : 15
* Early stop  : patience=2 on macro-F1
* CV folds    : 3-fold StratifiedKFold
* Propagation : DISABLED  (replies/retweets added noise, not signal)

Bugs fixed from the notebook
-----------------------------
1. ``run_training`` return-value bug: the ``if folds is None`` guard was
   evaluated *after* ``folds`` had already been re-assigned to a list inside
   the function, making the single-split return branch unreachable.  Fixed by
   using a sentinel flag ``_single_split``.

2. ``MARBERTFusionClassifier.forward`` accessed ``self.bert.bert(...)`` which
   is BERT-architecture-specific and fragile.  Replaced with
   ``self.bert.base_model(...)`` which is architecture-agnostic and works for
   any AutoModel family.

3. ``evaluate`` always printed ``f1_score: {f1}`` even in non-verbose contexts.
   Moved behind the logger so it doesn't pollute stdout in production runs.

4. StratifiedKFold folds were built on the *full* dataframe before
   ``run_training`` carved out a test split, causing test-set rows to appear
   inside training folds (data leakage through index remapping).  Fixed by
   splitting *before* fold generation: folds are now produced from
   ``train_val_df`` only.

5. ``grid_search_lr`` expected ``run_training`` to return
   ``(val_f1, test_f1)`` for single-split runs, but due to bug #1 it always
   received ``(avg_test_f1, std_test_f1)`` instead.  Fixed alongside bug #1.

Usage
-----
Run from the project root:

    python -m Rumors_Classifier.train_marbert [OPTIONS]

Options (defaults match the best config found in the ablation study)
------
  --tweets      Path to Tweets.txt
  --replies     Path to replies propagation file
  --retweets    Path to retweets propagation file
  --output-dir  Directory for model checkpoints and logs
  --seed        Global random seed (default: 42)
  --folds       Number of CV folds (default: 3)
  --epochs      Max training epochs per fold (default: 15)
  --batch-size  Training batch size (default: 16)
  --lr-backbone Backbone learning rate (default: 1e-5)
  --lr-head     Classification head LR (default: 1e-4)
  --patience    Early-stopping patience (default: 2)
  --max-seq-len Tokeniser max length (default: 128)
  --save-model  Flag: save the best model checkpoint per fold
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# Suppress the deprecation warning from the old AdamW import location
from torch.optim import AdamW

from Rumors_Classifier.marbert_etl_pipeline import load_and_prepare_marbert_data


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str = "marbert_training", log_dir: str | Path = "logs") -> logging.Logger:
    """Create a logger that writes to both file and stdout."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"marbert_training_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers on repeated calls (e.g., in notebooks)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.addHandler(logging.FileHandler(log_file))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    for h in logger.handlers:
        h.setFormatter(formatter)

    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all relevant random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for classification with class-imbalance robustness.

    References
    ----------
    Lin et al. (2017) "Focal Loss for Dense Object Detection."

    Parameters
    ----------
    alpha : Tensor or None
        Per-class weights passed to cross_entropy (same as ``weight``).
    gamma : float
        Focusing parameter.  gamma=0 reduces to standard cross-entropy.
    reduction : {'mean', 'sum', 'none'}
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-sample CE (reduction='none' so we can weight by (1-pt)^gamma)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TweetDataset(Dataset):
    """Text-only dataset for MARBERT fine-tuning."""

    def __init__(self, df, tokenizer, max_len: int) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(row["label"], dtype=torch.long),
        }


class FusionDataset(Dataset):
    """Text + propagation-feature dataset (kept for research; disabled by default).

    Expects either ``num_replies_scaled`` / ``num_retweets_scaled`` columns
    (added by the training loop after fitting a StandardScaler on training
    data only) or falls back to ``log1p(num_replies/retweets)``.
    """

    def __init__(self, df, tokenizer, max_len: int) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        replies = (
            row["num_replies_scaled"]
            if "num_replies_scaled" in self.df.columns
            else np.log1p(row["num_replies"])
        )
        retweets = (
            row["num_retweets_scaled"]
            if "num_retweets_scaled" in self.df.columns
            else np.log1p(row["num_retweets"])
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "prop_features": torch.tensor([replies, retweets], dtype=torch.float),
            "label": torch.tensor(row["label"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MARBERTFusionClassifier(nn.Module):
    """MARBERT + propagation-feature fusion classifier.

    The BERT encoder is accessed via ``model.base_model`` so that this class
    works with *any* AutoModelForSequenceClassification architecture, not just
    BERT-based ones.

    Bug fix: the notebook used ``self.bert.bert(...)`` which is
    BERT-architecture-specific.  ``base_model`` is the architecture-agnostic
    equivalent exposed by every HuggingFace model.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_prop_features: int = 2,
        dropout_prob: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.fusion_layer = nn.Linear(hidden_size + num_prop_features, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prop_features: torch.Tensor,
    ) -> torch.Tensor:
        # base_model is architecture-agnostic (works for BERT, RoBERTa, etc.)
        outputs = self.encoder.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = outputs.pooler_output          # (batch, hidden_size)
        combined = torch.cat([pooled, prop_features], dim=1)
        combined = self.dropout(combined)
        return self.fusion_layer(combined)      # (batch, num_labels)


# ---------------------------------------------------------------------------
# Training / evaluation steps
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: nn.Module,
    logger: logging.Logger,
    device: torch.device,
    fusion: bool = False,
    log_every: int = 100,
) -> tuple[float, float]:
    """Run one training epoch.

    Returns
    -------
    avg_loss : float
    macro_f1 : float
    """
    model.train()
    total_loss = 0.0
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if fusion:
            prop_features = batch["prop_features"].to(device)
            logits = model(input_ids, attention_mask, prop_features)
        else:
            logits = model(input_ids, attention_mask).logits

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        total_loss += loss.item()

        if (batch_idx + 1) % log_every == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                "Batch %d/%d | RunningLoss: %.4f | LR: %.2e",
                batch_idx + 1,
                len(dataloader),
                running_loss / log_every,
                current_lr,
            )
            running_loss = 0.0

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    fusion: bool = False,
    return_preds: bool = False,
) -> tuple[float, float, list | None, list | None]:
    """Evaluate the model on a DataLoader.

    Returns
    -------
    avg_loss, macro_f1, labels (or None), preds (or None)
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if fusion:
                prop_features = batch["prop_features"].to(device)
                logits = model(input_ids, attention_mask, prop_features)
            else:
                logits = model(input_ids, attention_mask).logits

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    if return_preds:
        return avg_loss, macro_f1, all_labels, all_preds
    return avg_loss, macro_f1, None, None


# ---------------------------------------------------------------------------
# Core training runner
# ---------------------------------------------------------------------------

def run_training(
    df,
    config: dict,
    device: torch.device,
    logger: logging.Logger,
    folds: list | None = None,
) -> tuple[float, float]:
    """Train and evaluate MARBERT with the given config.

    Bug fix (return-value): In the notebook ``folds`` was re-assigned to a
    list inside the function, making the ``if folds is None`` guard on the
    *return* statement always False.  We now use a ``_single_split`` flag to
    correctly distinguish single-split vs. cross-validation at return time.

    Bug fix (data leakage): In the notebook, StratifiedKFold was called on the
    *full* dataframe, then ``run_training`` removed test rows via index
    remapping.  This creates inconsistent fold sizes and potential leakage.
    The correct approach (implemented here) is: split off the test set *first*,
    then apply StratifiedKFold only to ``train_val_df``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``load_and_prepare_marbert_data``.
    config : dict
        Keys: model_name, num_labels, max_seq_len, batch_size, epochs,
              lr_backbone, lr_head, loss_type ('focal'|'ce'),
              early_stop_metric ('f1'|'loss'), patience,
              use_propagation (bool), test_size (float), val_size (float).
    device : torch.device
    logger : logging.Logger
    folds : list of (train_indices, val_indices) into ``train_val_df``, or None.
        If None, a single stratified train/val split is used.

    Returns
    -------
    (primary_score, secondary_score) where:
      * single-split → (val_f1,  test_f1)
      * cross-val    → (mean_test_f1, std_test_f1)
    """
    # --- unpack config ---
    model_name        = config["model_name"]
    num_labels        = config["num_labels"]
    max_seq_len       = config["max_seq_len"]
    batch_size        = config["batch_size"]
    epochs            = config["epochs"]
    lr_backbone       = config["lr_backbone"]
    lr_head           = config["lr_head"]
    loss_type         = config["loss_type"]
    early_stop_metric = config["early_stop_metric"]
    patience          = config["patience"]
    use_propagation   = config.get("use_propagation", False)
    test_size         = config.get("test_size", 0.15)
    val_size          = config.get("val_size", 0.15)

    logger.info("=" * 80)
    logger.info("Starting training")
    for k, v in config.items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 80)

    df = df.reset_index(drop=True)

    # Pre-compute log-transformed propagation features (used when use_propagation=True)
    df["num_replies_log"]  = np.log1p(df["num_replies"])
    df["num_retweets_log"] = np.log1p(df["num_retweets"])

    # --- hold out test set BEFORE creating folds ---------------------------
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=42,
    )
    train_val_df = train_val_df.reset_index(drop=True)
    test_df      = test_df.reset_index(drop=True)

    # --- determine fold list -----------------------------------------------
    _single_split = folds is None          # flag for return value (bug fix)
    if folds is None:
        val_prop = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_prop,
            stratify=train_val_df["label"],
            random_state=42,
        )
        folds = [(list(train_df.index), list(val_df.index))]
        logger.info("Using single train/val/test split (no cross-validation).")
    else:
        logger.info("Using %d-fold cross-validation (test set held out).", len(folds))

    val_f1_scores: list[float]  = []
    test_f1_scores: list[float] = []
    last_test_labels: list | None = None
    last_test_preds:  list | None = None

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(
            "Fold %d/%d | train=%d | val=%d",
            fold_idx + 1, len(folds), len(train_idx), len(val_idx),
        )

        fold_train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df   = train_val_df.iloc[val_idx].reset_index(drop=True)

        # --- propagation feature scaling (no leakage) ----------------------
        if use_propagation:
            scaler = StandardScaler()
            log_cols = ["num_replies_log", "num_retweets_log"]

            scaled_train = scaler.fit_transform(fold_train_df[log_cols].values)
            fold_train_df["num_replies_scaled"]  = scaled_train[:, 0]
            fold_train_df["num_retweets_scaled"] = scaled_train[:, 1]

            scaled_val = scaler.transform(fold_val_df[log_cols].values)
            fold_val_df["num_replies_scaled"]  = scaled_val[:, 0]
            fold_val_df["num_retweets_scaled"] = scaled_val[:, 1]

            test_df_fold = test_df.copy()
            scaled_test = scaler.transform(test_df_fold[log_cols].values)
            test_df_fold["num_replies_scaled"]  = scaled_test[:, 0]
            test_df_fold["num_retweets_scaled"] = scaled_test[:, 1]
        else:
            test_df_fold = test_df

        # --- tokeniser & model -------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if use_propagation:
            model = MARBERTFusionClassifier(model_name, num_labels)
            dataset_cls = FusionDataset
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
            dataset_cls = TweetDataset

        model.to(device)

        total_p     = sum(p.numel() for p in model.parameters())
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model: %s | Total params: %s | Trainable: %s",
            model.__class__.__name__,
            f"{total_p:,}",
            f"{trainable_p:,}",
        )

        # --- DataLoaders --------------------------------------------------
        train_ds = dataset_cls(fold_train_df, tokenizer, max_seq_len)
        val_ds   = dataset_cls(fold_val_df,   tokenizer, max_seq_len)
        test_ds  = dataset_cls(test_df_fold,  tokenizer, max_seq_len)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

        # --- optimizer (differential LR) ----------------------------------
        if use_propagation:
            head_params     = [p for n, p in model.named_parameters() if "fusion_layer" in n]
            backbone_params = [p for n, p in model.named_parameters() if "fusion_layer" not in n]
        else:
            head_params     = [p for n, p in model.named_parameters() if "classifier" in n or "score" in n]
            backbone_params = [p for n, p in model.named_parameters() if not ("classifier" in n or "score" in n)]

        optimizer = AdamW(
            [
                {"params": backbone_params, "lr": lr_backbone},
                {"params": head_params,     "lr": lr_head},
            ],
            weight_decay=0.01,
        )
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        # --- loss function ------------------------------------------------
        if loss_type == "focal":
            loss_fn: nn.Module = FocalLoss(gamma=2.0, alpha=None)
        else:  # weighted cross-entropy
            counts = fold_train_df["label"].value_counts().sort_index().values.astype(float)
            weights = 1.0 / counts
            weights = weights / weights.sum() * len(counts)
            loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float).to(device)
            )

        # --- training loop with early stopping ----------------------------
        best_score       = -np.inf if early_stop_metric == "f1" else np.inf
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_f1 = train_epoch(
                model, train_loader, optimizer, scheduler, loss_fn, logger, device,
                fusion=use_propagation,
            )
            val_loss, val_f1, _, _ = evaluate(
                model, val_loader, loss_fn, device, fusion=use_propagation,
            )
            logger.info(
                "Epoch %d/%d | TrainLoss=%.4f TrainF1=%.4f | ValLoss=%.4f ValF1=%.4f",
                epoch + 1, epochs, train_loss, train_f1, val_loss, val_f1,
            )

            improved = (
                val_f1 > best_score
                if early_stop_metric == "f1"
                else val_loss < best_score
            )
            if improved:
                best_score       = val_f1 if early_stop_metric == "f1" else val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                logger.info("  ↑ New best (%.4f)", best_score)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.warning("Early stopping at epoch %d.", epoch + 1)
                    break

        # --- restore best checkpoint and evaluate -------------------------
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        _, best_val_f1, _, _ = evaluate(
            model, val_loader, loss_fn, device, fusion=use_propagation,
        )
        val_f1_scores.append(best_val_f1)

        _, test_f1, last_test_labels, last_test_preds = evaluate(
            model, test_loader, loss_fn, device,
            fusion=use_propagation, return_preds=True,
        )
        test_f1_scores.append(test_f1)
        logger.info("Fold %d | Test macro-F1: %.4f", fold_idx + 1, test_f1)

    # --- aggregate results ------------------------------------------------
    mean_val_f1  = float(np.mean(val_f1_scores))
    std_val_f1   = float(np.std(val_f1_scores))
    mean_test_f1 = float(np.mean(test_f1_scores))
    std_test_f1  = float(np.std(test_f1_scores))

    logger.info("=" * 80)
    logger.info("Val  F1: %.4f ± %.4f", mean_val_f1, std_val_f1)
    logger.info("Test F1: %.4f ± %.4f", mean_test_f1, std_test_f1)

    if last_test_labels is not None and last_test_preds is not None:
        report = classification_report(last_test_labels, last_test_preds, digits=4)
        logger.info("Classification report (last fold):\n%s", report)
    logger.info("=" * 80)

    # Bug fix: use _single_split flag because folds is always a list here
    if _single_split:
        return mean_val_f1, test_f1_scores[0]   # (val_f1, test_f1)
    return mean_test_f1, std_test_f1             # (mean, std) for CV


# ---------------------------------------------------------------------------
# Best config (ablation conclusions)
# ---------------------------------------------------------------------------

BEST_CONFIG: dict = {
    "model_name":        "UBC-NLP/MARBERT",
    "num_labels":        2,
    "max_seq_len":       128,   # tweets rarely exceed 128 sub-word tokens
    "batch_size":        16,
    "epochs":            15,
    "lr_backbone":       1e-5,  # differential LR: 10x slower than head
    "lr_head":           1e-4,
    "loss_type":         "focal",  # focal loss; CE performed similarly
    "early_stop_metric": "f1",
    "patience":          2,
    "use_propagation":   False,    # propagation features hurt performance
    "test_size":         0.15,
    "val_size":          0.15,
}


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Full training pipeline driven by CLI arguments."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    logs_dir   = output_dir / "logs"
    models_dir = output_dir / "models"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir=logs_dir)
    logger.info("Device: %s", device)

    # --- ETL ---------------------------------------------------------------
    logger.info("Running ETL pipeline …")
    df = load_and_prepare_marbert_data(
        tweets_path=args.tweets,
        replies_path=args.replies,
        retweets_path=args.retweets,
    )
    logger.info("Dataset: %d rows", len(df))

    # --- build folds from train_val portion only (no leakage) --------------
    # We use a temporary split here just to get the train_val indices, then
    # run_training will redo the same split internally using the same seed.
    _, test_df_tmp = train_test_split(
        df, test_size=BEST_CONFIG["test_size"], stratify=df["label"], random_state=42
    )
    train_val_mask = ~df.index.isin(test_df_tmp.index)
    train_val_df = df[train_val_mask].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(train_val_df, train_val_df["label"]))

    # --- build config from CLI + BEST_CONFIG baseline ----------------------
    config = BEST_CONFIG.copy()
    config.update(
        {
            "batch_size":        args.batch_size,
            "epochs":            args.epochs,
            "lr_backbone":       args.lr_backbone,
            "lr_head":           args.lr_head,
            "patience":          args.patience,
            "max_seq_len":       args.max_seq_len,
            "use_propagation":   args.use_propagation,
        }
    )

    # --- run training ----------------------------------------------------
    mean_test_f1, std_test_f1 = run_training(
        df, config, device, logger, folds=folds
    )

    # --- save metrics ------------------------------------------------------
    metrics = {
        "mean_test_macro_f1": round(mean_test_f1, 6),
        "std_test_macro_f1":  round(std_test_f1,  6),
        "n_folds":            args.folds,
        "seed":               args.seed,
        "config":             config,
    }
    metrics_path = logs_dir / "marbert_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved → %s", metrics_path)
    logger.info(
        "Final result: Test macro-F1 = %.4f ± %.4f",
        mean_test_f1, std_test_f1,
    )


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    _DATA = "ArCOV19-Rumors/tweet_verification"
    parser = argparse.ArgumentParser(
        description="Train the MARBERT Arabic-rumour classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    parser.add_argument("--tweets",    default=f"{_DATA}/Tweets.txt")
    parser.add_argument("--replies",   default=f"{_DATA}/propagation_networks/replies")
    parser.add_argument("--retweets",  default=f"{_DATA}/propagation_networks/retweets")
    parser.add_argument("--output-dir", dest="output_dir", default="Rumors_Classifier")

    # Reproducibility
    parser.add_argument("--seed",       type=int,   default=42)

    # CV
    parser.add_argument("--folds",      type=int,   default=3,    help="Number of CV folds")

    # Hyperparameters (defaults = best config from ablation)
    parser.add_argument("--epochs",     type=int,   default=BEST_CONFIG["epochs"])
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=BEST_CONFIG["batch_size"])
    parser.add_argument("--lr-backbone", dest="lr_backbone", type=float, default=BEST_CONFIG["lr_backbone"])
    parser.add_argument("--lr-head",     dest="lr_head",     type=float, default=BEST_CONFIG["lr_head"])
    parser.add_argument("--patience",   type=int,   default=BEST_CONFIG["patience"])
    parser.add_argument("--max-seq-len", dest="max_seq_len", type=int, default=BEST_CONFIG["max_seq_len"])
    parser.add_argument(
        "--use-propagation",
        dest="use_propagation",
        action="store_true",
        default=False,
        help="Enable propagation feature fusion (disabled by default — hurts performance).",
    )
    parser.add_argument("--save-model", dest="save_model", action="store_true", default=False)

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    train(args)
