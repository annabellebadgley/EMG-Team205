"""
lstm_emg_classifier.py
======================
LSTM-based hand-gesture classifier for NinaPro DB1 EMG data.

Pipeline
--------
1. Load & preprocess EMG signals (via signalprocessing.py)
2. Segment into overlapping windows per gesture repetition
3. Split by repetition (leave-one-repetition-out or hold-out)
4. Train a stacked bidirectional LSTM
5. Evaluate and report per-class accuracy

Usage (quick start — single subject, all exercises)
-----------------------------------------------------
    python lstm_emg_classifier.py \
        --csv S1_A1_E1_export.csv S1_A1_E2_export.csv S1_A1_E3_export.csv \
        --epochs 50

Requirements
------------
    pip install torch scikit-learn pandas numpy scipy tqdm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── make sure signalprocessing.py is importable ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from signalprocessing import process_emg_continuous

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

EMG_PREFIX = "EMG_"
LABEL_COL = "restimulus"       # refined labels (recommended over 'stimulus')
REP_COL   = "rerepetition"     # repetition index (0 = rest/transition)
TIME_COL  = "Time"


def load_and_preprocess(
    csv_paths: List[str | Path],
    *,
    bandpass_low: float  = 20.0,
    bandpass_high: float = 45.0,
    zscore: bool         = True,
    exclude_rest: bool   = True,
    exercise_offset: bool = True,
) -> pd.DataFrame:
    """
    Load one or more NinaPro export CSVs, run the signal-processing pipeline,
    and concatenate them into a single DataFrame.

    When exercise_offset=True the gesture labels from E2 and E3 are shifted so
    that all labels are globally unique across exercises (matching DB1 layout:
    E1 → 1-12, E2 → 13-29, E3 → 30-52, 0 = rest).
    """
    # DB1 exercise label ranges (gesture 0 = rest in every exercise)
    offsets = {0: 0, 1: 0, 2: 12, 3: 29}   # exercise index → label offset

    frames: List[pd.DataFrame] = []
    for i, p in enumerate(csv_paths):
        p = Path(p)
        df = pd.read_csv(p)

        emg_cols = [c for c in df.columns if c.startswith(EMG_PREFIX)]
        df_proc, _ = process_emg_continuous(
            df,
            emg_cols,
            time_col=TIME_COL,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            do_notch=False,          # fs=100 Hz → 50 Hz notch impossible
            zscore=zscore,
        )

        # Use only the processed EMG columns + metadata
        proc_cols = [c + "_proc" for c in emg_cols]
        keep = proc_cols + [LABEL_COL, REP_COL]
        df_out = df_proc[keep].copy()

        # Remap column names to canonical EMG_1 … EMG_10
        rename = {c + "_proc": c for c in emg_cols}
        df_out.rename(columns=rename, inplace=True)

        # Offset labels for multi-exercise concatenation
        if exercise_offset:
            offset = offsets.get(i + 1, 0)
            non_rest = df_out[LABEL_COL] != 0
            df_out.loc[non_rest, LABEL_COL] += offset

        frames.append(df_out)

    full = pd.concat(frames, ignore_index=True)

    if exclude_rest:
        full = full[full[LABEL_COL] != 0].reset_index(drop=True)

    return full


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Windowing
# ─────────────────────────────────────────────────────────────────────────────

def segment_windows(
    df: pd.DataFrame,
    emg_cols: List[str],
    *,
    window_ms: int   = 300,    # window length in milliseconds
    step_ms: int     = 100,    # step / hop in milliseconds
    fs: float        = 100.0,
    majority_vote: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over each (gesture, repetition) segment independently
    to avoid contaminating segment boundaries.

    Returns
    -------
    X : (N, window_len, n_channels)
    y : (N,) integer gesture labels
    """
    win_len  = int(round(window_ms * fs / 1000))
    step_len = int(round(step_ms   * fs / 1000))

    X_list: List[np.ndarray] = []
    y_list: List[int]        = []

    groups = df.groupby([LABEL_COL, REP_COL])
    for (label, _rep), grp in groups:
        if len(grp) < win_len:
            continue
        sig = grp[emg_cols].to_numpy(dtype=np.float32)
        labels = grp[LABEL_COL].to_numpy(dtype=int)

        start = 0
        while start + win_len <= len(sig):
            chunk = sig[start : start + win_len]
            chunk_labels = labels[start : start + win_len]
            if majority_vote:
                win_label = int(np.bincount(chunk_labels).argmax())
            else:
                win_label = int(label)
            X_list.append(chunk)
            y_list.append(win_label)
            start += step_len

    X = np.stack(X_list, axis=0)          # (N, T, C)
    y = np.array(y_list, dtype=int)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LSTM model
# ─────────────────────────────────────────────────────────────────────────────

class EMGLSTMClassifier(nn.Module):
    """
    Stacked bidirectional LSTM for EMG gesture classification.

    Architecture
    ------------
    Input  : (batch, time_steps, n_channels)
    BiLSTM layers  → hidden state at last time step
    Dropout
    FC head → num_classes logits
    """

    def __init__(
        self,
        n_channels: int,
        num_classes: int,
        *,
        hidden_size: int  = 128,
        num_layers: int   = 2,
        dropout: float    = 0.4,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.bidirectional  = bidirectional
        self.directions     = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size   = n_channels,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            bidirectional= bidirectional,
        )

        lstm_out_dim = hidden_size * self.directions
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        out, _ = self.lstm(x)           # (B, T, H*dirs)
        last    = out[:, -1, :]         # take last time step
        return self.head(last)          # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_true = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        preds = model(xb).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.numpy())
    preds_arr = np.array(all_preds)
    true_arr  = np.array(all_true)
    acc = (preds_arr == true_arr).mean()
    return acc, preds_arr, true_arr


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    X: np.ndarray,
    y_encoded: np.ndarray,
    *,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Random split into train / val / test."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))

    n_test = int(len(X) * test_split)
    n_val  = int(len(X) * val_split)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    def to_ds(i):
        Xt = torch.tensor(X[i], dtype=torch.float32)
        yt = torch.tensor(y_encoded[i], dtype=torch.long)
        return TensorDataset(Xt, yt)

    return to_ds(train_idx), to_ds(val_idx), to_ds(test_idx)


def run(
    csv_paths: List[str],
    *,
    window_ms: int   = 300,
    step_ms: int     = 100,
    epochs: int      = 50,
    batch_size: int  = 64,
    lr: float        = 1e-3,
    hidden_size: int = 128,
    num_layers: int  = 2,
    dropout: float   = 0.4,
    seed: int        = 42,
    device_str: str  = "auto",
    save_model: str | None = None,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── device ──
    if device_str == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    else:
        device = torch.device(device_str)
    print(f"\n[Device] {device}")

    # ── load & preprocess ──
    print("[1/5] Loading and preprocessing EMG data …")
    df = load_and_preprocess(csv_paths)
    emg_cols = [c for c in df.columns if c.startswith(EMG_PREFIX)]
    print(f"      {len(df):,} samples | {len(emg_cols)} channels | "
          f"{df[LABEL_COL].nunique()} gesture classes")

    # ── windowing ──
    print("[2/5] Segmenting into windows …")
    X, y_raw = segment_windows(df, emg_cols, window_ms=window_ms, step_ms=step_ms)
    print(f"      X shape: {X.shape}  (windows × time × channels)")

    # ── label encoding ──
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"      {num_classes} classes after encoding")

    # ── datasets & loaders ──
    print("[3/5] Building train / val / test splits …")
    train_ds, val_ds, test_ds = build_datasets(X, y_enc, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"      train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")

    # ── model ──
    print("[4/5] Building LSTM model …")
    model = EMGLSTMClassifier(
        n_channels   = X.shape[2],
        num_classes  = num_classes,
        hidden_size  = hidden_size,
        num_layers   = num_layers,
        dropout      = dropout,
        bidirectional= True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Parameters: {n_params:,}")
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── training ──
    print(f"\n[5/5] Training for {epochs} epochs …\n")
    best_val_acc = 0.0
    best_state   = None

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="ep"):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            tqdm.write(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"best_val={best_val_acc:.4f}"
            )

    # ── test evaluation ──
    print("\n── Final Test Evaluation ──")
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_acc, test_preds, test_true = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    class_names = [str(le.inverse_transform([c])[0]) for c in range(num_classes)]
    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, target_names=class_names, zero_division=0))

    # ── optional model save ──
    if save_model:
        save_path = Path(save_model)
        torch.save(
            {
                "model_state_dict": best_state,
                "label_encoder_classes": le.classes_,
                "hparams": {
                    "n_channels":   X.shape[2],
                    "num_classes":  num_classes,
                    "hidden_size":  hidden_size,
                    "num_layers":   num_layers,
                    "dropout":      dropout,
                    "window_ms":    window_ms,
                    "step_ms":      step_ms,
                },
            },
            save_path,
        )
        print(f"\nModel saved to {save_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Command Line Interface
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LSTM EMG gesture classifier for NinaPro DB1"
    )
    parser.add_argument(
        "--csv", nargs="+", required=True,
        help="One or more NinaPro export CSVs (e.g., S1_A1_E1_export.csv …)"
    )
    parser.add_argument("--window_ms",  type=int,   default=300,   help="Window length in ms (default 300)")
    parser.add_argument("--step_ms",    type=int,   default=100,   help="Hop / stride in ms (default 100)")
    parser.add_argument("--epochs",     type=int,   default=50,    help="Training epochs (default 50)")
    parser.add_argument("--batch_size", type=int,   default=64,    help="Batch size (default 64)")
    parser.add_argument("--lr",         type=float, default=1e-3,  help="Learning rate (default 1e-3)")
    parser.add_argument("--hidden",     type=int,   default=128,   help="LSTM hidden size (default 128)")
    parser.add_argument("--layers",     type=int,   default=2,     help="LSTM layers (default 2)")
    parser.add_argument("--dropout",    type=float, default=0.4,   help="Dropout rate (default 0.4)")
    parser.add_argument("--device",     type=str,   default="auto",help="Device: auto | cpu | cuda | mps")
    parser.add_argument("--save",       type=str,   default=None,  help="Path to save best model checkpoint")
    parser.add_argument("--seed",       type=int,   default=42,    help="Random seed (default 42)")

    args = parser.parse_args()

    run(
        csv_paths   = args.csv,
        window_ms   = args.window_ms,
        step_ms     = args.step_ms,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        hidden_size = args.hidden,
        num_layers  = args.layers,
        dropout     = args.dropout,
        device_str  = args.device,
        save_model  = args.save,
        seed        = args.seed,
    )


if __name__ == "__main__":
    main()