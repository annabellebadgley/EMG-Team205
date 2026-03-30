import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


# =========================================
# Helper functions
# =========================================

def infer_fs_from_time(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("Could not infer sampling frequency from time column.")
    return 1.0 / np.median(dt)


def zscore_per_channel(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (x - mean) / std


def notch_filter(x: np.ndarray, fs: float, notch_hz: float = 50.0, q: float = 30.0) -> np.ndarray:
    from scipy.signal import iirnotch

    b, a = iirnotch(w0=notch_hz, Q=q, fs=fs)
    return filtfilt(b, a, x, axis=0)


# =========================================
# Your processing function
# =========================================

def process_emg_continuous(
    df: pd.DataFrame,
    emg_cols: list[str],
    *,
    fs: float | None = None,
    time_col: str = "Time",
    do_highpass: bool = False,
    highpass_hz: float = 5.0,
    do_lowpass: bool = False,
    lowpass_hz: float = 45.0,
    do_notch: bool = False,
    notch_hz: float = 50.0,
    notch_q: float = 30.0,
    zscore: bool = True,
) -> tuple[pd.DataFrame, float]:
    out = df.copy()

    if fs is None:
        if time_col not in out.columns:
            raise ValueError(f"Time column '{time_col}' not found.")
        t = pd.to_numeric(out[time_col], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(t)):
            raise ValueError(f"Time column '{time_col}' contains NaNs/non-numeric values.")
        fs = infer_fs_from_time(t)

    x = out[emg_cols].to_numpy(dtype=float)

    # 1) DC removal
    x = x - np.mean(x, axis=0, keepdims=True)

    nyq = 0.5 * fs

    # 2) Optional high-pass
    if do_highpass:
        if not (0 < highpass_hz < nyq):
            raise ValueError(f"Invalid highpass_hz={highpass_hz} for fs={fs}")
        b, a = butter(4, highpass_hz / nyq, btype="highpass")
        x = filtfilt(b, a, x, axis=0)

    # 3) Optional low-pass
    if do_lowpass:
        if not (0 < lowpass_hz < nyq):
            raise ValueError(f"Invalid lowpass_hz={lowpass_hz} for fs={fs}")
        b, a = butter(4, lowpass_hz / nyq, btype="lowpass")
        x = filtfilt(b, a, x, axis=0)

    # 4) Optional notch
    if do_notch:
        if notch_hz >= nyq:
            raise ValueError(
                f"Notch {notch_hz} Hz is >= Nyquist ({nyq:.2f} Hz). Disable notch."
            )
        x = notch_filter(x, fs=fs, notch_hz=notch_hz, q=notch_q)

    # 5) Normalize
    if zscore:
        x = zscore_per_channel(x)

    for j, c in enumerate(emg_cols):
        out[f"{c}_proc"] = x[:, j]

    return out, float(fs)


# =========================================
# Batch processing
# =========================================

ROOT_DIR = Path("/workspaces/EMG-Team205/Nina_DB1_CSV")

# Set these how you want
PROCESS_KWARGS = {
    "fs": None,              # infer from Time column
    "time_col": "Time",
    "do_highpass": False,
    "highpass_hz": 5.0,
    "do_lowpass": False,
    "lowpass_hz": 45.0,
    "do_notch": False,
    "notch_hz": 50.0,
    "notch_q": 30.0,
    "zscore": True,
}


def find_emg_columns(df: pd.DataFrame) -> list[str]:
    emg_cols = [c for c in df.columns if c.startswith("EMG_")]
    if not emg_cols:
        raise ValueError("No EMG columns found. Expected columns like EMG_1 ... EMG_10")
    return emg_cols


def output_name_for(csv_path: Path) -> Path:
    name = csv_path.name

    if name.endswith("_export.csv"):
        new_name = name.replace("_export.csv", "_processed.csv")
    else:
        new_name = csv_path.stem + "_processed.csv"

    return csv_path.with_name(new_name)


def process_one_file(csv_path: Path) -> None:
    print(f"\nProcessing: {csv_path}")

    df = pd.read_csv(csv_path)
    emg_cols = find_emg_columns(df)

    processed_df, fs = process_emg_continuous(
        df,
        emg_cols,
        **PROCESS_KWARGS,
    )

    out_path = output_name_for(csv_path)
    processed_df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Inferred fs: {fs:.3f} Hz")
    print(f"Processed columns added: {[f'{c}_proc' for c in emg_cols]}")


def main():
    csv_files = sorted(ROOT_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found under {ROOT_DIR}")
        return

    for csv_path in csv_files:
        # skip already processed files
        if csv_path.name.endswith("_processed.csv"):
            continue

        try:
            process_one_file(csv_path)
        except Exception as e:
            print(f"FAILED: {csv_path}")
            print(f"Reason: {e}")


if __name__ == "__main__":
    main()