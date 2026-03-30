from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# CONFIG
# =========================
TRAIN_FILES = [
    Path("../Nina_DB1_CSV/Participant1/S1_A1_E1_export.csv"),
    Path("../Nina_DB1_CSV/Participant1/S1_A1_E2_export.csv"),
]

TEST_FILES = [
    Path("../Nina_DB1_CSV/Participant1/S1_A1_E3_export.csv"),
]

EMG_COLUMNS = [f"EMG_{i}" for i in range(1, 11)]
TIME_COLUMN = "Time"
LABEL_COLUMN = "restimulus"

WINDOW_MS = 200
OVERLAP_MS = 50
STEP_MS = WINDOW_MS - OVERLAP_MS  # 150 ms

RANDOM_STATE = 42


# =========================
# PREPROCESSING
# =========================
def infer_fs(time_vector: np.ndarray) -> float:
    dt = np.diff(time_vector)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]

    if len(dt) == 0:
        raise ValueError("Could not infer sampling frequency from Time column.")

    return float(1.0 / np.median(dt))


def remove_dc(emg: np.ndarray) -> np.ndarray:
    return emg - np.mean(emg, axis=0, keepdims=True)


def bandpass_filter(
    emg: np.ndarray,
    fs: float,
    lowcut: float = 20.0,
    highcut: float = 45.0,
    order: int = 4,
) -> np.ndarray:
    nyquist = fs / 2.0

    if highcut >= nyquist:
        raise ValueError(
            f"Invalid bandpass for fs={fs:.2f}. "
            f"highcut={highcut} must be less than Nyquist={nyquist:.2f}"
        )

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, emg, axis=0)


def zscore_normalize(emg: np.ndarray) -> np.ndarray:
    mean = np.mean(emg, axis=0, keepdims=True)
    std = np.std(emg, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (emg - mean) / std


def preprocess_emg(time_vector: np.ndarray, emg: np.ndarray) -> Tuple[np.ndarray, float]:
    fs = infer_fs(time_vector)
    emg = remove_dc(emg)
    emg = bandpass_filter(emg, fs=fs, lowcut=20.0, highcut=45.0, order=4)
    emg = zscore_normalize(emg)
    return emg, fs


# =========================
# FEATURE EXTRACTION
# =========================
def mav(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def waveform_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))


def zero_crossings(x: np.ndarray, threshold: float = 1e-3) -> float:
    x1 = x[:-1]
    x2 = x[1:]
    crossings = ((x1 * x2) < 0) & (np.abs(x1 - x2) >= threshold)
    return float(np.sum(crossings))


def slope_sign_changes(x: np.ndarray, threshold: float = 1e-3) -> float:
    x_prev = x[:-2]
    x_mid = x[1:-1]
    x_next = x[2:]

    ssc = (((x_mid - x_prev) * (x_mid - x_next)) > 0) & (
        (np.abs(x_mid - x_prev) >= threshold)
        | (np.abs(x_mid - x_next) >= threshold)
    )
    return float(np.sum(ssc))


def variance(x: np.ndarray) -> float:
    return float(np.var(x))


def extract_features(window: np.ndarray) -> np.ndarray:
    features = []

    for ch in range(window.shape[1]):
        signal = window[:, ch]
        features.extend([
            mav(signal),
            rms(signal),
            waveform_length(signal),
            zero_crossings(signal),
            slope_sign_changes(signal),
            variance(signal),
        ])

    return np.asarray(features, dtype=np.float32)


# =========================
# DATA HANDLING
# =========================
def load_emg_csv(file_path: Path):
    df = pd.read_csv(file_path)

    time_vector = df[TIME_COLUMN].to_numpy(dtype=np.float64)
    emg = df[EMG_COLUMNS].to_numpy(dtype=np.float32)
    labels = df[LABEL_COLUMN].to_numpy(dtype=np.int64)

    return time_vector, emg, labels


def window_features(
    emg: np.ndarray,
    labels: np.ndarray,
    fs: float,
):
    window_size = int(round(fs * WINDOW_MS / 1000.0))
    step_size = int(round(fs * STEP_MS / 1000.0))

    X, y = [], []

    if len(emg) < window_size:
        return np.empty((0, emg.shape[1] * 6)), np.empty((0,), dtype=np.int64)

    for start in range(0, len(emg) - window_size + 1, step_size):
        end = start + window_size

        window = emg[start:end]
        window_labels = labels[start:end]

        unique, counts = np.unique(window_labels, return_counts=True)
        label = int(unique[np.argmax(counts)])

        if label == 0:
            continue

        X.append(extract_features(window))
        y.append(label)

    return np.array(X), np.array(y, dtype=np.int64)


def build_dataset(files):
    X_all, y_all = [], []

    for f in files:
        print(f"Processing: {f}")

        t, emg, labels = load_emg_csv(f)
        emg, fs = preprocess_emg(t, emg)

        X, y = window_features(emg, labels, fs)

        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)

    if len(X_all) == 0:
        raise ValueError("No valid windows were generated.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    return X, y


# =========================
# MAIN
# =========================
def main():
    X_train, y_train = build_dataset(TRAIN_FILES)
    X_test, y_test = build_dataset(TEST_FILES)

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)
    print("Train classes:", np.unique(y_train))
    print("Test classes:", np.unique(y_test))

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
