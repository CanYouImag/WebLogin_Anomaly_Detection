import numpy as np
import pandas as pd


def random_split(X, y, val_frac=0.1, test_frac=0.2, seed=42, stratify=True):
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = np.arange(n)
    if stratify:
        split = _stratified_indices(y, val_frac + test_frac, rng)
    else:
        rng.shuffle(idx)
        split = idx[: int(n * (val_frac + test_frac))]
    temp_idx = np.sort(split)
    remaining = np.setdiff1d(idx, temp_idx, assume_unique=True)
    return _split_remaining(remaining, temp_idx, y, val_frac, test_frac, rng, stratify)


def _stratified_indices(y, frac, rng):
    idx = np.arange(len(y))
    classes = np.unique(y)
    chosen = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        k = max(1, int(round(len(c_idx) * frac)))
        chosen.append(rng.choice(c_idx, size=k, replace=False))
    return np.concatenate(chosen)


def _split_remaining(remaining, temp_idx, y, val_frac, test_frac, rng, stratify):
    val_k = int(round(len(temp_idx) * (val_frac / (val_frac + test_frac))))
    if stratify:
        y_temp = y[temp_idx]
        val_chosen = _stratified_indices(y_temp, val_k / len(y_temp), rng)
        val_idx = temp_idx[val_chosen]
    else:
        shuffled = temp_idx.copy()
        rng.shuffle(shuffled)
        val_idx = shuffled[:val_k]
    test_idx = np.setdiff1d(temp_idx, val_idx, assume_unique=True)
    return remaining, val_idx, test_idx


def time_split(X, timestamp_col, test_frac=0.2, val_frac=0.1, seed=42):
    ts = pd.to_datetime(X[timestamp_col])
    order = np.argsort(ts.to_numpy(), kind="stable")
    n = len(order)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = order[n_test + n_val:]
    val_idx = order[n_test:n_test + n_val]
    train_idx = order[:n_test]
    return train_idx, val_idx, test_idx


def split_arrays(X, y, mode="random", timestamp_col=None, val_frac=0.1,
                 test_frac=0.2, seed=42, stratify=True):
    if mode == "time":
        if timestamp_col is None or timestamp_col not in X.columns:
            raise ValueError("time split requires timestamp_col in X")
        train_idx, val_idx, test_idx = time_split(
            X, timestamp_col, test_frac=test_frac, val_frac=val_frac, seed=seed
        )
    else:
        train_idx, val_idx, test_idx = random_split(
            X, y, val_frac=val_frac, test_frac=test_frac, seed=seed, stratify=stratify
        )
    return train_idx, val_idx, test_idx


def to_numpy(X):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32)
    return np.asarray(X, dtype=np.float32)