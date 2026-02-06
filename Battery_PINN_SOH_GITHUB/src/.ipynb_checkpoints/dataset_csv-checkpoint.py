# src/dataset_csv.py
# Minimal paired-cycle Dataset for the preprocessed XJTU CSVs you uploaded.
# Returns: x1, x2, y1, y2 where (x1,y1) is cycle t and (x2,y2) is cycle t+1.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def _3_sigma(ser: np.ndarray) -> np.ndarray:
    mu = np.nanmean(ser)
    sd = np.nanstd(ser)
    rule = (mu - 3 * sd > ser) | (mu + 3 * sd < ser)
    return np.arange(ser.shape[0])[rule]

#def delete_3_sigma(df: pd.DataFrame) -> pd.DataFrame:
    # Make sure "nan"/"inf"/"" strings become real NaN
    #df = df.apply(pd.to_numeric, errors="coerce")

    #df = df.replace([np.inf, -np.inf], np.nan)
    #df = df.dropna()
   # df = df.reset_index(drop=True)

   # out_index = []
   # for col in df.columns:
      #  idx = _3_sigma(df[col].to_numpy())
       # out_index.extend(idx.tolist())

   # out_index = list(set(out_index))
    #df = df.drop(out_index, axis=0)
    #df = df.reset_index(drop=True)
    #return df


@dataclass
class NormalizationStats:
    """Per-feature min/max for mapping features to [-1, 1]."""
    xmin: np.ndarray  # shape (D,)
    xmax: np.ndarray  # shape (D,)


class PairedCycleCSVDataset(Dataset):
    """
    Dataset over one or more battery CSV feature tables.

    Each CSV is assumed to be "preprocessed feature table":
      - 16 feature columns
      - last column is target 'capacity' (or SOH-like value)
    We add cycle_index as an extra feature unless you disable it.

    Each sample is a consecutive-cycle pair:
      (x_t, x_{t+1}, y_t, y_{t+1})
    """

    def __init__(
        self,
        csv_paths: List[str],
        *,
        target_col: str = "capacity",
        add_cycle_index: bool = True,
        normalize_x: bool = True,
        normalization: str = "minmax_-1_1",  # or "zscore"
        stats: Optional[NormalizationStats] = None,  # pass to reuse train stats for val/test
        dtype: torch.dtype = torch.float32,
    ):
        if not csv_paths:
            raise ValueError("csv_paths must be a non-empty list of paths.")

        if normalization not in ("minmax_-1_1", "zscore"):
            raise ValueError("normalization must be 'minmax_-1_1' or 'zscore'.")

        self.csv_paths = csv_paths
        self.target_col = target_col
        self.add_cycle_index = add_cycle_index
        self.normalize_x = normalize_x
        self.normalization = normalization
        self.dtype = dtype

        # Build paired arrays across all batteries
        x1_list, x2_list, y1_list, y2_list = [], [], [], []

        # We will compute normalization stats from the concatenated X if stats not provided.
        all_x_for_stats = []

        for p in csv_paths:
            df = pd.read_csv(p)
        
            # --- PAPER: insert "cycle index" column BEFORE cleaning ---
            if add_cycle_index and ("cycle index" not in df.columns):
                df.insert(df.shape[1] - 1, "cycle index", np.arange(df.shape[0], dtype=np.int64))
        
            # --- PAPER: clean rows (inf->nan, dropna, 3-sigma outlier rows) ---
            df = delete_3_sigma(df)
        
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in {p}. "
                                 f"Columns: {list(df.columns)}")
        
            # Features = all columns except target
            feature_cols = [c for c in df.columns if c != target_col]
            X = df[feature_cols].to_numpy(dtype=np.float32)              # (N, D)
            y = df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1) # (N, 1)
        
            # Need at least 2 cycles to form one pair
            if X.shape[0] < 2:
                continue
        
            x1 = X[:-1, :]
            x2 = X[1:, :]
            y1 = y[:-1, :]
            y2 = y[1:, :]
        
            x1_list.append(x1)
            x2_list.append(x2)
            y1_list.append(y1)
            y2_list.append(y2)
        
            all_x_for_stats.append(x1)
            all_x_for_stats.append(x2)


        if not x1_list:
            raise ValueError("No usable samples were created. Check CSV contents (need >=2 rows each).")

        self.x1 = np.concatenate(x1_list, axis=0)  # (M, D)
        self.x2 = np.concatenate(x2_list, axis=0)
        self.y1 = np.concatenate(y1_list, axis=0)  # (M, 1)
        self.y2 = np.concatenate(y2_list, axis=0)

        # Normalize X if requested
        self.stats: Optional[NormalizationStats] = None
        if normalize_x:
            if stats is None:
                X_all = np.concatenate(all_x_for_stats, axis=0)
                if normalization == "minmax_-1_1":
                    xmin = X_all.min(axis=0)
                    xmax = X_all.max(axis=0)
                    # avoid divide-by-zero if a feature is constant
                    xmax = np.where(xmax == xmin, xmin + 1e-6, xmax)
                    self.stats = NormalizationStats(xmin=xmin, xmax=xmax)
                else:  # zscore
                    # store mean/std in stats fields by reusing names
                    mu = X_all.mean(axis=0)
                    sigma = X_all.std(axis=0)
                    sigma = np.where(sigma == 0.0, 1e-6, sigma)
                    self.stats = NormalizationStats(xmin=mu, xmax=sigma)
            else:
                self.stats = stats

            self.x1 = self._apply_normalization(self.x1, self.stats)
            self.x2 = self._apply_normalization(self.x2, self.stats)

        # Convert to tensors once (fast)
        self.x1_t = torch.tensor(self.x1, dtype=dtype)
        self.x2_t = torch.tensor(self.x2, dtype=dtype)
        self.y1_t = torch.tensor(self.y1, dtype=dtype)
        self.y2_t = torch.tensor(self.y2, dtype=dtype)

    def _apply_normalization(self, X: np.ndarray, stats: NormalizationStats) -> np.ndarray:
        if self.normalization == "minmax_-1_1":
            xmin, xmax = stats.xmin, stats.xmax
            X01 = (X - xmin) / (xmax - xmin)  # [0,1]
            return 2.0 * X01 - 1.0  # [-1,1]
        else:
            mu, sigma = stats.xmin, stats.xmax
            return (X - mu) / sigma

    def __len__(self) -> int:
        return self.x1_t.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x1_t[idx], self.x2_t[idx], self.y1_t[idx], self.y2_t[idx]


def infer_input_dim(csv_path: str, target_col: str = "capacity", add_cycle_index: bool = True) -> int:
    """Convenience: determine D for your FNet/GNet input."""
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != target_col]
    D = len(feature_cols) + (1 if add_cycle_index else 0)
    return D

import os
import glob

def list_csvs(root_dir: str, pattern: str = "*.csv") -> list[str]:
    files = sorted(glob.glob(os.path.join(root_dir, pattern)))
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found in {root_dir} with pattern {pattern}")
    return files
import torch
from torch.utils.data import TensorDataset, DataLoader
def normalize_features(x: pd.DataFrame, method: str = "minmax") -> pd.DataFrame:
    """
    Authors-style per-battery normalization.
    - minmax: scale each column to [-1, 1]
    - zscore: (x - mean)/std
    """
    x = x.copy()

    if method == "minmax":
        for c in x.columns:
            vmin = x[c].min()
            vmax = x[c].max()
            if pd.notna(vmin) and pd.notna(vmax) and vmax != vmin:
                x[c] = 2.0 * (x[c] - vmin) / (vmax - vmin) - 1.0
            else:
                x[c] = 0.0
        return x

    if method == "zscore":
        for c in x.columns:
            mu = x[c].mean()
            sd = x[c].std()
            if pd.notna(sd) and sd != 0:
                x[c] = (x[c] - mu) / sd
            else:
                x[c] = 0.0
        return x

    raise ValueError(f"Unknown normalization method: {method}")

def read_one_csv(
    file_path: str,
    nominal_capacity: float | None = None,
    normalization: bool = True,
    normalization_method: str = "minmax",
    add_cycle_index: bool = True,
):
    df = pd.read_csv(file_path)

    # numeric cleanup
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # add cycle index before target (assume last col is capacity)
    if add_cycle_index:
        n = len(df)
        df.insert(df.shape[1] - 1, "cycle index", np.arange(1, n + 1))

    # OPTIONAL: if you still have delete_3_sigma() in this file, you can call it here.
    # For duplication-mode baseline, keep it OFF to avoid nuking data:
    # df = delete_3_sigma(df)

    # capacity -> SOH if nominal provided
    if nominal_capacity is not None:
        if "capacity" not in df.columns:
            raise ValueError(f"'capacity' column not found in {file_path}")
        df["capacity"] = df["capacity"].astype(float) / float(nominal_capacity)

    # normalize features (all but last col)
    if normalization:
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        x = normalize_features(x, method=normalization_method)
        df = pd.concat([x, y], axis=1)

    return df.reset_index(drop=True)

def make_dataloader_from_files(
    csv_files,
    nominal_capacity=None,
    batch_size=64,
    shuffle=True,
    normalization=True,
    normalization_method="minmax",
    num_workers=0,
):
    X1s, Y1s, X2s, Y2s = [], [], [], []

    for fp in csv_files:
        df = read_one_csv(
            fp,
            nominal_capacity=nominal_capacity,
            normalization=normalization,
            normalization_method=normalization_method,
            add_cycle_index=True,
        )

        x = df.iloc[:, :-1].to_numpy(dtype="float32")
        y = df.iloc[:, -1].to_numpy(dtype="float32")

        if len(x) < 2:
            continue

        X1s.append(x[:-1])
        Y1s.append(y[:-1])
        X2s.append(x[1:])
        Y2s.append(y[1:])

    if len(X1s) == 0:
        raise RuntimeError("No valid battery data found for training.")

    X1 = torch.tensor(np.vstack(X1s))
    Y1 = torch.tensor(np.hstack(Y1s)).unsqueeze(1)
    X2 = torch.tensor(np.vstack(X2s))
    Y2 = torch.tensor(np.hstack(Y2s)).unsqueeze(1)

    dataset = TensorDataset(X1, Y1, X2, Y2)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

