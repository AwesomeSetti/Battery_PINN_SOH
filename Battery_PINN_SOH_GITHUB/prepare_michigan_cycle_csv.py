import os
import re
import glob
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------
def safe_float_array(x: pd.Series) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)

def finite_mask(*arrs: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m

def linear_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Slope of y vs t using least squares. Returns NaN if insufficient points."""
    if t.size < 2:
        return float("nan")
    t0 = t - np.nanmean(t)
    y0 = y - np.nanmean(y)
    denom = np.nansum(t0 * t0)
    if denom == 0:
        return float("nan")
    return float(np.nansum(t0 * y0) / denom)

def entropy_1d(x: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy of histogram (in nats)."""
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float("nan")
    hist, _ = np.histogram(x, bins=bins, density=False)
    p = hist.astype(float)
    s = p.sum()
    if s <= 0:
        return float("nan")
    p /= s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def kurtosis_pearson(x: np.ndarray) -> float:
    """Pearson kurtosis (normal==3)."""
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float("nan")
    mu = x.mean()
    m2 = np.mean((x - mu) ** 2)
    if m2 == 0:
        return float("nan")
    m4 = np.mean((x - mu) ** 4)
    return float(m4 / (m2 ** 2))

def skewness(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    mu = x.mean()
    m2 = np.mean((x - mu) ** 2)
    if m2 == 0:
        return float("nan")
    m3 = np.mean((x - mu) ** 3)
    return float(m3 / (m2 ** 1.5))

def trapz_ah(t_s: np.ndarray, i_a: np.ndarray) -> float:
    """Integrate current over time (seconds) -> Ah."""
    m = finite_mask(t_s, i_a)
    t_s = t_s[m]
    i_a = i_a[m]
    if t_s.size < 2:
        return float("nan")
    # Ensure increasing time
    order = np.argsort(t_s)
    t_s = t_s[order]
    i_a = i_a[order]
    q_as = np.trapz(i_a, t_s)  # A*s
    return float(q_as / 3600.0)  # Ah


@dataclass
class ColumnMap:
    v_col: str = "Potential (V)"
    i_col: str = "Current (A)"
    t_col: str = "Test Time (s)"
    cyc_col: str = "Cycle Number"
    step_col: str = "Step Index"
    qd_col: str = "Discharge Capacity (Ah)"


def infer_cell_id(filename: str) -> str:
    """
    Extract cell id from names like:
    UM_Internal_0620_-_BL_Form_-_Cycling_Cell_11.001.csv  -> 11
    """
    m = re.search(r"Cycling_Cell_(\d+)", filename)
    return m.group(1) if m else os.path.splitext(os.path.basename(filename))[0]


def process_one_cycling_csv(
    csv_path: str,
    out_dir: str,
    cols: ColumnMap,
    vend_window_v: float = 0.2,
    cv_i_low: float = 0.1,
    cv_i_high: float = 0.5,
    cc_i_threshold: float = 0.5,
) -> str:
    """
    Convert one Michigan cycling time-series CSV into one per-cycle feature table CSV.

    Output columns (paper-style):
      voltage mean, voltage std, voltage kurtosis, voltage skewness,
      CC Q, CC charge time, voltage slope, voltage entropy,
      current mean, current std, current kurtosis, current skewness,
      CV Q, CV charge time, current slope, current entropy,
      capacity  (SOH)
    """
    os.makedirs(out_dir, exist_ok=True)
    cell_id = infer_cell_id(os.path.basename(csv_path))

    usecols = [
        cols.v_col, cols.i_col, cols.t_col, cols.cyc_col, cols.step_col, cols.qd_col
    ]
    df = pd.read_csv(csv_path, usecols=usecols)

    # Coerce numeric
    for c in [cols.v_col, cols.i_col, cols.t_col, cols.cyc_col, cols.step_col, cols.qd_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows without essential fields
    df = df.dropna(subset=[cols.v_col, cols.i_col, cols.t_col, cols.cyc_col])

    # We will build per-cycle rows
    rows: List[Dict[str, float]] = []

    # Group by Cycle Number
    for cyc, g in df.groupby(cols.cyc_col, sort=True):
        # Charge rows: current > 0
        charge = g[g[cols.i_col] > 0]
        discharge = g[g[cols.i_col] < 0]

        if charge.shape[0] < 10:
            continue

        v = safe_float_array(charge[cols.v_col])
        i = safe_float_array(charge[cols.i_col])
        t = safe_float_array(charge[cols.t_col])

        m = finite_mask(v, i, t)
        v, i, t = v[m], i[m], t[m]
        if v.size < 10:
            continue

        vend = float(np.nanmax(v))
        # Voltage window near end-of-charge
        vw_mask = (v >= (vend - vend_window_v)) & (v <= vend)
        v_w = v[vw_mask]
        t_w = t[vw_mask]

        # CV current window (paper): I in [0.1, 0.5] A (during charge)
        cv_mask = (i >= cv_i_low) & (i <= cv_i_high) & vw_mask
        i_cv = i[cv_mask]
        t_cv = t[cv_mask]

        # CC region (simple practical definition): I >= 0.5 A during charge
        cc_mask = (i >= cc_i_threshold)
        i_cc = i[cc_mask]
        t_cc = t[cc_mask]

        # Features: voltage stats computed on voltage window
        v_mean = float(np.nanmean(v_w)) if v_w.size else float("nan")
        v_std = float(np.nanstd(v_w)) if v_w.size else float("nan")
        v_kurt = kurtosis_pearson(v_w) if v_w.size else float("nan")
        v_skew = skewness(v_w) if v_w.size else float("nan")
        v_slope = linear_slope(t_w, v_w) if v_w.size else float("nan")
        v_ent = entropy_1d(v_w) if v_w.size else float("nan")

        # Current stats computed on CV current window (closest to paper)
        i_mean = float(np.nanmean(i_cv)) if i_cv.size else float("nan")
        i_std = float(np.nanstd(i_cv)) if i_cv.size else float("nan")
        i_kurt = kurtosis_pearson(i_cv) if i_cv.size else float("nan")
        i_skew = skewness(i_cv) if i_cv.size else float("nan")
        i_slope = linear_slope(t_cv, i_cv) if i_cv.size else float("nan")
        i_ent = entropy_1d(i_cv) if i_cv.size else float("nan")

        # Times (seconds) and Q (Ah)
        cc_time = float(np.nanmax(t_cc) - np.nanmin(t_cc)) if t_cc.size else float("nan")
        cv_time = float(np.nanmax(t_cv) - np.nanmin(t_cv)) if t_cv.size else float("nan")

        cc_q = trapz_ah(t_cc, i_cc) if t_cc.size else float("nan")
        cv_q = trapz_ah(t_cv, i_cv) if t_cv.size else float("nan")

        # Capacity label for SOH: use discharge capacity column if present
        # In these exports, Discharge Capacity (Ah) is cumulative within a cycle/step;
        # taking max in the cycle is a robust "cycle discharge capacity".
        qd = pd.to_numeric(discharge[cols.qd_col], errors="coerce").to_numpy(dtype=float)
        qd = qd[np.isfinite(qd)]
        qd_cycle = float(np.nanmax(qd)) if qd.size else float("nan")

        rows.append({
            "voltage mean": v_mean,
            "voltage std": v_std,
            "voltage kurtosis": v_kurt,
            "voltage skewness": v_skew,
            "CC Q": cc_q,
            "CC charge time": cc_time,
            "voltage slope": v_slope,
            "voltage entropy": v_ent,
            "current mean": i_mean,
            "current std": i_std,
            "current kurtosis": i_kurt,
            "current skewness": i_skew,
            "CV Q": cv_q,
            "CV charge time": cv_time,
            "current slope": i_slope,
            "current entropy": i_ent,
            "raw_discharge_capacity_ah": qd_cycle,
            "cycle": float(cyc),
        })

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise RuntimeError(f"No usable cycles extracted from: {csv_path}")

    # Compute SOH from discharge capacity normalized by first valid cycle
    # Use the earliest cycle with finite discharge capacity as baseline.
    base_idx = out_df["raw_discharge_capacity_ah"].first_valid_index()
    base_q = float(out_df.loc[base_idx, "raw_discharge_capacity_ah"]) if base_idx is not None else float("nan")
    if not np.isfinite(base_q) or base_q <= 0:
        raise RuntimeError(f"Could not compute baseline discharge capacity for: {csv_path}")

    out_df["capacity"] = out_df["raw_discharge_capacity_ah"] / base_q  # SOH
    out_df = out_df.drop(columns=["raw_discharge_capacity_ah"])

    # Sort by cycle and drop rows with NaNs in essential features
    out_df = out_df.sort_values("cycle").reset_index(drop=True)

    # Save per-cell processed CSV (paper-style)
    out_path = os.path.join(out_dir, f"Michigan_Cell_{int(float(cell_id)):02d}.csv")
    out_df.to_csv(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True,
                    help="Folder containing UM_Internal_*Cycling_Cell_*.csv files")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Where to write processed per-cell per-cycle CSVs")
    ap.add_argument("--pattern", type=str, default="UM_Internal_*Cycling_Cell_*.csv",
                    help="Glob pattern for cycling time-series CSV files")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not csvs:
        raise RuntimeError(f"No files matched in {args.input_dir} with pattern {args.pattern}")

    cols = ColumnMap()

    print(f"Found {len(csvs)} cycling CSV(s). Processing...")
    for p in csvs:
        try:
            out = process_one_cycling_csv(p, args.output_dir, cols=cols)
            print(f"✓ {os.path.basename(p)} -> {out}")
        except Exception as e:
            print(f"✗ {os.path.basename(p)} failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
