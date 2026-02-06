#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_calce_cycle_xlsx_fixed.py

CALCE Continuous Cycling DOE .xlsx -> per-(temp,cond) aggregated CSV

Outputs columns:
[16 Michigan-style features] + [cycle, temp_C, temp_K, condition_id, capacity]

Key fixes vs your old version:
- CC/CV time and Q are computed with dt-based integration with gap filtering (dt_max).
- Time is effectively “per-cycle active time” (we integrate only within step samples; big gaps dropped).
- CC/CV segmentation is more robust:
  * CC level = median of top 20% of charging currents
  * CC region = i >= 0.8 * i_cc_level
  * CV region = near end-of-charge voltage AND i between [i_charge_min_abs, 0.8*i_cc_level)
- Discharge capacity per cycle is computed from discharge step (Step_Index with most negative total current)
  using (max(QD)-min(QD)) to handle cumulative counters.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np

# =========================
# Michigan-style helpers
# =========================
def kurtosis_pearson(x):
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nan
    mu = x.mean()
    m2 = np.mean((x - mu) ** 2)
    if m2 == 0:
        return 0.0
    m4 = np.mean((x - mu) ** 4)
    return m4 / (m2 ** 2)

def skewness(x):
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    mu = x.mean()
    m2 = np.mean((x - mu) ** 2)
    if m2 == 0:
        return 0.0
    m3 = np.mean((x - mu) ** 3)
    return m3 / (m2 ** 1.5)

def linear_slope(t, y):
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size < 2:
        return np.nan
    t0 = t - t.mean()
    denom = np.sum(t0 ** 2)
    if denom == 0:
        return np.nan
    return np.sum(t0 * (y - y.mean())) / denom

def entropy_1d(x, bins=30):
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return np.nan
    return -np.sum(hist * np.log(hist))

def robust_time_and_ah(t_s, i_a, dt_max=120.0):
    """
    Robustly compute active_time_s and Ah using dt between consecutive samples,
    filtering out large gaps (dt >= dt_max) and non-forward time.
    """
    m = np.isfinite(t_s) & np.isfinite(i_a)
    t_s, i_a = t_s[m], i_a[m]
    if t_s.size < 2:
        return 0.0, 0.0

    idx = np.argsort(t_s)
    t_s, i_a = t_s[idx], i_a[idx]

    dt = np.diff(t_s)
    good = (dt > 0) & (dt < dt_max)
    if not np.any(good):
        return 0.0, 0.0

    active_time = float(np.sum(dt[good]))
    ah = float(np.sum(i_a[1:][good] * dt[good]) / 3600.0)
    return active_time, ah


# =========================
# CALCE column names
# =========================
V_COL    = "Voltage(V)"
I_COL    = "Current(A)"
T_COL    = "Test_Time(s)"
CYC_COL  = "Cycle_Index"
STEP_COL = "Step_Index"
QD_COL   = "Discharge_Capacity(Ah)"


# =========================
# filename parsing helpers
# =========================
def parse_temp_and_condition_id(path: str):
    # e.g. DOE-001-050-10DU 01.xlsx  OR DOE-401-450-10DU-01.xlsx
    base = os.path.basename(path)
    m = re.search(r"-(\d+)DU[-\s_]+(\d+)\.xlsx$", base, re.IGNORECASE)
    if not m:
        return None, None
    return float(m.group(1)), int(m.group(2))

def parse_cycle_range(path: str):
    # DOE-001-050-10DU 01.xlsx -> (1, 50)
    base = os.path.basename(path)
    m = re.search(r"DOE-(\d+)-(\d+)-", base, re.IGNORECASE)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def make_global_cycle(df_part: pd.DataFrame, xlsx_path: str) -> pd.DataFrame:
    lo, _ = parse_cycle_range(xlsx_path)
    if lo is None:
        return df_part
    offset = lo - 1
    df_part = df_part.copy()
    df_part["cycle"] = df_part["cycle"] + offset
    return df_part


# =========================
# IO
# =========================
def read_all_channel_sheets(xlsx_path: str) -> pd.DataFrame:
    xf = pd.ExcelFile(xlsx_path)
    sheets = [s for s in xf.sheet_names if s.lower().startswith("channel") and s.endswith("_1")]
    if not sheets:
        raise RuntimeError(f"No Channel_*_1 sheets found in {xlsx_path}")
    frames = []
    for s in sheets:
        frames.append(xf.parse(s))
    return pd.concat(frames, ignore_index=True)


# =========================
# Q0
# =========================
def compute_q0_from_initial(initial_dir: str):
    candidates = [f for f in os.listdir(initial_dir)
                  if f.lower().endswith(".xlsx") and "initial" in f.lower()]
    if not candidates:
        candidates = [f for f in os.listdir(initial_dir) if f.lower().endswith(".xlsx")]
    if not candidates:
        raise RuntimeError(f"No initial characterization xlsx found in: {initial_dir}")

    init_path = os.path.join(initial_dir, sorted(candidates)[0])
    df = read_all_channel_sheets(init_path)

    for c in [CYC_COL, QD_COL]:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in Initial file: {init_path}")

    df[CYC_COL] = pd.to_numeric(df[CYC_COL], errors="coerce")
    df[QD_COL]  = pd.to_numeric(df[QD_COL], errors="coerce")
    df = df.dropna(subset=[CYC_COL, QD_COL])

    # Initial file: take per-cycle max as discharge capacity
    q_by_cyc = df.groupby(CYC_COL)[QD_COL].max().sort_index()
    q_by_cyc = q_by_cyc[q_by_cyc > 0]
    if q_by_cyc.empty:
        raise RuntimeError(f"Could not compute Q0 from initial file: {init_path}")

    q0 = float(q_by_cyc.iloc[0])
    return q0, init_path


# =========================
# Feature extraction
# =========================
def extract_features_from_continuous(
    xlsx_path: str,
    tempC: float,
    cond_id: int,
    dt_max: float = 120.0,
    i_charge_min_abs: float = 0.02,  # A: filters tiny noise
):
    df = read_all_channel_sheets(xlsx_path)

    needed = [V_COL, I_COL, T_COL, CYC_COL, QD_COL, STEP_COL]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in continuous file: {xlsx_path}")

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # We need at least V,I,T,cycle
    df = df.dropna(subset=[V_COL, I_COL, T_COL, CYC_COL])

    rows = []

    for cyc, g in df.groupby(CYC_COL, sort=True):
        # ---- isolate discharge step (capacity) ----
        # Step selection: the step with most negative total current is the discharge step
        if STEP_COL in g.columns and g[STEP_COL].notna().any():
            step_score = g.groupby(STEP_COL)[I_COL].sum()
            discharge_step = step_score.idxmin()  # most negative total current
            discharge = g[g[STEP_COL] == discharge_step]
        else:
            discharge = g[g[I_COL] < 0]

        qd = discharge[QD_COL].to_numpy(dtype=float)
        qd = qd[np.isfinite(qd)]
        if qd.size == 0:
            continue

        # robust capacity: (max - min) in that discharge step
        qd_cycle = float(np.nanmax(qd) - np.nanmin(qd))
        if (not np.isfinite(qd_cycle)) or (qd_cycle <= 0):
            continue

        # ---- charge segment ----
        charge = g[g[I_COL] > i_charge_min_abs]
        if len(charge) < 10:
            continue

        v = charge[V_COL].to_numpy(dtype=float)
        i = charge[I_COL].to_numpy(dtype=float)
        t = charge[T_COL].to_numpy(dtype=float)

        m = np.isfinite(v) & np.isfinite(i) & np.isfinite(t)
        v, i, t = v[m], i[m], t[m]
        if v.size < 10:
            continue

        # ---- end-of-charge voltage window (for voltage stats) ----
        vend = float(np.nanmax(v))
        vw_mask = (v >= vend - 0.2) & (v <= vend)
        v_w = v[vw_mask]
        t_w = t[vw_mask]

        # ---- estimate CC level robustly ----
        i_pos = i[i > i_charge_min_abs]
        if i_pos.size < 5:
            continue

        # CC level approx = median of top 20% currents (robust against CV tail)
        q80 = np.nanquantile(i_pos, 0.80)
        top = i_pos[i_pos >= q80]
        if top.size >= 3:
            i_cc_level = float(np.nanmedian(top))
        else:
            i_cc_level = float(np.nanmedian(i_pos))

        if not np.isfinite(i_cc_level) or i_cc_level <= 0:
            continue

        # ---- CC/CV segmentation (more robust) ----
        # CC: current near CC level
        cc_mask = (i >= 0.80 * i_cc_level)
        i_cc = i[cc_mask]
        t_cc = t[cc_mask]

        # CV: near top voltage AND current below CC (but still charging)
        cv_vmask = (v >= vend - 0.05) & (v <= vend)
        cv_imask = (i > i_charge_min_abs) & (i < 0.80 * i_cc_level)
        cv_mask = cv_vmask & cv_imask
        i_cv = i[cv_mask]
        t_cv = t[cv_mask]

        # For "current features", prefer CV if present else CC
        if i_cv.size >= 5 and t_cv.size >= 5:
            i_feat, t_feat = i_cv, t_cv
        else:
            i_feat, t_feat = i_cc, t_cc

        # ---- voltage stats ----
        v_mean  = float(np.nanmean(v_w)) if v_w.size else np.nan
        v_std   = float(np.nanstd(v_w))  if v_w.size else np.nan
        v_kurt  = float(kurtosis_pearson(v_w)) if v_w.size else np.nan
        v_skew  = float(skewness(v_w)) if v_w.size else np.nan

        # Use relative time for slope to avoid huge absolute values
        if t_w.size >= 2:
            tw_rel = t_w - np.nanmin(t_w)
            v_slope = float(linear_slope(tw_rel, v_w))
        else:
            v_slope = np.nan

        v_ent   = float(entropy_1d(v_w)) if v_w.size else np.nan

        # ---- current stats ----
        i_mean  = float(np.nanmean(i_feat)) if i_feat.size else 0.0
        i_std   = float(np.nanstd(i_feat))  if i_feat.size else 0.0
        i_kurt  = float(kurtosis_pearson(i_feat)) if i_feat.size else 0.0
        i_skew  = float(skewness(i_feat)) if i_feat.size else 0.0
        if t_feat.size >= 2:
            tf_rel = t_feat - np.nanmin(t_feat)
            i_slope = float(linear_slope(tf_rel, i_feat))
        else:
            i_slope = 0.0
        i_ent   = float(entropy_1d(i_feat)) if i_feat.size else 0.0

        # ---- robust time + Ah (filters gaps in Test_Time) ----
        cc_time, cc_q = robust_time_and_ah(t_cc, i_cc, dt_max=dt_max)
        cv_time, cv_q = robust_time_and_ah(t_cv, i_cv, dt_max=dt_max)

        # If both empty, skip
        if cc_time <= 0 and cv_time <= 0:
            continue

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
            "has_CV": int(cv_time > 0),
            "current slope": i_slope,
            "current entropy": i_ent,
            "cycle": float(cyc),
            "temp_C": float(tempC),
            "temp_K": float(tempC + 273.15),
            "condition_id": float(cond_id),
            "raw_Qd_Ah": float(qd_cycle),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No valid cycles extracted from continuous file: {xlsx_path}")

    out = out.sort_values("cycle").reset_index(drop=True)
    return out


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="CALCE dataset root folder (contains Initial Characterization & Continuous Cycling Data)")
    ap.add_argument("--mode", type=str, default="batch", choices=["single", "batch"])
    ap.add_argument("--continuous_xlsx", type=str, default=None, help="Used only in single mode")
    ap.add_argument("--out_dir", type=str, default=None, help="Output folder (default: <root>/CALCE_processed_fixed)")
    ap.add_argument("--dt_max", type=float, default=120.0,
                    help="Max allowed dt (seconds) when integrating time/Ah to filter gaps")
    ap.add_argument("--i_charge_min_abs", type=float, default=0.02,
                    help="Min +current (A) to consider as charging (filters noise)")
    args = ap.parse_args()

    root = args.root
    out_dir = args.out_dir or os.path.join(root, "CALCE_processed_fixed")
    os.makedirs(out_dir, exist_ok=True)

    initial_dir = os.path.join(root, "Initial Characterization")
    if not os.path.isdir(initial_dir):
        raise RuntimeError(f"Could not find Initial Characterization folder at: {initial_dir}")

    q0, init_path = compute_q0_from_initial(initial_dir)
    print(f"[OK] Q0={q0:.6f} Ah from: {init_path}")

    if args.mode == "single":
        if not args.continuous_xlsx:
            raise RuntimeError("--continuous_xlsx is required in single mode")

        tempC, cond_id = parse_temp_and_condition_id(args.continuous_xlsx)
        if tempC is None or cond_id is None:
            raise RuntimeError("Could not parse temp/condition_id from filename like: DOE-001-050-10DU 01.xlsx")

        out_df = extract_features_from_continuous(
            args.continuous_xlsx,
            tempC=tempC,
            cond_id=cond_id,
            dt_max=args.dt_max,
            i_charge_min_abs=args.i_charge_min_abs,
        )
        out_df = make_global_cycle(out_df, args.continuous_xlsx)

        # capacity baseline from first cycle in this file
        first_cycle = float(out_df["cycle"].min())
        q0_group = float(out_df.loc[out_df["cycle"] == first_cycle, "raw_Qd_Ah"].iloc[0])
        out_df["capacity"] = out_df["raw_Qd_Ah"] / q0_group

        out_df = out_df.drop(columns=["raw_Qd_Ah"])
        cap = out_df.pop("capacity")
        out_df["capacity"] = cap

        lo, hi = parse_cycle_range(args.continuous_xlsx)
        suffix = f"cycles{lo:03d}_{hi:03d}" if lo is not None else "cycles"
        out_name = f"T{int(tempC)}C_cond{cond_id:02d}_{suffix}_processed.csv"
        out_path = os.path.join(out_dir, out_name)
        out_df.to_csv(out_path, index=False)
        print(f"[DONE] Saved: {out_path}")
        return

    # -------- batch mode --------
    cont_root = os.path.join(root, "Continuous Cycling Data")
    if not os.path.isdir(cont_root):
        raise RuntimeError(f"Could not find Continuous Cycling Data folder at: {cont_root}")

    all_files = []
    for dirpath, _, filenames in os.walk(cont_root):
        for fn in filenames:
            if fn.lower().endswith(".xlsx") and fn.lower().startswith("doe-"):
                all_files.append(os.path.join(dirpath, fn))
    if not all_files:
        raise RuntimeError(f"No DOE-*.xlsx files found under: {cont_root}")

    # group by (tempC, cond_id)
    groups = {}
    for p in all_files:
        tempC, cond_id = parse_temp_and_condition_id(p)
        if tempC is None or cond_id is None:
            continue
        groups.setdefault((tempC, cond_id), []).append(p)

    if not groups:
        raise RuntimeError("Found .xlsx files but none matched pattern: DOE-xxx-yyy-10DU 01.xlsx")

    total_files = sum(len(v) for v in groups.values())
    print(f"[OK] Found {total_files} continuous files in {len(groups)} (temp,cond) groups.")

    # DEBUG show first 3 groups
    print("\n[DEBUG] Showing file lists for the first 3 (temp,cond) groups:")
    for k in sorted(groups.keys())[:3]:
        print(" ", k, "->", len(groups[k]), "files")
        for p in sorted(groups[k])[:15]:
            print("    ", os.path.basename(p))

    # process each group
    for (tempC, cond_id), files in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        def sort_key(p):
            lo, hi = parse_cycle_range(p)
            return (lo if lo is not None else 1_000_000,
                    hi if hi is not None else 1_000_000)

        files = sorted(files, key=sort_key)

        dfs = []
        for p in files:
            try:
                df_part = extract_features_from_continuous(
                    p,
                    tempC=tempC,
                    cond_id=cond_id,
                    dt_max=args.dt_max,
                    i_charge_min_abs=args.i_charge_min_abs,
                )
                df_part = make_global_cycle(df_part, p)
                dfs.append(df_part)
                print(f"  ✓ T{int(tempC)}C cond{cond_id:02d}: {os.path.basename(p)} -> {len(df_part)} rows")
            except Exception as e:
                print(f"  ✗ T{int(tempC)}C cond{cond_id:02d}: {os.path.basename(p)} failed: {e}")

        if not dfs:
            print(f"[SKIP] T{int(tempC)}C cond{cond_id:02d}: no usable blocks.")
            continue

        full = pd.concat(dfs, ignore_index=True)

        # cycles may repeat across blocks; keep the latest occurrence
        full = (
            full.sort_values("cycle")
                .drop_duplicates(subset=["cycle"], keep="last")
                .reset_index(drop=True)
        )

        # compute capacity baseline from earliest global cycle in this (temp,cond) group
        first_cycle = float(full["cycle"].min())
        q0_group = float(full.loc[full["cycle"] == first_cycle, "raw_Qd_Ah"].iloc[0])
        full["capacity"] = full["raw_Qd_Ah"] / q0_group
        print(f"[DEBUG] T{int(tempC)}C cond{cond_id:02d}: q0_group={q0_group:.4f}Ah at cycle={first_cycle}")

        # drop raw capacity; keep capacity last
        full = full.drop(columns=["raw_Qd_Ah"])
        cap = full.pop("capacity")
        full["capacity"] = cap

        out_name = f"T{int(tempC)}C_cond{cond_id:02d}_ALL_processed.csv"
        out_path = os.path.join(out_dir, out_name)
        full.to_csv(out_path, index=False)
        print(f"[DONE] Saved group: {out_path} (rows={len(full)}, cols={len(full.columns)})")


if __name__ == "__main__":
    main()
