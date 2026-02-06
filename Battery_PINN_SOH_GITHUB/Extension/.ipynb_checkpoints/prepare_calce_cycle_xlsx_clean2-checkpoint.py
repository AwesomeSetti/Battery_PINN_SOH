import os
import re
import argparse
import pandas as pd
import numpy as np

# =========================
# Feature helper functions
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
    # sort by time to be safe
    idx = np.argsort(t)
    t, y = t[idx], y[idx]
    t0 = t - t.mean()
    denom = np.sum(t0 ** 2)
    if denom == 0:
        return np.nan
    return np.sum(t0 * (y - y.mean())) / denom

def entropy_1d(x, bins=30):
    """
    Proper discrete entropy: histogram COUNTS -> probabilities.
    This avoids negative huge values from density=True PDF.
    """
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    counts, _ = np.histogram(x, bins=bins, density=False)
    p = counts.astype(float)
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    p = p / p.sum()
    return -np.sum(p * np.log(p))

def trapz_ah(t_s, i_a):
    """
    Robust Ah integration:
    - sorts by time
    - converts absolute Test_Time to relative (t - t0)
    This prevents negative integrals when time is not monotonic
    or is absolute across the whole experiment.
    """
    m = np.isfinite(t_s) & np.isfinite(i_a)
    t_s, i_a = t_s[m], i_a[m]
    if t_s.size < 2:
        return np.nan
    idx = np.argsort(t_s)
    t_s, i_a = t_s[idx], i_a[idx]
    t_s = t_s - t_s[0]
    return np.trapz(i_a, t_s) / 3600.0

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
# Toggle: keep temp_C?
# =========================
KEEP_TEMP_C = False  # recommended False (keep only temp_K)

# =========================
# Filename parsing helpers
# =========================
def parse_temp_and_condition_id(path: str):
    # "...-10DU 01.xlsx"
    base = os.path.basename(path)
    m = re.search(r"-(\d+)DU[-\s_]+(\d+)\.xlsx$", base, re.IGNORECASE)
    if not m:
        return None, None
    return float(m.group(1)), int(m.group(2))

def parse_cycle_range(path: str):
    # DOE-001-050-... -> (1, 50)
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
# IO helpers
# =========================
def read_all_channel_sheets(xlsx_path: str):
    xf = pd.ExcelFile(xlsx_path)
    sheets = [s for s in xf.sheet_names if s.lower().startswith("channel") and s.endswith("_1")]
    if not sheets:
        raise RuntimeError(f"No Channel_*_1 sheets found in {xlsx_path}")
    frames = []
    for s in sheets:
        frames.append(xf.parse(s))
    return pd.concat(frames, ignore_index=True)

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
    df[QD_COL] = pd.to_numeric(df[QD_COL], errors="coerce")
    df = df.dropna(subset=[CYC_COL, QD_COL])

    # conservative: per cycle max
    q_by_cyc = df.groupby(CYC_COL)[QD_COL].max().sort_index()
    q_by_cyc = q_by_cyc[q_by_cyc > 0]
    if q_by_cyc.empty:
        raise RuntimeError(f"Could not compute Q0 from initial file: {init_path}")

    q0 = float(q_by_cyc.iloc[0])
    return q0, init_path

# =========================
# Robust CV detection
# =========================
def detect_cc_cv_segments(charge_df: pd.DataFrame, i_cc_level: float):
    """
    Robust CC/CV rule (cycle-internal):
    - CC: current >= 0.8 * i_cc_level
    - candidate CV starts when current drops below 0.8 * i_cc_level
    - restrict CV to "near end-of-charge voltage window" (top 0.2V)
    - if that yields too few points, fallback to end-of-charge window alone
    """
    v = charge_df[V_COL].to_numpy(dtype=float)
    i = charge_df[I_COL].to_numpy(dtype=float)
    t = charge_df[T_COL].to_numpy(dtype=float)

    # clean
    m = np.isfinite(v) & np.isfinite(i) & np.isfinite(t)
    v, i, t = v[m], i[m], t[m]
    if v.size < 10:
        return (np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]), np.array([]))

    # sort by time
    idx = np.argsort(t)
    v, i, t = v[idx], i[idx], t[idx]

    # make time relative for segment computations
    t = t - t[0]

    vend = np.nanmax(v)
    vw_mask = (v >= vend - 0.2) & (v <= vend)

    # CC: >= 80% median positive current
    cc_mask = i >= (0.80 * i_cc_level)
    i_cc = i[cc_mask]
    t_cc = t[cc_mask]
    v_cc = v[cc_mask]

    # CV: after current has dropped below 80% CC (candidate),
    # AND near end-of-charge voltage window
    below = i < (0.80 * i_cc_level)
    if np.any(below):
        first_below_idx = np.argmax(below)  # first True index (because sorted)
    else:
        first_below_idx = v.size  # no below region -> no CV

    after_drop = np.zeros_like(i, dtype=bool)
    after_drop[first_below_idx:] = True

    cv_mask = after_drop & vw_mask & (i > 0)  # still charging, near end
    i_cv = i[cv_mask]
    t_cv = t[cv_mask]
    v_cv = v[cv_mask]

    # fallback if too small: just end-of-charge window with positive current
    if i_cv.size < 5:
        cv_mask2 = vw_mask & (i > 0)
        i_cv = i[cv_mask2]
        t_cv = t[cv_mask2]
        v_cv = v[cv_mask2]

    # also return end-of-charge window arrays for voltage features
    v_w = v[vw_mask]
    t_w = t[vw_mask]

    return v_w, t_w, i_cc, t_cc, v_cc, i_cv, t_cv, v_cv

# =========================
# Core feature extraction
# =========================
def extract_features_from_continuous(xlsx_path: str, q0: float, tempC: float, cond_id: int):
    df = read_all_channel_sheets(xlsx_path)

    needed = [V_COL, I_COL, T_COL, CYC_COL, QD_COL, STEP_COL]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in continuous file: {xlsx_path}")

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[V_COL, I_COL, T_COL, CYC_COL])

    rows = []
    for cyc, g in df.groupby(CYC_COL, sort=True):
        # charge portion
        charge = g[g[I_COL] > 0]
        if len(charge) < 10:
            continue

        # discharge isolation (Step-based preferred)
        if STEP_COL in g.columns:
            step_groups = g.groupby(STEP_COL)
            step_score = step_groups[I_COL].sum()
            discharge_step = step_score.idxmin()
            discharge = g[g[STEP_COL] == discharge_step]
        else:
            discharge = g[g[I_COL] < 0]

        # discharge capacity (robust: max-min)
        qd = discharge[QD_COL].to_numpy(dtype=float)
        qd = qd[np.isfinite(qd)]
        if qd.size < 2:
            continue

        qd_cycle = float(np.nanmax(qd) - np.nanmin(qd))
        if not np.isfinite(qd_cycle) or qd_cycle <= 0:
            continue

        # arrays for charge
        v_ch = charge[V_COL].to_numpy(dtype=float)
        i_ch = charge[I_COL].to_numpy(dtype=float)
        t_ch = charge[T_COL].to_numpy(dtype=float)

        m = np.isfinite(v_ch) & np.isfinite(i_ch) & np.isfinite(t_ch)
        v_ch, i_ch, t_ch = v_ch[m], i_ch[m], t_ch[m]
        if v_ch.size < 10:
            continue

        i_pos = i_ch[i_ch > 0]
        if i_pos.size < 5:
            continue
        i_cc_level = float(np.nanmedian(i_pos))

        v_w, t_w, i_cc, t_cc, v_cc, i_cv, t_cv, v_cv = detect_cc_cv_segments(charge, i_cc_level)

        # choose current window for current stats: prefer CV if meaningful else CC
        if i_cv.size >= 5 and t_cv.size >= 5:
            i_feat, t_feat = i_cv, t_cv
        else:
            i_feat, t_feat = i_cc, t_cc

        # voltage stats (end-of-charge window)
        v_mean  = np.nanmean(v_w) if v_w.size else np.nan
        v_std   = np.nanstd(v_w) if v_w.size else np.nan
        v_kurt  = kurtosis_pearson(v_w) if v_w.size else np.nan
        v_skew  = skewness(v_w) if v_w.size else np.nan
        v_slope = linear_slope(t_w, v_w) if v_w.size else np.nan
        v_ent   = entropy_1d(v_w) if v_w.size else np.nan

        # current stats (robust window)
        i_mean  = np.nanmean(i_feat) if i_feat.size else 0.0
        i_std   = np.nanstd(i_feat) if i_feat.size else 0.0
        i_kurt  = kurtosis_pearson(i_feat) if i_feat.size else 0.0
        i_skew  = skewness(i_feat) if i_feat.size else 0.0
        i_slope = linear_slope(t_feat, i_feat) if i_feat.size else 0.0
        i_ent   = entropy_1d(i_feat) if i_feat.size else 0.0

        # CC and CV time and charge quantities
        cc_time = (np.nanmax(t_cc) - np.nanmin(t_cc)) if t_cc.size >= 2 else 0.0
        cc_q    = trapz_ah(t_cc, i_cc) if t_cc.size >= 2 else 0.0

        cv_time = (np.nanmax(t_cv) - np.nanmin(t_cv)) if t_cv.size >= 2 else 0.0
        cv_q    = trapz_ah(t_cv, i_cv) if t_cv.size >= 2 else 0.0

        # temperature features
        tempK = float(tempC + 273.15)

        row = {
            "voltage mean": v_mean,
            "voltage std": v_std,
            "voltage kurtosis": v_kurt,
            "voltage skewness": v_skew,
            "CC Q": float(cc_q) if np.isfinite(cc_q) else 0.0,
            "CC charge time": float(cc_time) if np.isfinite(cc_time) else 0.0,
            "voltage slope": v_slope,
            "voltage entropy": v_ent,
            "current mean": float(i_mean) if np.isfinite(i_mean) else 0.0,
            "current std": float(i_std) if np.isfinite(i_std) else 0.0,
            "current kurtosis": float(i_kurt) if np.isfinite(i_kurt) else 0.0,
            "current skewness": float(i_skew) if np.isfinite(i_skew) else 0.0,
            "CV Q": float(cv_q) if np.isfinite(cv_q) else 0.0,
            "CV charge time": float(cv_time) if np.isfinite(cv_time) else 0.0,
            "current slope": float(i_slope) if np.isfinite(i_slope) else 0.0,
            "current entropy": float(i_ent) if np.isfinite(i_ent) else 0.0,
            "cycle": float(cyc),
            "temp_K": tempK,
            "condition_id": float(cond_id),
            "raw_Qd_Ah": float(qd_cycle),
        }

        if KEEP_TEMP_C:
            row["temp_C"] = float(tempC)

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"No valid cycles extracted from continuous file: {xlsx_path}")
    return out

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Extension root folder")
    ap.add_argument("--mode", type=str, default="batch", choices=["single", "batch"])
    ap.add_argument("--continuous_xlsx", type=str, default=None, help="Used only in single mode")
    ap.add_argument("--out_dir", type=str, default=None, help="Output folder (default: <root>/CALCE_processed)")
    args = ap.parse_args()

    root = args.root
    out_dir = args.out_dir or os.path.join(root, "CALCE_processed")
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
            raise RuntimeError("Could not parse temp/condition_id from filename like: ...-10DU 01.xlsx")

        out_df = extract_features_from_continuous(args.continuous_xlsx, q0=q0, tempC=tempC, cond_id=cond_id)
        out_df = make_global_cycle(out_df, args.continuous_xlsx)

        # compute capacity with local earliest cycle for single file
        first_cycle = float(out_df["cycle"].min())
        q0_group = float(out_df.loc[out_df["cycle"] == first_cycle, "raw_Qd_Ah"].iloc[0])
        out_df["capacity"] = out_df["raw_Qd_Ah"] / q0_group
        out_df = out_df.drop(columns=["raw_Qd_Ah"])

        # ensure capacity last column
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

    groups = {}
    for p in all_files:
        tempC, cond_id = parse_temp_and_condition_id(p)
        if tempC is None or cond_id is None:
            continue
        groups.setdefault((tempC, cond_id), []).append(p)

    if not groups:
        raise RuntimeError("Found .xlsx files but none matched pattern: DOE-xxx-yyy-10DU 01.xlsx")

    print(f"[OK] Found {sum(len(v) for v in groups.values())} continuous files in {len(groups)} (temp,cond) groups.")

    # debug first 3 groups
    print("\n[DEBUG] Showing file lists for the first 3 (temp,cond) groups:")
    for k in sorted(groups.keys())[:3]:
        print(" ", k, "->", len(groups[k]), "files")
        for p in sorted(groups[k])[:15]:
            print("    ", os.path.basename(p))

    for (tempC, cond_id), files in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):

        def sort_key(p):
            lo, hi = parse_cycle_range(p)
            return (lo if lo is not None else 1_000_000, hi if hi is not None else 1_000_000)

        files = sorted(files, key=sort_key)

        dfs = []
        for p in files:
            try:
                df_part = extract_features_from_continuous(p, q0=q0, tempC=tempC, cond_id=cond_id)
                df_part = make_global_cycle(df_part, p)
                dfs.append(df_part)
                print(f"  ✓ T{int(tempC)}C cond{cond_id:02d}: {os.path.basename(p)} -> {len(df_part)} rows")
            except Exception as e:
                print(f"  ✗ T{int(tempC)}C cond{cond_id:02d}: {os.path.basename(p)} failed: {e}")

        if not dfs:
            print(f"[SKIP] T{int(tempC)}C cond{cond_id:02d}: no usable blocks.")
            continue

        full = pd.concat(dfs, ignore_index=True)

        # remove duplicate cycles if any
        full = full.sort_values("cycle").drop_duplicates(subset=["cycle"], keep="last").reset_index(drop=True)

        # compute capacity globally using earliest cycle in group
        first_cycle = float(full["cycle"].min())
        q0_group = float(full.loc[full["cycle"] == first_cycle, "raw_Qd_Ah"].iloc[0])
        full["capacity"] = full["raw_Qd_Ah"] / q0_group
        print(f"[DEBUG] T{int(tempC)}C cond{cond_id:02d}: q0_group={q0_group:.4f}Ah at cycle={first_cycle}")

        full = full.drop(columns=["raw_Qd_Ah"])

        # put capacity last
        cap = full.pop("capacity")
        full["capacity"] = cap

        out_name = f"T{int(tempC)}C_cond{cond_id:02d}_ALL_processed.csv"
        out_path = os.path.join(out_dir, out_name)
        full.to_csv(out_path, index=False)
        print(f"[DONE] Saved group: {out_path} (rows={len(full)}, cols={len(full.columns)})")

if __name__ == "__main__":
    main()
