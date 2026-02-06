import os
import re
import argparse
import pandas as pd
import numpy as np
import glob 
# ---------- Michigan-style feature helpers ----------
def kurtosis_pearson(x):
    x = x[np.isfinite(x)]
    if x.size < 4: return np.nan
    mu = x.mean()
    m2 = np.mean((x - mu)**2)
    if m2 == 0: return np.nan
    m4 = np.mean((x - mu)**4)
    return m4 / (m2**2)

def skewness(x):
    x = x[np.isfinite(x)]
    if x.size < 3: return np.nan
    mu = x.mean()
    m2 = np.mean((x - mu)**2)
    if m2 == 0: return np.nan
    m3 = np.mean((x - mu)**3)
    return m3 / (m2**1.5)

def linear_slope(t, y):
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size < 2: return np.nan
    t0 = t - t.mean()
    denom = np.sum(t0**2)
    if denom == 0: return np.nan
    return np.sum(t0 * (y - y.mean())) / denom

def entropy_1d(x, bins=30):
    x = x[np.isfinite(x)]
    if x.size < 5: return np.nan
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0: return np.nan
    return -np.sum(hist * np.log(hist))

def trapz_ah(t_s, i_a):
    m = np.isfinite(t_s) & np.isfinite(i_a)
    t_s, i_a = t_s[m], i_a[m]
    if t_s.size < 2: return np.nan
    return np.trapz(i_a, t_s) / 3600.0  # As -> Ah

# ---------- CALCE column names (verified from your files) ----------
V_COL   = "Voltage(V)"
I_COL   = "Current(A)"
T_COL   = "Test_Time(s)"
CYC_COL = "Cycle_Index"
STEP_COL= "Step_Index"
QD_COL  = "Discharge_Capacity(Ah)"

def parse_temp_and_condition_id(path: str):
    base = os.path.basename(path)

    # Handles BOTH:
    # DOE-101-150-10DU 06.xlsx
    # DOE-501-550-10DU-01.xlsx
    m = re.search(r"-(\d+)DU(?:\s+|-)(\d+)\.xlsx$", base, re.IGNORECASE)

    if not m:
        return None, None

    tempC = float(m.group(1))
    cond_id = int(m.group(2))
    return tempC, cond_id

    
def parse_cycle_range(path: str):
    # DOE-001-050-10DU 01.xlsx -> (1, 50)
    base = os.path.basename(path)
    m = re.search(r"DOE-(\d+)-(\d+)-", base, re.IGNORECASE)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def read_all_channel_sheets(xlsx_path: str):
    xf = pd.ExcelFile(xlsx_path)
    sheets = [s for s in xf.sheet_names if s.lower().startswith("channel") and s.endswith("_1")]
    if not sheets:
        raise RuntimeError(f"No Channel_*_1 sheets found in {xlsx_path}")
    frames = []
    for s in sheets:
        df = xf.parse(s)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def compute_q0_from_initial(initial_dir: str):
    # finds first Initial Chrz*.xlsx and computes Q0 from it
    candidates = [f for f in os.listdir(initial_dir) if f.lower().endswith(".xlsx") and "initial" in f.lower()]
    if not candidates:
        # fallback: any xlsx in initial dir
        candidates = [f for f in os.listdir(initial_dir) if f.lower().endswith(".xlsx")]
    if not candidates:
        raise RuntimeError(f"No initial characterization xlsx found in: {initial_dir}")

    init_path = os.path.join(initial_dir, sorted(candidates)[0])
    df = read_all_channel_sheets(init_path)

    # keep necessary cols only
    for c in [CYC_COL, QD_COL]:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in Initial file: {init_path}")
    df[CYC_COL] = pd.to_numeric(df[CYC_COL], errors="coerce")
    df[QD_COL]  = pd.to_numeric(df[QD_COL], errors="coerce")
    df = df.dropna(subset=[CYC_COL, QD_COL])

    # discharge capacity per cycle: max
    q_by_cyc = df.groupby(CYC_COL)[QD_COL].max().sort_index()

    # pick first valid >0
    q_by_cyc = q_by_cyc[q_by_cyc > 0]
    if q_by_cyc.empty:
        raise RuntimeError(f"Could not compute Q0 from initial file: {init_path}")

    q0 = float(q_by_cyc.iloc[0])
    return q0, init_path

def extract_features_from_continuous(xlsx_path: str, q0: float, tempC: float, cond_id: int):
    df = read_all_channel_sheets(xlsx_path)

    needed = [V_COL, I_COL, T_COL, CYC_COL, QD_COL]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in continuous file: {xlsx_path}")

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[V_COL, I_COL, T_COL, CYC_COL])

    rows = []
    for cyc, g in df.groupby(CYC_COL, sort=True):
        charge = g[g[I_COL] > 0]
        discharge = g[g[I_COL] < 0]

        if len(charge) < 10:
            continue

        v = charge[V_COL].to_numpy(dtype=float)
        i = charge[I_COL].to_numpy(dtype=float)
        t = charge[T_COL].to_numpy(dtype=float)

        m = np.isfinite(v) & np.isfinite(i) & np.isfinite(t)
        v, i, t = v[m], i[m], t[m]
        if v.size < 10:
            continue

        # ---- feature windows (same spirit as Michigan) ----
        vend = np.nanmax(v)
        vw_mask = (v >= vend - 0.2) & (v <= vend)  # end-of-charge voltage window
        v_w = v[vw_mask]
        t_w = t[vw_mask]

        i_pos = i[i > 0]
        if i_pos.size < 5:
            continue
        i_cc_level = np.nanmedian(i_pos)

        # CC region = high current
        cc_mask = (i >= 0.80 * i_cc_level)
        i_cc = i[cc_mask]
        t_cc = t[cc_mask]

        # CV region = low-ish current near end-of-charge
        cv_mask = vw_mask & (i >= 0.05 * i_cc_level) & (i <= 0.25 * i_cc_level)
        i_cv = i[cv_mask]
        t_cv = t[cv_mask]

        # voltage stats (end-of-charge window)
        v_mean = np.nanmean(v_w) if v_w.size else np.nan
        v_std  = np.nanstd(v_w) if v_w.size else np.nan
        v_kurt = kurtosis_pearson(v_w) if v_w.size else np.nan
        v_skew = skewness(v_w) if v_w.size else np.nan
        v_slope= linear_slope(t_w, v_w) if v_w.size else np.nan
        v_ent  = entropy_1d(v_w) if v_w.size else np.nan

        # current stats (CV window)
        i_mean = np.nanmean(i_cv) if i_cv.size else np.nan
        i_std  = np.nanstd(i_cv) if i_cv.size else np.nan
        i_kurt = kurtosis_pearson(i_cv) if i_cv.size else np.nan
        i_skew = skewness(i_cv) if i_cv.size else np.nan
        i_slope= linear_slope(t_cv, i_cv) if i_cv.size else np.nan
        i_ent  = entropy_1d(i_cv) if i_cv.size else np.nan

        # CC/CV time and charge quantities
        cc_time = (np.nanmax(t_cc) - np.nanmin(t_cc)) if t_cc.size else np.nan
        cv_time = (np.nanmax(t_cv) - np.nanmin(t_cv)) if t_cv.size else np.nan
        cc_q = trapz_ah(t_cc, i_cc) if t_cc.size else np.nan
        cv_q = trapz_ah(t_cv, i_cv) if t_cv.size else np.nan

        # discharge capacity per cycle
        qd = discharge[QD_COL].to_numpy(dtype=float)
        qd = qd[np.isfinite(qd)]
        qd_cycle = np.nanmax(qd) if qd.size else np.nan
        if not np.isfinite(qd_cycle) or qd_cycle <= 0:
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
            "current slope": i_slope,
            "current entropy": i_ent,
            "cycle": float(cyc),
            "temp_C": float(tempC),
            "temp_K": float(tempC + 273.15),
            "condition_id": float(cond_id),
            "raw_Qd_Ah": float(qd_cycle),  # store raw discharge capacity; compute SOH later

        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No valid cycles extracted from continuous file: {xlsx_path}")

    out_ = out.sort_values("cycle").reset_index(drop=True)

    first_cycle = out["cycle"].min()
    q0_local = float(out.loc[out["cycle"] == first_cycle, "raw_Qd_Ah"].iloc[0])

    out["capacity"] = out["raw_Qd_Ah"] / q0_local
    out = out.drop(columns=["raw_Qd_Ah"])


    # ensure capacity last column
    cap = out.pop("capacity")
    out["capacity"] = cap
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Extension root folder")
    ap.add_argument("--mode", type=str, default="batch", choices=["single", "batch"])
    ap.add_argument("--continuous_root", type=str, required=True, help="Root folder of Continuous Cycling Data")
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
        lo, hi = parse_cycle_range(args.continuous_xlsx)
        suffix = f"cycles{lo:03d}_{hi:03d}" if lo is not None else "cycles"
        out_name = f"T{int(tempC)}C_cond{cond_id:02d}_{suffix}_processed.csv"
        out_path = os.path.join(out_dir, out_name)
        out_df.to_csv(out_path, index=False)
        print(f"[DONE] Saved: {out_path}")
        return

    # -------- batch mode --------
    cont_root = args.continuous_root

    if not os.path.isdir(cont_root):
        raise RuntimeError(f"Could not find Continuous Cycling Data folder at: {cont_root}")

    # find all DOE continuous excel files
    # Process all .xlsx files recursively across all cycle folders
    all_files = glob.glob(os.path.join(cont_root, "**", "DOE-*.xlsx"), recursive=True)



    if not all_files:
        raise RuntimeError(f"No DOE-*.xlsx files found under: {cont_root}")

    # group by (tempC, cond_id)
    groups = {}
    for p in all_files:
        tempC, cond_id = parse_temp_and_condition_id(p)
        if tempC is None or cond_id is None:
            # skip files that don't match the naming pattern
            continue
        groups.setdefault((tempC, cond_id), []).append(p)

    if not groups:
        raise RuntimeError("Found .xlsx files but none matched pattern: DOE-xxx-yyy-10DU 01.xlsx")

    print(f"[OK] Found {sum(len(v) for v in groups.values())} continuous files in {len(groups)} (temp,cond) groups.")

    for (tempC, cond_id), files in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        # sort by cycle range start (001, 051, ...)
        def sort_key(p):
            lo, hi = parse_cycle_range(p)
            return (lo if lo is not None else 1_000_000, hi if hi is not None else 1_000_000)
        files = sorted(files, key=sort_key)

        dfs = []
        for p in files:
            try:
                df_part = extract_features_from_continuous(p, q0=q0, tempC=tempC, cond_id=cond_id)
                dfs.append(df_part)
                lo, hi = parse_cycle_range(p)
                print(f"  ✓ T{int(tempC)}C cond{cond_id:02d}: {os.path.basename(p)} -> {len(df_part)} rows")
            except Exception as e:
                print(f"  ✗ T{int(tempC)}C cond{cond_id:02d}: {os.path.basename(p)} failed: {e}")

        if not dfs:
            print(f"[SKIP] T{int(tempC)}C cond{cond_id:02d}: no usable blocks.")
            continue

        full = pd.concat(dfs, ignore_index=True)

        # IMPORTANT: cycles may repeat across blocks; keep the latest occurrence
        full = full.sort_values("cycle").drop_duplicates(subset=["cycle"], keep="last").reset_index(drop=True)

        # ensure capacity last column
        cap = full.pop("capacity")
        full["capacity"] = cap

        out_name = f"T{int(tempC)}C_cond{cond_id:02d}_ALL_processed.csv"
        out_path = os.path.join(out_dir, out_name)
        full.to_csv(out_path, index=False)
        print(f"[DONE] Saved group: {out_path} (rows={len(full)}, cols={len(full.columns)})")


if __name__ == "__main__":
    main()
