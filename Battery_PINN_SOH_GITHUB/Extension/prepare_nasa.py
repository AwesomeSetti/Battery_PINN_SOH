import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew


# NASA README discharge cutoff voltages (you can verify in your README)
V_MIN_BY_BATT = {"B0005": 2.7, "B0006": 2.5, "B0007": 2.2, "B0018": 2.5}


# -------------------------
# Utilities (robust + auditable)
# -------------------------
def safe_float_array(x) -> np.ndarray:
    return pd.to_numeric(pd.Series(np.ravel(x)), errors="coerce").to_numpy(dtype=float)


def finite_mask(*arrs: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def shannon_entropy(x: np.ndarray, bins: int = 30) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return float("nan")
    return float(-(hist * np.log(hist)).sum())


def robust_slope(t: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]
    y = y[m]
    if t.size < 5:
        return float("nan")
    t0 = t - np.nanmin(t)
    if np.allclose(np.nanmax(t0), 0):
        return float("nan")
    return float(np.polyfit(t0, y, 1)[0])


def coulomb_count_Q_Ah(time_s: np.ndarray, current_a: np.ndarray) -> float:
    m = np.isfinite(time_s) & np.isfinite(current_a)
    t = time_s[m]
    i = current_a[m]
    if t.size < 2:
        return float("nan")
    order = np.argsort(t)
    t = t[order]
    i = i[order]
    dt = np.diff(t)
    q_as = np.sum(0.5 * (i[:-1] + i[1:]) * dt)
    return float(q_as / 3600.0)


def split_cc_cv(time_s: np.ndarray, voltage_v: np.ndarray, current_a: np.ndarray,
                v_target: float = 4.2, tol: float = 0.01):
    """
    NASA charging: CC until ~4.2V, then CV.
    We split at first index reaching v_target - tol.
    """
    m = finite_mask(time_s, voltage_v, current_a)
    t = time_s[m]
    v = voltage_v[m]
    i = current_a[m]
    if t.size < 5:
        return np.array([], dtype=int), np.array([], dtype=int), t, v, i

    idx = np.where(v >= (v_target - tol))[0]
    if idx.size == 0:
        # no CV portion detected; treat all as CC
        return np.arange(len(t)), np.array([], dtype=int), t, v, i

    k = int(idx[0])
    cc_idx = np.arange(0, k + 1)
    cv_idx = np.arange(k + 1, len(t))
    return cc_idx, cv_idx, t, v, i


def get_cycle_type(cycle_struct) -> str:
    """
    MATLAB structs can be awkward. This is robust for NASA's format.
    """
    try:
        # often stored as array(['charge'], dtype='<U6')
        t = cycle_struct["type"][0]
        return str(t).strip().lower()
    except Exception:
        try:
            return str(cycle_struct["type"]).strip().lower()
        except Exception:
            return ""


# -------------------------
# Core extraction
# -------------------------
def extract_battery_cycles(mat_path: str):
    batt_id = os.path.splitext(os.path.basename(mat_path))[0]
    mat = loadmat(mat_path)
    if batt_id not in mat:
        raise KeyError(f"Expected top-level key '{batt_id}' not found in {mat_path}")

    batt = mat[batt_id]
    cycles = batt["cycle"][0, 0]  # struct array (1, N)
    return batt_id, cycles


def make_feature_row_from_charge(data_struct, ambient_C: float):
    t = safe_float_array(data_struct["Time"])
    v = safe_float_array(data_struct["Voltage_measured"])
    i = safe_float_array(data_struct["Current_measured"])
    temp = safe_float_array(data_struct["Temperature_measured"])

    m = finite_mask(t, v, i)
    t2, v2, i2 = t[m], v[m], i[m]

    # Voltage features
    v_mean = float(np.nanmean(v2)) if v2.size else float("nan")
    v_std = float(np.nanstd(v2)) if v2.size else float("nan")
    v_kurt = float(kurtosis(v2, fisher=True, nan_policy="omit")) if v2.size > 5 else float("nan")
    v_skew = float(skew(v2, nan_policy="omit")) if v2.size > 5 else float("nan")
    v_slope = robust_slope(t2, v2)
    v_ent = shannon_entropy(v2)

    # Current features
    i_mean = float(np.nanmean(i2)) if i2.size else float("nan")
    i_std = float(np.nanstd(i2)) if i2.size else float("nan")
    i_kurt = float(kurtosis(i2, fisher=True, nan_policy="omit")) if i2.size > 5 else float("nan")
    i_skew = float(skew(i2, nan_policy="omit")) if i2.size > 5 else float("nan")
    i_slope = robust_slope(t2, i2)
    i_ent = shannon_entropy(i2)

    # CC/CV split
    cc_idx, cv_idx, tt, vv, ii = split_cc_cv(t, v, i, v_target=4.2, tol=0.01)
    cc_q = coulomb_count_Q_Ah(tt[cc_idx], np.abs(ii[cc_idx])) if cc_idx.size > 1 else float("nan")
    cv_q = coulomb_count_Q_Ah(tt[cv_idx], np.abs(ii[cv_idx])) if cv_idx.size > 1 else float("nan")
    cc_time = float(tt[cc_idx][-1] - tt[cc_idx][0]) if cc_idx.size > 1 else float("nan")
    cv_time = float(tt[cv_idx][-1] - tt[cv_idx][0]) if cv_idx.size > 1 else float("nan")

    temp_mean = float(np.nanmean(temp[np.isfinite(temp)])) if np.any(np.isfinite(temp)) else float("nan")

    feats = {
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
        "temp_C": temp_mean,
        "ambient_C": ambient_C,
    }
    return feats


def convert_one_battery(mat_path: str) -> pd.DataFrame:
    batt_id, cycles = extract_battery_cycles(mat_path)

    rows = []
    last_charge_feats = None
    discharge_cycle_idx = 0

    for j in range(cycles.shape[1]):
        c = cycles[0, j]
        ctype = get_cycle_type(c)

        if ctype == "charge":
            ambient_C = float(np.ravel(c["ambient_temperature"])[0])
            data = c["data"][0, 0]
            last_charge_feats = make_feature_row_from_charge(data, ambient_C)

        elif ctype == "discharge":
            data = c["data"][0, 0]
            cap = float(np.ravel(data["Capacity"])[0])  # Ah (provided by NASA)
            discharge_cycle_idx += 1

            feats = last_charge_feats if last_charge_feats is not None else {}

            v_min = V_MIN_BY_BATT.get(batt_id, float("nan"))
            row = {k: feats.get(k, float("nan")) for k in [
                "voltage mean","voltage std","voltage kurtosis","voltage skewness",
                "CC Q","CC charge time","voltage slope","voltage entropy",
                "current mean","current std","current kurtosis","current skewness",
                "CV Q","CV charge time","current slope","current entropy"
            ]}

            temp_C = feats.get("temp_C", float("nan"))
            row.update({
                "cycle": float(discharge_cycle_idx),
                "temp_C": temp_C,
                "temp_K": (temp_C + 273.15) if np.isfinite(temp_C) else float("nan"),
                "ambient_C": feats.get("ambient_C", float("nan")),

                # charging conditions (protocol descriptors)
                "charge_cc_A": 1.5,
                "charge_cv_V": 4.2,
                "charge_cutoff_A": 0.02,
                "discharge_cc_A": 2.0,
                "discharge_cutoff_V": v_min,

                # labels
                "capacity_ah": cap,
                "battery_id": batt_id,
                "condition_id": f"{batt_id}_roomT_protocol1",
            })

            rows.append(row)
            last_charge_feats = None  # next cycle should have a new charge

        else:
            # ignore 'impedance' cycles, etc.
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No discharge cycles extracted for {batt_id}")

    # SOH normalized by first valid discharge capacity
    base_idx = df["capacity_ah"].first_valid_index()
    base_q = float(df.loc[base_idx, "capacity_ah"]) if base_idx is not None else float("nan")
    df["SOH"] = df["capacity_ah"] / base_q if np.isfinite(base_q) and base_q > 0 else np.nan

    df = df.sort_values("cycle").reset_index(drop=True)
    return df


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder containing NASA .mat files (B0005.mat etc.)")
    ap.add_argument("--out_dir", required=True, help="Output folder for processed CSVs")
    ap.add_argument("--pattern", default="B*.mat", help="Glob pattern for mat files")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mats = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    if not mats:
        raise FileNotFoundError(f"No .mat files found under {args.root} with pattern {args.pattern}")

    manifest = []

    for mpath in mats:
        batt_id = os.path.splitext(os.path.basename(mpath))[0]
        try:
            df = convert_one_battery(mpath)
            out_path = os.path.join(args.out_dir, f"NASA_{batt_id}.csv")
            df.to_csv(out_path, index=False)

            manifest.append({
                "battery_id": batt_id,
                "source_mat": mpath,
                "out_csv": out_path,
                "n_cycles": len(df),
                "soh_start": float(df["SOH"].iloc[0]),
                "soh_end": float(df["SOH"].iloc[-1]),
                "cap_start_ah": float(df["capacity_ah"].iloc[0]),
                "cap_end_ah": float(df["capacity_ah"].iloc[-1]),
                "status": "OK",
                "error": "",
            })
            print(f"[OK] {batt_id}: wrote {out_path} with {len(df)} discharge cycles")

        except Exception as e:
            manifest.append({
                "battery_id": batt_id,
                "source_mat": mpath,
                "out_csv": "",
                "n_cycles": 0,
                "soh_start": np.nan,
                "soh_end": np.nan,
                "cap_start_ah": np.nan,
                "cap_end_ah": np.nan,
                "status": "FAIL",
                "error": repr(e),
            })
            print(f"[FAIL] {batt_id}: {repr(e)}")

    mf = pd.DataFrame(manifest)
    mf_path = os.path.join(args.out_dir, "manifest.csv")
    mf.to_csv(mf_path, index=False)
    print(f"\nWrote manifest: {mf_path}")


if __name__ == "__main__":
    main()
