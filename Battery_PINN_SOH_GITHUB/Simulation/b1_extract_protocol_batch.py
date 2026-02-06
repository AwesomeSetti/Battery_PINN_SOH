import os
import glob
import pandas as pd
import numpy as np

# EDIT THIS PATH to your folder that contains the xlsx files you want to scan
# Example:
# RAW_DIR = "/Users/setti/Desktop/Battery_PINN_SOH/Calce data/Continuous Cycling Data/001~050 Cycles"
RAW_DIR = r"/Users/setti/Desktop/Battery_PINN_SOH/Extension/Calce data/Continuous Cycling Data/001~050 Cycles"

OUT_DIR = "protocol_summaries"
os.makedirs(OUT_DIR, exist_ok=True)

REQUIRED_COLS = ["Test_Time(s)", "Step_Index", "Cycle_Index", "Voltage(V)", "Current(A)"]

def find_channel_sheet(xlsx_path):
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    # Prefer Channel sheets
    channel_sheets = [s for s in xl.sheet_names if "Channel" in s]
    if channel_sheets:
        return channel_sheets[0]  # typically only one relevant channel per file
    # fallback
    return xl.sheet_names[0]

def load_df(xlsx_path, sheet):
    df = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")
    # normalize column names (some files have extra spaces)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {xlsx_path} sheet {sheet}")
    return df[REQUIRED_COLS].copy()

def segment_steps(df):
    seg_id = (df["Step_Index"].ne(df["Step_Index"].shift())).cumsum()
    g = df.groupby(seg_id)

    rows = []
    for sid, d in g:
        step = int(d["Step_Index"].iloc[0])
        cyc  = int(d["Cycle_Index"].iloc[0])

        t0 = float(d["Test_Time(s)"].iloc[0])
        t1 = float(d["Test_Time(s)"].iloc[-1])
        dur = max(0.0, t1 - t0)

        I = d["Current(A)"].to_numpy()
        V = d["Voltage(V)"].to_numpy()

        rows.append({
            "seg": int(sid),
            "cycle": cyc,
            "step_index": step,
            "dur_s": dur,
            "I_mean": float(np.mean(I)),
            "I_min": float(np.min(I)),
            "I_max": float(np.max(I)),
            "V_start": float(V[0]),
            "V_end": float(V[-1]),
            "V_min": float(np.min(V)),
            "V_max": float(np.max(V)),
        })
    return pd.DataFrame(rows)

def label_step(r):
    # rule-based labeling
    if abs(r["I_mean"]) < 1e-3:
        return "Rest"
    if r["I_mean"] > 0:
        # positive → charge
        if (r["V_max"] - r["V_min"]) < 0.01:
            return "Hold (CV-ish)"
        return "Charge (CC-ish)"
    else:
        return "Discharge (CC-ish)"

def summarize_one_file(xlsx_path):
    sheet = find_channel_sheet(xlsx_path)
    df = load_df(xlsx_path, sheet)
    steps = segment_steps(df)
    steps["type"] = steps.apply(label_step, axis=1)

    # Cycle 1 (minimum cycle index in file)
    cmin = steps["cycle"].min()
    cycle1 = steps[steps["cycle"] == cmin].copy()

    # Build a compact "protocol signature" string (useful later)
    signature = " | ".join(
        f"{row['type']} ({row['dur_s']:.0f}s, I~{row['I_mean']:.2f}A, V[{row['V_min']:.2f},{row['V_max']:.2f}])"
        for _, row in cycle1.iterrows()
    )

    return sheet, cycle1, signature

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.xlsx")))
    if not files:
        raise FileNotFoundError(f"No .xlsx found in RAW_DIR: {RAW_DIR}")

    index_rows = []

    for fpath in files:
        base = os.path.basename(fpath)
        try:
            sheet, cycle1, sig = summarize_one_file(fpath)

            out_csv = os.path.join(OUT_DIR, base.replace(".xlsx", f"__{sheet}.csv"))
            cycle1.to_csv(out_csv, index=False)

            index_rows.append({
                "file": base,
                "sheet": sheet,
                "cycle1_min_cycle": int(cycle1["cycle"].min()),
                "n_steps_cycle1": int(len(cycle1)),
                "signature": sig,
                "out_csv": out_csv
            })

            print(f"[OK] {base} -> {out_csv}")

        except Exception as e:
            index_rows.append({
                "file": base,
                "sheet": None,
                "cycle1_min_cycle": None,
                "n_steps_cycle1": None,
                "signature": None,
                "out_csv": None,
                "error": str(e)
            })
            print(f"[FAIL] {base}: {e}")

    index_df = pd.DataFrame(index_rows)
    index_path = os.path.join(OUT_DIR, "INDEX_protocol_signatures.csv")
    index_df.to_csv(index_path, index=False)
    print("\nSaved index:", index_path)

if __name__ == "__main__":
    main()
