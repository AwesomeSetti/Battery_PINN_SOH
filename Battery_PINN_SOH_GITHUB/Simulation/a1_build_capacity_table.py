import os, glob, re
import pandas as pd
import numpy as np

# CHANGE THIS PATH to your folder on Mac:
PROCESSED_DIR = "/Users/setti/Desktop/Battery_PINN_SOH/Extension/CALCE_processed_fixed3"
OUT_CSV = "data_processed/capacity_table.csv"

def parse_name(fname):
    # expects: T10C_cond01_ALL_processed.csv
    base = os.path.basename(fname)
    m = re.match(r"T(\d+)C_cond(\d+)_ALL_processed\.csv", base)
    if not m:
        raise ValueError(f"Unexpected filename: {base}")
    T_C = int(m.group(1))
    cond = int(m.group(2))
    traj_id = f"T{T_C}C_cond{cond:02d}"
    return T_C, cond, traj_id

def main():
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "T*C_cond*_ALL_processed.csv")))
    assert len(files) > 0, f"No processed CSVs found in {PROCESSED_DIR}"

    rows = []
    for f in files:
        T_C, cond, traj_id = parse_name(f)
        df = pd.read_csv(f)

        # Basic required columns
        required = {"cycle", "capacity", "temp_K", "condition_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{f} missing columns: {missing}")

        d = df[["cycle", "capacity", "temp_K", "condition_id"]].copy()
        d = d.dropna()
        d = d.sort_values("cycle")

        # Filter obvious bad points
        d = d[(d["capacity"] > 0.2) & (d["capacity"] < 1.2)]  # keep sane SOH range

        d["traj_id"] = traj_id
        d["temp_C"] = T_C
        d["N"] = d["cycle"].astype(int) - 1
        d["u_true"] = d["capacity"].astype(float)

        rows.append(d)

    all_df = pd.concat(rows, ignore_index=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    all_df.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)
    print("Trajectories:", all_df["traj_id"].nunique())
    print("Temps:", sorted(all_df["temp_C"].unique().tolist()))
    print("Rows:", all_df.shape)
    print(all_df.head())

if __name__ == "__main__":
    main()
