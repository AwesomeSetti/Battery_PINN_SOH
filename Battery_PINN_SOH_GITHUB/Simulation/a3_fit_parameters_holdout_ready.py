import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import json, os

R = 8.314
DATA_PATH = "data_processed/capacity_table.csv"

def degradation_model(theta, N, T, cond_idx):
    a, b_, e = theta[:3]
    z = theta[3:]

    A = np.exp(a)
    b = np.exp(b_)
    Ea = np.exp(e)
    s = np.exp(z)  # one per condition index

    return 1.0 - A * (N ** b) * np.exp(-Ea / (R * T)) * s[cond_idx]

def residual(theta, data):
    N = data["N"].values
    T = data["temp_K"].values
    cond_idx = data["cond_idx"].values
    u_true = data["u_true"].values
    u_pred = degradation_model(theta, N, T, cond_idx)
    return u_pred - u_true

def main():
    df = pd.read_csv(DATA_PATH)

    # Build condition map using ALL conditions (1..24)
    all_conditions = sorted(df["condition_id"].unique())
    cond_map = {cid: i for i, cid in enumerate(all_conditions)}
    df["cond_idx"] = df["condition_id"].map(cond_map)

    n_cond = len(all_conditions)
    print("Total conditions:", n_cond, all_conditions)

    # Choose held-out (validation) conditions (same seed as before)
    np.random.seed(42)
    n_val = int(0.25 * n_cond)
    val_conditions = np.random.choice(all_conditions, size=n_val, replace=False)
    cal_conditions = [c for c in all_conditions if c not in val_conditions]

    print("Calibration conditions:", cal_conditions)
    print("Validation conditions:", val_conditions.tolist())

    calib = df[df["condition_id"].isin(cal_conditions)].copy()

    # Initial guess (log space)
    theta0 = np.zeros(3 + n_cond)
    theta0[0] = np.log(1e-4)   # A
    theta0[1] = np.log(0.7)    # b
    theta0[2] = np.log(4e4)    # Ea

    # Initialize s_c around 1
    theta0[3:] = np.log(1.0)

    res = least_squares(
        residual,
        theta0,
        args=(calib,),
        verbose=2,
        max_nfev=5000
    )

    theta = res.x
    out = {
        "A": float(np.exp(theta[0])),
        "b": float(np.exp(theta[1])),
        "Ea_J_per_mol": float(np.exp(theta[2])),
        "s_c": {int(cid): float(np.exp(theta[3 + cond_map[cid]])) for cid in all_conditions},
        "val_conditions": [int(x) for x in val_conditions],
        "cal_conditions": [int(x) for x in cal_conditions],
    }

    os.makedirs("outputs/tables", exist_ok=True)
    with open("outputs/tables/fitted_params_holdout.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved: outputs/tables/fitted_params_holdout.json")
    print("A,b,Ea:", out["A"], out["b"], out["Ea_J_per_mol"])

if __name__ == "__main__":
    main()
