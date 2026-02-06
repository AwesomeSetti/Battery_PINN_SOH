import numpy as np
import pandas as pd
from scipy.optimize import least_squares

R = 8.314  # J/mol/K
DATA_PATH = "data_processed/capacity_table.csv"

# ---------- Model ----------
def degradation_model(theta, N, T, cond_ids, n_cond):
    a, b_, e = theta[:3]
    z = theta[3:]

    A = np.exp(a)
    b = np.exp(b_)
    Ea = np.exp(e)

    s = np.exp(z)  # condition multipliers

    return 1.0 - A * (N ** b) * np.exp(-Ea / (R * T)) * s[cond_ids]

# ---------- Residual ----------
def residual(theta, data, n_cond):
    N = data["N"].values
    T = data["temp_K"].values
    cond_ids = data["cond_idx"].values
    u_true = data["u_true"].values

    u_pred = degradation_model(theta, N, T, cond_ids, n_cond)
    return u_pred - u_true

# ---------- Main ----------
def main():
    df = pd.read_csv(DATA_PATH)

    # Calibration set: exclude 60C
    calib = df[df["temp_C"] != 60].copy()

    # Map condition_id -> index
    cond_map = {cid: i for i, cid in enumerate(sorted(calib["condition_id"].unique()))}
    calib["cond_idx"] = calib["condition_id"].map(cond_map)

    n_cond = len(cond_map)

    # Initial guess (log-space)
    theta0 = np.zeros(3 + n_cond)
    theta0[0] = np.log(1e-4)      # A
    theta0[1] = np.log(0.7)       # b
    theta0[2] = np.log(4e4)       # Ea ~ 40 kJ/mol

    print("Fitting parameters...")
    res = least_squares(
        residual,
        theta0,
        args=(calib, n_cond),
        verbose=2,
        max_nfev=5000
    )

    # Save results
    theta = res.x
    out = {
        "A": float(np.exp(theta[0])),
        "b": float(np.exp(theta[1])),
        "Ea_J_per_mol": float(np.exp(theta[2])),
        "s_c": {int(k): float(np.exp(theta[3 + i])) for k, i in cond_map.items()}
    }

    import json, os
    os.makedirs("outputs/tables", exist_ok=True)
    with open("outputs/tables/fitted_params.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved fitted parameters to outputs/tables/fitted_params.json")
    print(out)

if __name__ == "__main__":
    main()
