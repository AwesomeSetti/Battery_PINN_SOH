import json
import numpy as np
import pandas as pd
import os

R = 8.314
DATA_PATH = "data_processed/capacity_table.csv"
PARAM_PATH = "outputs/tables/fitted_params.json"

OUT_PRED = "outputs/tables/predictions_validation.csv"
OUT_METRICS = "outputs/tables/metrics_validation.csv"

np.random.seed(42)

# ---------- Load ----------
df = pd.read_csv(DATA_PATH)

with open(PARAM_PATH, "r") as f:
    params = json.load(f)

A = params["A"]
b = params["b"]
Ea = params["Ea_J_per_mol"]
s_c = params["s_c"]

# ---------- Choose validation conditions ----------
all_conditions = sorted(df["condition_id"].unique())
n_val = int(0.25 * len(all_conditions))

val_conditions = np.random.choice(all_conditions, size=n_val, replace=False)
cal_conditions = [c for c in all_conditions if c not in val_conditions]

print("Calibration conditions:", cal_conditions)
print("Validation conditions:", val_conditions.tolist())

# ---------- Model ----------
def simulate_u(N, T, cond):
    return 1.0 - A * (N ** b) * np.exp(-Ea / (R * T)) * s_c[str(int(cond))]

# ---------- Simulate validation trajectories ----------
rows = []

for cond in val_conditions:
    dfc = df[df["condition_id"] == cond]

    for traj_id, dft in dfc.groupby("traj_id"):
        dft = dft.sort_values("N")
        N = dft["N"].values
        T = dft["temp_K"].values
        u_true = dft["u_true"].values

        u_pred = simulate_u(N, T, cond)

        for i in range(len(N)):
            rows.append({
                "traj_id": traj_id,
                "condition_id": cond,
                "temp_C": dft["temp_C"].iloc[i],
                "N": N[i],
                "u_true": u_true[i],
                "u_pred": u_pred[i]
            })

pred_df = pd.DataFrame(rows)
os.makedirs("outputs/tables", exist_ok=True)
pred_df.to_csv(OUT_PRED, index=False)

print("Saved predictions:", OUT_PRED)

# ---------- Metrics ----------
metrics = []

for traj_id, dft in pred_df.groupby("traj_id"):
    u_t = dft["u_true"].values
    u_p = dft["u_pred"].values

    rmse = np.sqrt(np.mean((u_t - u_p) ** 2))
    mae = np.mean(np.abs(u_t - u_p))

    # EOL cycles
    def eol_cycle(u):
        idx = np.where(u <= 0.8)[0]
        return idx[0] if len(idx) else np.nan

    eol_true = eol_cycle(u_t)
    eol_pred = eol_cycle(u_p)

    metrics.append({
        "traj_id": traj_id,
        "rmse": rmse,
        "mae": mae,
        "eol_true": eol_true,
        "eol_pred": eol_pred,
        "eol_error_cycles": None if np.isnan(eol_true) or np.isnan(eol_pred)
                              else eol_pred - eol_true
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(OUT_METRICS, index=False)

print("Saved metrics:", OUT_METRICS)
print(metrics_df.describe())
