import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED_PATH = "outputs/tables/predictions_validation.csv"
METRICS_PATH = "outputs/tables/metrics_validation.csv"
PARAM_PATH = "outputs/tables/fitted_params_holdout.json"

OUT_DIR = "outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load ----------
pred = pd.read_csv(PRED_PATH)
metrics = pd.read_csv(METRICS_PATH)

with open(PARAM_PATH, "r") as f:
    params = json.load(f)

s_c = {int(k): float(v) for k, v in params["s_c"].items()}
val_conditions = params["val_conditions"]

print("Validation conditions:", val_conditions)

# ---------- FIG 1: SOH vs cycle overlay for a few validation trajectories ----------

metrics_sorted = metrics.sort_values("rmse", ascending=False)
traj_pick = metrics_sorted["traj_id"].head(4).tolist()

plt.figure()
for traj_id in traj_pick:
    d = pred[pred["traj_id"] == traj_id].sort_values("N")
    plt.plot(d["N"], d["u_true"], label=f"{traj_id} true")
    plt.plot(d["N"], d["u_pred"], linestyle="--", label=f"{traj_id} sim")
plt.xlabel("Cycle index N")
plt.ylabel("SOH (u)")
plt.title("Validation: SOH vs cycle (true vs simulated)")
plt.legend(fontsize=8)
plt.tight_layout()
f1 = os.path.join(OUT_DIR, "fig1_soh_overlay.png")
plt.savefig(f1, dpi=300)
print("Saved:", f1)
plt.close()

# ---------- Helper: EOL cycle from a curve ----------
def eol_from_curve(N, u, thresh=0.8):
    idx = np.where(u <= thresh)[0]
    if len(idx) == 0:
        return np.nan
    return int(N[idx[0]])

# ---------- FIG 2: EOL scatter (pred vs true) ----------
# compute EOL per trajectory directly from pred df (more reliable than metrics if NaNs)
eol_rows = []
for traj_id, d in pred.groupby("traj_id"):
    d = d.sort_values("N")
    N = d["N"].values
    u_t = d["u_true"].values
    u_p = d["u_pred"].values
    e_true = eol_from_curve(N, u_t, 0.8)
    e_pred = eol_from_curve(N, u_p, 0.8)
    eol_rows.append({"traj_id": traj_id, "eol_true": e_true, "eol_pred": e_pred})

eol_df = pd.DataFrame(eol_rows).dropna()

plt.figure()
plt.scatter(eol_df["eol_true"], eol_df["eol_pred"])
mn = min(eol_df["eol_true"].min(), eol_df["eol_pred"].min())
mx = max(eol_df["eol_true"].max(), eol_df["eol_pred"].max())
plt.plot([mn, mx], [mn, mx], linestyle="--")
plt.xlabel("True EOL cycle (u<=0.8)")
plt.ylabel("Predicted EOL cycle (u<=0.8)")
plt.title("EOL prediction (validation)")
plt.tight_layout()
f2 = os.path.join(OUT_DIR, "fig2_eol_scatter.png")
plt.savefig(f2, dpi=300)
print("Saved:", f2)
plt.close()

# ---------- FIG 3: Condition severity multipliers (s_c) ----------
sc_df = pd.DataFrame({
    "condition_id": sorted(s_c.keys()),
    "s_c": [s_c[c] for c in sorted(s_c.keys())]
})
# mark held-out conditions
sc_df["is_validation_condition"] = sc_df["condition_id"].isin([int(x) for x in val_conditions])

plt.figure()
plt.bar(sc_df["condition_id"].astype(int), sc_df["s_c"])
plt.xlabel("Condition ID")
plt.ylabel("Stress multiplier s_c")
plt.title("Protocol severity multipliers (all conditions)")
plt.tight_layout()
f3 = os.path.join(OUT_DIR, "fig3_condition_severity.png")
plt.savefig(f3, dpi=300)
print("Saved:", f3)
plt.close()

# ---------- Print quick summary for report ----------
print("\n=== Validation metrics summary ===")
print(metrics.describe())

print("\n=== EOL rows used in scatter ===")
print(eol_df)
