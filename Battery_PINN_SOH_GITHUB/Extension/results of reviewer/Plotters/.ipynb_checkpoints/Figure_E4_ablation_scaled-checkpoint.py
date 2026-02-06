"""
E4 Ablation Study (Scaled)
CALCE: Baseline vs Arrhenius(T) vs Arrhenius+StressNetPlus
Separate y-axis per metric for visibility
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots

# ---------------- STYLE ----------------
plt.style.use(["science", "nature"])
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.default"] = "regular"
mpl.rcParams["font.family"] = "DejaVu Sans"

# ---------------- CONFIG ----------------
GROUPS = {
    "Baseline": r"/Users/setti/Desktop/Battery_PINN_SOH/Extension/results/Part2A_CALCE_baseline/0-0",
    "Arrhenius (T45C)": r"/Users/setti/Desktop/Battery_PINN_SOH/Extension/Extension_PartB_Arrhenius/results/Part2B_Arrhenius_T45C/0-0",
    "Arrhenius + StressNetPlus": r"/Users/setti/Desktop/Battery_PINN_SOH/Extension/Extension_PartB_Arrhenius/results/Part2C_Arrhenius_StressNetPlus_T45C/0-0",
}

N_RUNS = 10
SAVE = True
OUT_PNG = "calce_E4_ablation_scaled.png"
OUT_PDF = "calce_E4_ablation_scaled.pdf"

# --------------------------------------

def metrics(true_y, pred_y):
    true_y = np.asarray(true_y).reshape(-1)
    pred_y = np.asarray(pred_y).reshape(-1)
    err = pred_y - true_y
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = 100.0 * np.mean(np.abs(err) / (np.abs(true_y) + 1e-12))
    return mae, mape, rmse

metrics_order = ["MAE", "MAPE", "RMSE"]

results = {g: {m: [] for m in metrics_order} for g in GROUPS}

for gname, groot in GROUPS.items():
    for k in range(1, N_RUNS + 1):
        exp = os.path.join(groot, f"Experiment{k}")
        pred = np.load(os.path.join(exp, "pred_label.npy"))
        true = np.load(os.path.join(exp, "true_label.npy"))
        mae, mape, rmse = metrics(true, pred)
        results[gname]["MAE"].append(mae)
        results[gname]["MAPE"].append(mape)
        results[gname]["RMSE"].append(rmse)

means = {g: [np.mean(results[g][m]) for m in metrics_order] for g in GROUPS}
stds  = {g: [np.std(results[g][m], ddof=1) for m in metrics_order] for g in GROUPS}

# ---------------- PLOT ----------------
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.8), dpi=200, sharex=True)

colors = ["#9ecae1", "#6baed6", "#3182bd"]
width = 0.22
x = np.arange(len(GROUPS))

for mi, metric in enumerate(metrics_order):
    ax = axes[mi]
    for gi, g in enumerate(GROUPS):
        ax.bar(
            gi,
            means[g][mi],
            yerr=stds[g][mi],
            width=width,
            color=colors[gi],
            edgecolor="black",
            linewidth=0.6,
            capsize=4
        )

    ax.set_title(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(list(GROUPS.keys()), rotation=15)
    ax.set_ylabel("Error")

axes[0].legend(GROUPS.keys(), frameon=False, loc="upper left")

fig.suptitle("E4 Ablation Study on CALCE (10 runs, mean ± std)", y=1.02)
plt.tight_layout()

if SAVE:
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved:\n  {OUT_PNG}\n  {OUT_PDF}")

plt.show()
