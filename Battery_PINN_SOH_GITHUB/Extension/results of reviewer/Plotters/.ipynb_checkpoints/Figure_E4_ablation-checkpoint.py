"""
E4 Ablation Summary Plot
CALCE: Baseline vs Arrhenius(T) vs Arrhenius+StressNetPlus
(mean ± std across 10 runs)
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
OUT_PNG = "calce_E4_ablation.png"
OUT_PDF = "calce_E4_ablation.pdf"

# ---------------------------------------

def metrics(true_y, pred_y):
    true_y = np.asarray(true_y).reshape(-1)
    pred_y = np.asarray(pred_y).reshape(-1)
    err = pred_y - true_y
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = 100.0 * np.mean(np.abs(err) / (np.abs(true_y) + 1e-12))
    return mae, mape, rmse

# collect results
results = {g: {"MAE": [], "MAPE": [], "RMSE": []} for g in GROUPS}

for gname, groot in GROUPS.items():
    for k in range(1, N_RUNS + 1):
        exp = os.path.join(groot, f"Experiment{k}")
        pred = np.load(os.path.join(exp, "pred_label.npy"))
        true = np.load(os.path.join(exp, "true_label.npy"))
        mae, mape, rmse = metrics(true, pred)
        results[gname]["MAE"].append(mae)
        results[gname]["MAPE"].append(mape)
        results[gname]["RMSE"].append(rmse)

# compute mean/std
metrics_order = ["MAE", "MAPE", "RMSE"]
means = np.zeros((len(GROUPS), len(metrics_order)))
stds  = np.zeros_like(means)

for gi, g in enumerate(GROUPS):
    for mi, m in enumerate(metrics_order):
        arr = np.array(results[g][m])
        means[gi, mi] = arr.mean()
        stds[gi, mi]  = arr.std(ddof=1)

# ---------------- PLOT ----------------
fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=200)

x = np.arange(len(metrics_order))
width = 0.22

colors = ["#9ecae1", "#6baed6", "#3182bd"]

for gi, g in enumerate(GROUPS):
    ax.bar(
        x + (gi - 1) * width,
        means[gi],
        yerr=stds[gi],
        width=width,
        label=g,
        color=colors[gi],
        capsize=4,
        edgecolor="black",
        linewidth=0.6
    )

ax.set_xticks(x)
ax.set_xticklabels(metrics_order)
ax.set_ylabel("Error")
ax.set_title("E4 Ablation Study on CALCE (10 runs, mean ± std)")
ax.legend(frameon=False)

plt.tight_layout()

if SAVE:
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved:\n  {OUT_PNG}\n  {OUT_PDF}")

plt.show()
