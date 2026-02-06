"""
Paper-like Fig.4b for Michigan:
- Compares multiple groups (e.g., Baseline vs +Temp vs +Temp+Cond)
- Computes MAE / MAPE / RMSE across Experiment1..10 for each group
- Uses ONLY numpy + matplotlib (no pandas/seaborn, avoids pyarrow/numpy2 issues)
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'nature'])
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.default'] = 'regular'
mpl.rcParams['font.family'] = 'DejaVu Sans'

# ---------------- CONFIG ----------------
EXPERIMENTS = list(range(1, 11))  # 1..10

# Put your result folders here.
# For now you ONLY have Baseline. Leave the others commented until you create them.
GROUPS = {
    "Baseline": r"/Users/setti/Desktop/Battery_PINN_SOH/Paper/results of reviewer/Michigan results/0-0",
GROUPS = {
    "Baseline":   "/Users/setti/Desktop/Battery_PINN_SOH/Paper/results of reviewer/Michigan results/0-0",
    "+Temp":      "/path/to/T45C_results",
    "+Temp+Cond": "/path/to/Part2C_Arrhenius_StressNetPlus_T45C_results",
}

}

OUT_PNG = "michigan_fig4b_grouped.png"
OUT_PDF = "michigan_fig4b_grouped.pdf"
SAVE = True
# ----------------------------------------

def compute_metrics(pred, true):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    mae  = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true)**2))
    mape = np.mean(np.abs((pred - true) / (true + 1e-12))) * 100.0
    return mae, mape, rmse

# metrics_store[group][metric] = list of values over experiments
metrics_store = {g: {"MAE": [], "MAPE": [], "RMSE": []} for g in GROUPS.keys()}

for gname, groot in GROUPS.items():
    for exp_id in EXPERIMENTS:
        exp_folder = os.path.join(groot, f"Experiment{exp_id}")
        pred_path = os.path.join(exp_folder, "pred_label.npy")
        true_path = os.path.join(exp_folder, "true_label.npy")

        if not (os.path.exists(pred_path) and os.path.exists(true_path)):
            raise FileNotFoundError(f"[{gname}] Missing files in {exp_folder}")

        pred = np.load(pred_path)
        true = np.load(true_path)
        mae, mape, rmse = compute_metrics(pred, true)

        metrics_store[gname]["MAE"].append(mae)
        metrics_store[gname]["MAPE"].append(mape)
        metrics_store[gname]["RMSE"].append(rmse)

# ---- Plot: grouped violins per metric ----
metrics = ["MAE", "MAPE", "RMSE"]
group_names = list(GROUPS.keys())
ng = len(group_names)

fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.2), dpi=200)

base_x = np.arange(len(metrics))  # 0,1,2
width = 0.20 if ng <= 3 else 0.12
offsets = (np.arange(ng) - (ng - 1) / 2.0) * (width * 1.6)

# collect data in the order matplotlib.violinplot expects
positions = []
data_list = []

for mi, m in enumerate(metrics):
    for gi, g in enumerate(group_names):
        positions.append(base_x[mi] + offsets[gi])
        data_list.append(metrics_store[g][m])

vp = ax.violinplot(
    data_list,
    positions=positions,
    widths=width,
    showmeans=False,
    showmedians=False,
    showextrema=False
)

# style violins a bit
for body in vp["bodies"]:
    body.set_alpha(0.45)
    body.set_linewidth(0.8)

# add points + mean (red) + mean±std (black)
for idx, vals in enumerate(data_list):
    vals = np.array(vals)
    x = positions[idx]

    # points
    jitter = (np.random.rand(len(vals)) - 0.5) * width * 0.35
    ax.scatter(np.full_like(vals, x) + jitter, vals, s=10, c="k", alpha=0.6, zorder=3)

    mu = vals.mean()
    sd = vals.std()

    # mean line (red)
    ax.hlines(mu, x - width*0.45, x + width*0.45, colors="r", linewidth=1.3, zorder=4)
    # mean ± std (black vertical)
    ax.vlines(x, mu - sd, mu + sd, colors="k", linewidth=1.2, zorder=4)

ax.set_xticks(base_x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Error")
ax.set_title("Michigan (10 runs)")

# legend (use colored patches)
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor="gray", alpha=0.45, label=g) for g in group_names]
ax.legend(handles=legend_handles, loc="upper left", frameon=False)

fig.subplots_adjust(bottom=0.22, left=0.10, right=0.98, top=0.88)

if SAVE:
    print("Saving to:", os.path.abspath(OUT_PNG))
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

plt.show()

# print summary like paper table text
print("\nMichigan metrics across 10 runs (mean ± std):")
for g in group_names:
    print(f"\n{g}:")
    for m in metrics:
        vals = np.array(metrics_store[g][m])
        print(f"  {m}: {vals.mean():.6f} ± {vals.std():.6f}")
