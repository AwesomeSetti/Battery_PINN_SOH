"""
CALCE Fig4b-style grouped distribution plot (MAE/MAPE/RMSE across 10 runs)
- Compares multiple variants on the SAME dataset (CALCE)
- NO pandas/seaborn (safe with your numpy2 environment)
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "nature"])
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.default"] = "regular"
mpl.rcParams["font.family"] = "DejaVu Sans"

# -------- CONFIG --------
GROUPS = {
    "Baseline": r"/Users/setti/Desktop/Battery_PINN_SOH/Extension/results/Part2A_CALCE_baseline/0-0",
    "Arrhenius (T45C)": r"/Users/setti/Desktop/Battery_PINN_SOH/Extension/Extension_PartB_Arrhenius/results/Part2B_Arrhenius_T45C/0-0",
    "Arrhenius+StressNetPlus (T45C)": r"/Users/setti/Desktop/Battery_PINN_SOH/Extension/Extension_PartB_Arrhenius/results/Part2C_Arrhenius_StressNetPlus_T45C/0-0",
}
N_RUNS = 10
SAVE = True
OUT_PNG = "calce_fig4b_grouped.png"
OUT_PDF = "calce_fig4b_grouped.pdf"
# ------------------------

def metrics(true_y, pred_y):
    true_y = np.asarray(true_y).reshape(-1)
    pred_y = np.asarray(pred_y).reshape(-1)
    err = pred_y - true_y
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    eps = 1e-12
    mape = 100.0 * np.mean(np.abs(err) / (np.abs(true_y) + eps))  # percent
    return mae, mape, rmse

# values[group][metric] = list over runs
values = {g: {"MAE": [], "MAPE": [], "RMSE": []} for g in GROUPS}

for gname, groot in GROUPS.items():
    for k in range(1, N_RUNS + 1):
        exp_dir = os.path.join(groot, f"Experiment{k}")
        pred_path = os.path.join(exp_dir, "pred_label.npy")
        true_path = os.path.join(exp_dir, "true_label.npy")
        if not (os.path.exists(pred_path) and os.path.exists(true_path)):
            raise FileNotFoundError(f"[{gname}] Missing pred/true in {exp_dir}")

        pred = np.load(pred_path)
        true = np.load(true_path)
        mae, mape, rmse = metrics(true, pred)

        values[gname]["MAE"].append(mae)
        values[gname]["MAPE"].append(mape)
        values[gname]["RMSE"].append(rmse)

# print mean±std for your report
print("\nCALCE metrics across 10 runs (mean ± std):")
for gname in GROUPS:
    print(f"\n{gname}:")
    for m in ["MAE", "MAPE", "RMSE"]:
        arr = np.asarray(values[gname][m])
        print(f"  {m}: {arr.mean():.6f} ± {arr.std(ddof=1):.6f}")

# ---- plot grouped violins ----
metrics_order = ["MAE", "MAPE", "RMSE"]
group_names = list(GROUPS.keys())
ng = len(group_names)

fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=200)

base_x = np.arange(len(metrics_order))  # 0,1,2
width = 0.18
offsets = (np.arange(ng) - (ng - 1) / 2.0) * (width * 1.8)

data_list, pos_list = [], []
for mi, m in enumerate(metrics_order):
    for gi, g in enumerate(group_names):
        pos_list.append(base_x[mi] + offsets[gi])
        data_list.append(values[g][m])

vp = ax.violinplot(
    dataset=data_list,
    positions=pos_list,
    widths=width,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)

for body in vp["bodies"]:
    body.set_alpha(0.50)
    body.set_linewidth(0.8)

rng = np.random.default_rng(0)
idx = 0
for mi, m in enumerate(metrics_order):
    for gi, g in enumerate(group_names):
        arr = np.asarray(values[g][m])
        x0 = pos_list[idx]

        jitter = rng.normal(0, width * 0.12, size=len(arr))
        ax.scatter(x0 + jitter, arr, s=14, c="k", alpha=0.55, zorder=3)

        mu = arr.mean()
        sd = arr.std(ddof=1)
        ax.plot([x0 - width*0.45, x0 + width*0.45], [mu, mu], color="red", linewidth=1.4, zorder=4)
        ax.plot([x0, x0], [mu - sd, mu + sd], color="black", linewidth=1.4, zorder=4)
        idx += 1

ax.set_xticks(base_x)
ax.set_xticklabels(metrics_order)
ax.set_ylabel("Error")
ax.set_title("CALCE (10 runs): Baseline vs Arrhenius(T) vs Arrhenius+StressNetPlus")

from matplotlib.patches import Patch
ax.legend([Patch(facecolor="gray", alpha=0.50) for _ in group_names],
          group_names, frameon=False, loc="upper left")

plt.tight_layout()

if SAVE:
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nSaved:\n  {os.path.abspath(OUT_PNG)}\n  {os.path.abspath(OUT_PDF)}")

plt.show()
