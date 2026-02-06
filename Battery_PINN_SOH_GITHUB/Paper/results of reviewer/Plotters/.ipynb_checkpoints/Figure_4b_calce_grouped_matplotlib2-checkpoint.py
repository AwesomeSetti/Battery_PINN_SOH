"""
CALCE Fig4b-style grouped violin plot (MAE/MAPE/RMSE across 10 runs)
- Compares multiple variants on the SAME dataset (CALCE)
- NO pandas/seaborn (safe with your numpy2 environment)
- FIXED: distinct colors + proper legend + non-overlapping grouped violins
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.patches import Patch

# ---- style like paper, but robust ----
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
OUT_PNG = "calce_fig4b_grouped_colored.png"
OUT_PDF = "calce_fig4b_grouped_colored.pdf"

# distinct colors (paper-ish)
GROUP_COLORS = {
    "Baseline": "#BDBDBD",                        # gray
    "Arrhenius (T45C)": "#74AED4",                # blue
    "Arrhenius+StressNetPlus (T45C)": "#F46F43",  # orange/red
}
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

# print mean±std for report
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

fig, ax = plt.subplots(figsize=(8.6, 4.1), dpi=200)

base_x = np.arange(len(metrics_order))  # [0,1,2] for MAE,MAPE,RMSE

# spacing control
width = 0.22                     # violin width
group_spacing = 0.28             # distance between groups within each metric
offsets = (np.arange(ng) - (ng - 1) / 2.0) * group_spacing

# build dataset list + positions in a consistent order
data_list, pos_list, color_list = [], [], []
for mi, m in enumerate(metrics_order):
    for gi, g in enumerate(group_names):
        pos_list.append(base_x[mi] + offsets[gi])
        data_list.append(values[g][m])
        color_list.append(GROUP_COLORS.get(g, "#CCCCCC"))

vp = ax.violinplot(
    dataset=data_list,
    positions=pos_list,
    widths=width,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)

# apply colors per violin (IMPORTANT FIX)
for body, c in zip(vp["bodies"], color_list):
    body.set_facecolor(c)
    body.set_edgecolor("black")
    body.set_alpha(0.65)
    body.set_linewidth(0.8)

# scatter points + mean/std overlays
rng = np.random.default_rng(0)
idx = 0
for mi, m in enumerate(metrics_order):
    for gi, g in enumerate(group_names):
        arr = np.asarray(values[g][m])
        x0 = pos_list[idx]
        c = color_list[idx]

        # jittered points (slightly tinted)
        jitter = rng.normal(0, width * 0.10, size=len(arr))
        ax.scatter(x0 + jitter, arr, s=16, c="k", alpha=0.50, zorder=3)

        # mean (red) and mean±std (black)
        mu = arr.mean()
        sd = arr.std(ddof=1)
        ax.plot([x0 - width*0.45, x0 + width*0.45], [mu, mu],
                color="red", linewidth=1.6, zorder=4)
        ax.plot([x0, x0], [mu - sd, mu + sd],
                color="black", linewidth=1.6, zorder=4)

        idx += 1

# x ticks and labels
ax.set_xticks(base_x)
ax.set_xticklabels(metrics_order)
ax.set_ylabel("Error")
ax.set_title("CALCE (10 runs): Baseline vs Arrhenius(T) vs Arrhenius+StressNetPlus")

# legend (FIXED: color-matched)
legend_handles = [
    Patch(facecolor=GROUP_COLORS[g], edgecolor="black", alpha=0.65, label=g)
    for g in group_names
]
ax.legend(handles=legend_handles, frameon=False, loc="upper left")

# improve readability: add faint vertical separators between metrics
for x in [0.5, 1.5]:
    ax.axvline(x, color="k", alpha=0.08, linewidth=1)

plt.tight_layout()

if SAVE:
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nSaved:\n  {os.path.abspath(OUT_PNG)}\n  {os.path.abspath(OUT_PDF)}")

plt.show()
