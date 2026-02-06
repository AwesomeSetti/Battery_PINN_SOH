"""
Michigan Fig4b-style distribution plot (MAE / MAPE / RMSE across 10 runs)
- NO pandas / NO seaborn (avoids numpy2/pyarrow issues)
- Reads pred_label.npy and true_label.npy from Experiment1..Experiment10
- Draws violin + scatter points + mean (red) + mean±std (black)
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots

# ---- style like the paper, but keep it robust (no LaTeX needed) ----
plt.style.use(["science", "nature"])
mpl.rcParams["text.usetex"] = False

# -------- CONFIG --------
MICHIGAN_ROOT = r"/Users/setti/Desktop/Battery_PINN_SOH/Paper/results of reviewer/Michigan results_cell_split/0-0"

N_RUNS = 10
SAVE = True
OUT_PNG = os.path.join(os.path.dirname(__file__), "michigan_split_fig4b_violin_matplotlib.png")
OUT_PDF = os.path.join(os.path.dirname(__file__), "michigan_split_fig4b_violin_matplotlib.pdf")
# ------------------------

def metrics(true_y, pred_y):
    true_y = np.asarray(true_y).reshape(-1)
    pred_y = np.asarray(pred_y).reshape(-1)
    err = pred_y - true_y
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    # MAPE in percent (paper reports %)
    eps = 1e-12
    mape = 100.0 * np.mean(np.abs(err) / (np.abs(true_y) + eps))
    return mae, mape, rmse

# Collect (10 runs) for each metric
MAE_list, MAPE_list, RMSE_list = [], [], []

for k in range(1, N_RUNS + 1):
    exp_dir = os.path.join(MICHIGAN_ROOT, f"Experiment{k}")
    pred_path = os.path.join(exp_dir, "pred_label.npy")
    true_path = os.path.join(exp_dir, "true_label.npy")

    if not (os.path.exists(pred_path) and os.path.exists(true_path)):
        print(f"[skip] missing files in {exp_dir}")
        continue

    pred = np.load(pred_path)
    true = np.load(true_path)

    mae, mape, rmse = metrics(true, pred)
    MAE_list.append(mae)
    MAPE_list.append(mape)
    RMSE_list.append(rmse)

data = [MAE_list, MAPE_list, RMSE_list]
labels = ["MAE", "MAPE", "RMSE"]

# Print mean/std (you asked for it)
print("\nMichigan metrics across runs:")
for name, arr in zip(labels, data):
    arr = np.asarray(arr)
    print(f"{name:>4}: mean={arr.mean():.6f}, std={arr.std(ddof=1):.6f}")

fig, ax = plt.subplots(figsize=(5.8, 3.6), dpi=200)

# Violin plot
parts = ax.violinplot(
    dataset=data,
    positions=[1, 2, 3],
    widths=0.8,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)

# Make violins paper-like (soft fill, black edge)
for pc in parts["bodies"]:
    pc.set_alpha(0.55)
    pc.set_linewidth(0.8)

# Scatter the 10 points (like the paper shows 10 points per violin)
rng = np.random.default_rng(0)
for i, arr in enumerate(data, start=1):
    arr = np.asarray(arr)
    x = i + rng.normal(0, 0.06, size=len(arr))  # tiny horizontal jitter
    ax.scatter(x, arr, s=16, c="k", alpha=0.55, zorder=3)

    # mean (red) and mean±std (black vertical)
    mu = arr.mean()
    sd = arr.std(ddof=1) if len(arr) > 1 else 0.0
    ax.plot([i - 0.18, i + 0.18], [mu, mu], color="red", linewidth=1.4, zorder=4)
    ax.plot([i, i], [mu - sd, mu + sd], color="black", linewidth=1.4, zorder=4)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(labels)
ax.set_ylabel("Error")
ax.set_title("Michigan (10 runs)")

ax.margins(x=0.08)

plt.tight_layout()

if SAVE:
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    print(f"\nSaved:\n  {OUT_PNG}\n  {OUT_PDF}")

plt.show()
