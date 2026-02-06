"""
CALCE parity plots (paper Fig.4a style), for multiple variants.
- Loads pred_label.npy and true_label.npy from Experiment1 (configurable)
- Works WITHOUT LaTeX
- Saves 3 png/pdf files: baseline, arrhenius_T45C, stressnetplus_T45C
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots

# ---- style like paper, but robust ----
plt.style.use(["science", "nature"])
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.default"] = "regular"
mpl.rcParams["font.family"] = "DejaVu Sans"

# ---------- CONFIG ----------
EXP_ID = 1  # which run to visualize (1..10)

VARIANTS = {
    "CALCE_Baseline": r"/Users/setti/Desktop/Battery_PINN_SOH/Extension/results/Part2A_CALCE_baseline/0-0",
    "Arrhenius_T45C": r"/Users/setti/Desktop/Battery_PINN_SOH/Extension/Extension_PartB_Arrhenius/results/Part2B_Arrhenius_T45C/0-0",
    "Arrhenius_StressNetPlus_T45C": r"/Users/setti/Desktop/Battery_PINN_SOH/Extension/Extension_PartB_Arrhenius/results/Part2C_Arrhenius_StressNetPlus_T45C/0-0",
}

# axis limits (we can adjust after first plot if needed)
XLIM = (0.7, 1.05)
YLIM = (0.7, 1.05)

SAVE = True
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots_out_calce")
# ----------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# paper-like colormap
color_list = ['#74AED4', '#7BDFF2', '#FBDD85', '#F46F43', '#CF3D3E']
cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_cmap", color_list, N=256)

def load_pred_true(root, exp_id):
    exp_folder = os.path.join(root, f"Experiment{exp_id}")
    pred_path = os.path.join(exp_folder, "pred_label.npy")
    true_path = os.path.join(exp_folder, "true_label.npy")
    if not (os.path.exists(pred_path) and os.path.exists(true_path)):
        raise FileNotFoundError(f"Missing pred/true in: {exp_folder}\n{pred_path}\n{true_path}")
    pred = np.load(pred_path).reshape(-1)
    true = np.load(true_path).reshape(-1)
    return pred, true

for name, root in VARIANTS.items():
    pred, true = load_pred_true(root, EXP_ID)
    err = np.abs(pred - true)

    fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=200)

    sc = ax.scatter(true, pred, c=err, cmap=cmap, s=6, alpha=0.75, vmin=0, vmax=0.1)
    ax.plot([XLIM[0], XLIM[1]], [XLIM[0], XLIM[1]], "--", c="#ff4d4e", linewidth=1)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True SOH")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{name} (Exp {EXP_ID})")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)

    fig.colorbar(sc, ax=ax, label="Absolute error")

    # stable layout
    fig.subplots_adjust(right=0.86)

    if SAVE:
        out_png = os.path.join(OUT_DIR, f"calce_parity_{name}_exp{EXP_ID}.png")
        out_pdf = os.path.join(OUT_DIR, f"calce_parity_{name}_exp{EXP_ID}.pdf")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        print("Saved:", out_png)

    plt.show()
