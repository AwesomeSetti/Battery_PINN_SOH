"""
Michigan parity plot (paper Fig.4a style)
- Loads pred_label.npy and true_label.npy
- Works WITHOUT LaTeX (disables text.usetex after scienceplots style)
- Stable layout (no constrained_layout / no tight_layout issues with colorbar)
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots

# ---- Apply paper-like style, then force-disable LaTeX ----
plt.style.use(['science', 'nature'])
mpl.rcParams['text.usetex'] = False            # IMPORTANT: override style
mpl.rcParams['mathtext.default'] = 'regular'
mpl.rcParams['font.family'] = 'DejaVu Sans'

# ------------------ CONFIG ------------------
MICHIGAN_ROOT = r"/Users/setti/Desktop/Battery_PINN_SOH/Paper/results of reviewer/Michigan results/0-0"
EXP_ID = 1  # 1..10

# Adjust if your SOH range differs
XLIM = (0.8, 1.02)
YLIM = (0.8, 1.02)

SAVE = True
OUT_PNG = f"michigan_parity_exp{EXP_ID}.png"
OUT_PDF = f"michigan_parity_exp{EXP_ID}.pdf"
# -------------------------------------------

# paper-like colormap
color_list = ['#74AED4', '#7BDFF2', '#FBDD85', '#F46F43', '#CF3D3E']
colors = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', color_list, N=256)

# Load npy
exp_folder = os.path.join(MICHIGAN_ROOT, f"Experiment{EXP_ID}")
pred_path = os.path.join(exp_folder, "pred_label.npy")
true_path = os.path.join(exp_folder, "true_label.npy")

if not (os.path.exists(pred_path) and os.path.exists(true_path)):
    raise FileNotFoundError(
        f"Could not find pred/true npy in: {exp_folder}\n"
        f"Expected:\n  {pred_path}\n  {true_path}"
    )

pred_label = np.load(pred_path).reshape(-1)
true_label = np.load(true_path).reshape(-1)

error = np.abs(pred_label - true_label)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.1), dpi=200)

sc = ax.scatter(
    true_label, pred_label,
    c=error, cmap=colors,
    s=6, alpha=0.75,
    vmin=0, vmax=0.1
)

# diagonal y=x
ax.plot([XLIM[0], XLIM[1]], [XLIM[0], XLIM[1]], '--', c='#ff4d4e', linewidth=1)

ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('True SOH')
ax.set_ylabel('Prediction')
ax.set_title(f"Michigan (Experiment {EXP_ID})")
ax.set_xlim(XLIM)
ax.set_ylim(YLIM)

# colorbar
fig.colorbar(sc, ax=ax, label='Absolute error')

# Stable layout: reserve space on the right (avoids constrained/tight_layout crashes)
fig.subplots_adjust(right=0.86)

# Save
if SAVE:
    print("Saving to:", os.path.abspath(OUT_PNG))
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

plt.show()
