import os
import numpy as np
import matplotlib.pyplot as plt

# EDIT THIS PATH:
ROOT = r"/Users/setti/Desktop/Battery_PINN_SOH/Extension/results of reviewer/Michigan results/0-0"

OUT_DIR = os.path.join(ROOT, "parity_all")
os.makedirs(OUT_DIR, exist_ok=True)

def parity_plot(true_y, pred_y, title, outpath):
    true_y = np.asarray(true_y).reshape(-1)
    pred_y = np.asarray(pred_y).reshape(-1)
    ae = np.abs(pred_y - true_y)

    plt.figure(figsize=(4.8, 4.2), dpi=220)
    plt.scatter(true_y, pred_y, c=ae, s=10, alpha=0.85)
    mn = min(true_y.min(), pred_y.min())
    mx = max(true_y.max(), pred_y.max())
    plt.plot([mn, mx], [mn, mx], "r--", lw=1.5)

    plt.title(title)
    plt.xlabel("True SOH")
    plt.ylabel("Prediction")
    cb = plt.colorbar()
    cb.set_label("Absolute error")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

for k in range(1, 11):
    exp_dir = os.path.join(ROOT, f"Experiment{k}")
    tpath = os.path.join(exp_dir, "true_label.npy")
    ppath = os.path.join(exp_dir, "pred_label.npy")

    if not (os.path.exists(tpath) and os.path.exists(ppath)):
        print(f"[Skip] Missing npy in {exp_dir}")
        continue

    true_y = np.load(tpath)
    pred_y = np.load(ppath)

    out_png = os.path.join(OUT_DIR, f"parity_exp{k}.png")
    parity_plot(true_y, pred_y, f"Michigan (Experiment {k})", out_png)
    print("Saved:", out_png)

print("\nDone. Check:", OUT_DIR)
