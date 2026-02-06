import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = r"/Users/setti/Desktop/Battery_PINN_SOH/Paper/results of reviewer/Michigan results_cell_split/0-0"
OUT_DIR = os.path.join(ROOT, "parity_all")
os.makedirs(OUT_DIR, exist_ok=True)

def make_parity(true_y, pred_y, title, out_png):
    true_y = np.asarray(true_y).reshape(-1)
    pred_y = np.asarray(pred_y).reshape(-1)
    ae = np.abs(pred_y - true_y)

    plt.figure(figsize=(4.8, 4.2), dpi=220)
    sc = plt.scatter(true_y, pred_y, c=ae, s=10, alpha=0.85)
    mn = float(min(true_y.min(), pred_y.min()))
    mx = float(max(true_y.max(), pred_y.max()))
    plt.plot([mn, mx], [mn, mx], "r--", lw=1.5)
    plt.xlabel("True SOH")
    plt.ylabel("Prediction")
    plt.title(title)
    cb = plt.colorbar(sc)
    cb.set_label("Absolute error")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

found = 0
for k in range(1, 11):
    exp_dir = os.path.join(ROOT, f"Experiment{k}")
    true_p = os.path.join(exp_dir, "true_label.npy")
    pred_p = os.path.join(exp_dir, "pred_label.npy")

    if not (os.path.isfile(true_p) and os.path.isfile(pred_p)):
        print(f"[Skip] Missing npy in {exp_dir}")
        continue

    true_y = np.load(true_p)
    pred_y = np.load(pred_p)

    out_png = os.path.join(OUT_DIR, f"michigan_parity_exp{k}.png")
    make_parity(true_y, pred_y, f"Michigan (Experiment {k})", out_png)
    print("Saved:", out_png)
    found += 1

print(f"\nDone. Saved {found} parity plots to:\n{OUT_DIR}")
