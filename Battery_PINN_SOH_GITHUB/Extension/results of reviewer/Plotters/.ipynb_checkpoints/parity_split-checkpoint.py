import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

IN_DIR = r"/Users/setti/Desktop/Battery_PINN_SOH/Paper/results of reviewer/Michigan results_cell_split/0-0/parity_all"
OUT_PNG = os.path.join(IN_DIR, "michigan_parity_grid_2x5.png")

fig, axes = plt.subplots(2, 5, figsize=(18, 7), dpi=200)
axes = axes.ravel()

for i in range(10):
    k = i + 1
    img_path = os.path.join(IN_DIR, f"michigan_parity_exp{k}.png")
    img = mpimg.imread(img_path)
    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(f"Exp {k}", fontsize=12)

plt.tight_layout()
plt.savefig(OUT_PNG, bbox_inches="tight")
plt.close()
print("Saved:", OUT_PNG)
