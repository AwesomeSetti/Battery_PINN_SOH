import re
import numpy as np

log_path = "/Users/setti/Desktop/Battery_PINN_SOH/Extension/results/Part2A_CALCE_baseline/0-0/Experiment1/logging.txt"

experiments = {}
current = None

with open(log_path, "r") as f:
    for line in f:
        line = line.strip()

        if line.startswith("=== Experiment"):
            name = line.replace("=== ", "").replace(" ===", "")
            experiments[name] = {"MSE": [], "MAE": [], "MAPE": [], "RMSE": []}
            current = name

        elif "[Test]" in line and current is not None:
            matches = re.findall(r"(MSE|MAE|MAPE|RMSE): ([0-9.]+)", line)
            for k, v in matches:
                experiments[current][k].append(float(v))

for exp in sorted(experiments.keys()):
    print(exp, end=" ")
    for m in ["MSE", "MAE", "MAPE", "RMSE"]:
        arr = np.array(experiments[exp][m])
        if len(arr) == 0:
            print("& NA $\\pm$ NA", end=" ")
        else:
            print(f"& {arr.mean():.6f} $\\pm$ {arr.std(ddof=1):.6f}", end=" ")
    print("\\\\")


xx
