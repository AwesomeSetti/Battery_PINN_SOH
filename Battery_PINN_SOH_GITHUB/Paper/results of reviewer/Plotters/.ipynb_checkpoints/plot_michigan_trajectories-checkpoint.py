import os
import re
import numpy as np
import matplotlib.pyplot as plt

from dataloader import DF

# ---- EDIT THESE TWO PATHS ----
ROOT = r"/Users/setti/Desktop/Battery_PINN_SOH/Paper/results of reviewer/Michigan results_cell_split/0-0/"
DATA_ROOT = r"/Users/setti/Desktop/Battery_PINN_SOH/Data Michigan/fast-formation/Michigan_processed"
# -----------------------------

OUT_ROOT = os.path.join(ROOT, "traj_all_fixed_v2")
os.makedirs(OUT_ROOT, exist_ok=True)

# These MUST match the training run settings used in main_michigan_cell_split.py
TEST_RATIO  = 0.2
VALID_RATIO = 0.2
BASE_SEED   = 420   # default in main_michigan_cell_split.py. If you used different, change here.
N_EXPERIMENTS = 10

# Dummy args for DF
class _Args:
    normalization_method = "min-max"
    log_dir = None
    save_folder = None

df_helper = DF(_Args())


def extract_cell_id(path: str):
    name = os.path.basename(path)
    m = re.search(r"(?:Cell[_\-\s]*)(\d+)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)(?=\.csv$)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def split_cells(cell_ids, test_ratio, valid_ratio, seed):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(cell_ids))
    rng.shuffle(ids)

    n = len(ids)
    n_test = max(1, int(round(n * test_ratio)))
    n_valid = max(1, int(round(n * valid_ratio)))

    test_ids = ids[:n_test].tolist()
    valid_ids = ids[n_test:n_test + n_valid].tolist()
    train_ids = ids[n_test + n_valid:].tolist()

    if len(train_ids) == 0:
        train_ids = valid_ids[:-1]
        valid_ids = valid_ids[-1:]

    return train_ids, valid_ids, test_ids


def paths_for_ids(all_paths, wanted_ids):
    wanted = set(wanted_ids)
    out = []
    for p in all_paths:
        cid = extract_cell_id(p)
        if cid in wanted:
            out.append(p)
    return sorted(out)


def get_test_paths_for_experiment(exp_idx_1based: int):
    all_paths = sorted([
        os.path.join(DATA_ROOT, f)
        for f in os.listdir(DATA_ROOT)
        if f.lower().endswith(".csv")
    ])

    ids = []
    for p in all_paths:
        cid = extract_cell_id(p)
        if cid is not None:
            ids.append(cid)
    unique_ids = sorted(set(ids))
    if len(unique_ids) < 3:
        raise RuntimeError("Not enough Michigan cell CSVs found. Check DATA_ROOT and filenames.")

    exp_seed = BASE_SEED + (exp_idx_1based - 1)
    _, _, test_ids = split_cells(unique_ids, TEST_RATIO, VALID_RATIO, exp_seed)
    test_paths = paths_for_ids(all_paths, test_ids)
    return test_paths


def read_cycles_and_ytrue_with_df(csv_path: str):
    """
    Use the same delete_3_sigma as training, but DO NOT normalize the cycle axis:
    - nominal_capacity=None => DF.read_one_csv applies delete_3_sigma, but no feature normalization.
    Training used nominal_capacity=1.0 which doesn't change capacity anyway.
    """
    df = df_helper.read_one_csv(csv_path, nominal_capacity=None)

    # cycle column
    cyc_col = None
    for c in df.columns:
        if c.lower() == "cycle":
            cyc_col = c
            break
    if cyc_col is None:
        for c in df.columns:
            if c.lower() in ["cycle index", "cycle_index"]:
                cyc_col = c
                break
    if cyc_col is None:
        raise ValueError(f"No cycle column found after DF.read_one_csv() in {csv_path}")

    if "capacity" not in df.columns:
        raise ValueError(f"'capacity' column not found in {csv_path}")

    # training uses y[:-1]
    cycles = df[cyc_col].to_numpy()[:-1]
    y_true = df["capacity"].to_numpy()[:-1]
    n = len(y_true)
    return cycles, y_true, n


def plot_one_cell(cycles, y_true, y_pred, title, out_png):
    plt.figure(figsize=(7.2, 4.2), dpi=220)
    plt.plot(cycles, y_true, label="True SOH")
    plt.plot(cycles, y_pred, label="Predicted SOH")
    plt.xlabel("Cycle")
    plt.ylabel("SOH")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


for k in range(1, N_EXPERIMENTS + 1):
    exp_dir = os.path.join(ROOT, f"Experiment{k}")
    true_p = os.path.join(exp_dir, "true_label.npy")
    pred_p = os.path.join(exp_dir, "pred_label.npy")

    if not (os.path.isfile(true_p) and os.path.isfile(pred_p)):
        print(f"[Skip] Exp{k} missing true/pred npy")
        continue

    y_true_all = np.load(true_p).reshape(-1)
    y_pred_all = np.load(pred_p).reshape(-1)

    test_paths = get_test_paths_for_experiment(k)

    out_dir = os.path.join(OUT_ROOT, f"Experiment{k}")
    os.makedirs(out_dir, exist_ok=True)

    cursor = 0
    for p in test_paths:
        csv_path = p

        cycles, y_true_cell, n = read_cycles_and_ytrue_with_df(csv_path)

        if cursor + n > len(y_true_all):
            left = len(y_true_all) - cursor
            print(f"[Warn] Exp{k} would overflow at {os.path.basename(csv_path)} (need {n}, left {left})")
            break

        y_pred_cell = y_pred_all[cursor:cursor+n]
        cursor += n

        title = f"Exp{k} | {os.path.basename(csv_path)} | n={n}"
        out_png = os.path.join(out_dir, os.path.basename(csv_path).replace(".csv", "_traj.png"))
        plot_one_cell(cycles, y_true_cell, y_pred_cell, title, out_png)

    if cursor != len(y_true_all):
        print(f"[Warn] Exp{k} cursor mismatch: used {cursor}/{len(y_true_all)} test points")
    else:
        print(f"[OK] Exp{k} saved plots to: {out_dir} (used {cursor}/{len(y_true_all)} test points)")

print("\nDone.")
