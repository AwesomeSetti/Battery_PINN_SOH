"""
main_michigan.py

Run the paper's PINN training loop on **processed Michigan per-cycle feature CSVs**.

This version is UPDATED to do a clean, rigorous split:

✅ Train / Valid / Test are split BY CELL (no row-level leakage across splits)
✅ 10 experiments are TRUE repeated runs with different random cell splits (seed + experiment index)
✅ Saves a small "split.txt" file in each Experiment folder so you can report what cells were used

Expected Michigan processed CSV format:
  feature columns ... , capacity
Optionally a "cycle" column may exist; it will be treated as a feature.

Notes:
- Your Michigan processed CSVs already have capacity normalized to SOH.
  Therefore, we set nominal_capacity=1.0 so the dataloader's SOH normalization is a no-op.
"""

from __future__ import annotations

import os
import sys
import argparse
import re
import random
from typing import List, Optional, Tuple, Dict

# Ensure we can import sibling files: dataloader.py, Model.py, util.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Battery_PINN_SOH/Paper
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from dataloader import DF
from Model import PINN
from util import write_to_txt

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


# -----------------------------
# Dataset wrapper
# -----------------------------
class MichiganData(DF):
    """Michigan dataset wrapper.

    Uses the same CSV reading + pairing logic as DF.
    The only dataset-specific difference is the nominal_capacity used
    for SOH normalization.
    """

    def __init__(self, root: str, args: argparse.Namespace):
        super().__init__(args)
        self.root = root
        self.file_list = sorted([f for f in os.listdir(root) if f.lower().endswith(".csv")])
        self.num = len(self.file_list)

        # Michigan processed CSVs already have capacity normalized to SOH.
        # Setting nominal_capacity=1.0 keeps y unchanged while still normalizing X features.
        self.nominal_capacity = 1.0 if self.normalization else None

    def read_all(self, specific_path_list: Optional[List[str]] = None):
        if specific_path_list is None:
            paths = [os.path.join(self.root, f) for f in self.file_list]
        else:
            paths = specific_path_list
        return self.load_all_battery(path_list=paths, nominal_capacity=self.nominal_capacity)


# -----------------------------
# Cell ID extraction
# -----------------------------
def _extract_cell_id(path: str) -> Optional[int]:
    """Try to extract a battery/cell id from filename.

    Supports patterns like:
      Michigan_Cell_11.csv
      Cell_11.csv
      ..._cell11.csv
      ..._11.csv

    Returns None if no id is found.
    """
    name = os.path.basename(path)
    m = re.search(r"(?:Cell[_\-\s]*)(\d+)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)(?=\.csv$)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _paths_to_id_map(paths: List[str]) -> Dict[str, Optional[int]]:
    return {p: _extract_cell_id(p) for p in paths}


# -----------------------------
# Split logic (CELL-LEVEL)
# -----------------------------
def _split_cells_random(ids: List[int], ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """Return (keep_ids, split_ids) where split_ids has size ~ratio*len(ids)."""
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n_split = max(1, int(round(len(ids) * ratio)))
    split_ids = sorted(ids[:n_split])
    keep_ids = sorted(ids[n_split:])
    return keep_ids, split_ids


def _make_cell_splits(
    all_paths: List[str],
    test_cells: Optional[List[int]],
    valid_cells: Optional[List[int]],
    test_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """
    Split paths into train/valid/test BY CELL.

    Priority:
      1) Explicit test_cells / valid_cells (if provided)
      2) Otherwise random by ratio using seed

    Returns:
      train_paths, valid_paths, test_paths,
      train_ids, valid_ids, test_ids
    """
    path2id = _paths_to_id_map(all_paths)
    known_ids = sorted({cid for cid in path2id.values() if cid is not None})
    unknown_paths = [p for p, cid in path2id.items() if cid is None]

    # --- TEST split ---
    if test_cells and len(test_cells) > 0:
        test_id_set = set(int(x) for x in test_cells)
        remaining_ids = [i for i in known_ids if i not in test_id_set]
        test_ids = sorted(test_id_set.intersection(set(known_ids)))
    else:
        remaining_ids, test_ids = _split_cells_random(known_ids, test_ratio, seed)

    # --- VALID split (from remaining) ---
    if valid_cells and len(valid_cells) > 0:
        valid_id_set = set(int(x) for x in valid_cells)
        train_ids = [i for i in remaining_ids if i not in valid_id_set]
        valid_ids = sorted(valid_id_set.intersection(set(remaining_ids)))
    else:
        train_ids, valid_ids = _split_cells_random(remaining_ids, valid_ratio, seed + 1337)

    train_id_set = set(train_ids)
    valid_id_set = set(valid_ids)
    test_id_set = set(test_ids)

    train_paths = [p for p, cid in path2id.items() if (cid in train_id_set)]
    valid_paths = [p for p, cid in path2id.items() if (cid in valid_id_set)]
    test_paths  = [p for p, cid in path2id.items() if (cid in test_id_set)]

    # Put unknown-id files into TRAIN (safe default)
    train_paths += unknown_paths

    # de-dup + sort for reproducibility
    train_paths = sorted(set(train_paths))
    valid_paths = sorted(set(valid_paths))
    test_paths = sorted(set(test_paths))

    return train_paths, valid_paths, test_paths, train_ids, valid_ids, test_ids


def _write_split_file(save_folder: str, train_ids: List[int], valid_ids: List[int], test_ids: List[int],
                      train_paths: List[str], valid_paths: List[str], test_paths: List[str]) -> None:
    out = []
    out.append("=== CELL SPLIT (CELL-LEVEL) ===")
    out.append(f"Train cell ids ({len(train_ids)}): {train_ids}")
    out.append(f"Valid cell ids ({len(valid_ids)}): {valid_ids}")
    out.append(f"Test  cell ids ({len(test_ids)}): {test_ids}")
    out.append("")
    out.append("=== FILES ===")
    out.append(f"Train files ({len(train_paths)}):")
    out.extend([f"  {p}" for p in train_paths])
    out.append(f"\nValid files ({len(valid_paths)}):")
    out.extend([f"  {p}" for p in valid_paths])
    out.append(f"\nTest files ({len(test_paths)}):")
    out.extend([f"  {p}" for p in test_paths])

    with open(os.path.join(save_folder, "split.txt"), "w") as f:
        f.write("\n".join(out) + "\n")


# -----------------------------
# Data loader builder
# -----------------------------
def load_data(args: argparse.Namespace):
    root = args.root
    data = MichiganData(root=root, args=args)

    all_paths = [os.path.join(root, f) for f in data.file_list]
    if args.small_sample is not None:
        all_paths = all_paths[: int(args.small_sample)]

    train_list, valid_list, test_list, train_ids, valid_ids, test_ids = _make_cell_splits(
        all_paths=all_paths,
        test_cells=args.test_cells,
        valid_cells=args.valid_cells,
        test_ratio=args.test_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    # IMPORTANT:
    # We avoid DF's internal row-split (train_2/valid_2).
    # Instead, we treat each subset as its own "dataset" and use its test_3 as the full set.
    train_pack = data.read_all(specific_path_list=train_list)
    valid_pack = data.read_all(specific_path_list=valid_list) if len(valid_list) > 0 else None
    test_pack  = data.read_all(specific_path_list=test_list) if len(test_list) > 0 else None

    dataloader = {
        "train": train_pack["test_3"],  # full train subset, no internal split
        "valid": valid_pack["test_3"] if valid_pack is not None else train_pack["valid"],  # fallback
        "test":  test_pack["test_3"] if test_pack is not None else train_pack["test"],     # fallback
        "_meta": {
            "train_ids": train_ids,
            "valid_ids": valid_ids,
            "test_ids": test_ids,
            "train_paths": train_list,
            "valid_paths": valid_list,
            "test_paths": test_list,
        }
    }
    return dataloader


# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()

    for e in range(args.n_experiments):
        # Make each experiment a different split (paper-style repeated runs)
        args.seed = int(args.base_seed) + e

        save_folder = os.path.join(args.results_root, "0-0", f"Experiment{e + 1}")
        os.makedirs(save_folder, exist_ok=True)

        setattr(args, "save_folder", save_folder)
        setattr(args, "log_dir", args.log_name)

        dl = load_data(args)

        # Save split info for reporting/debugging
        meta = dl["_meta"]
        _write_split_file(
            save_folder=save_folder,
            train_ids=meta["train_ids"],
            valid_ids=meta["valid_ids"],
            test_ids=meta["test_ids"],
            train_paths=meta["train_paths"],
            valid_paths=meta["valid_paths"],
            test_paths=meta["test_paths"],
        )

        pinn = PINN(args)
        pinn.Train(trainloader=dl["train"], validloader=dl["valid"], testloader=dl["test"])


# -----------------------------
# Args
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser("Hyper Parameters for Michigan dataset (processed CSVs)")

    # data paths
    parser.add_argument(
        "--root",
        type=str,
        default="../Data Michigan/fast-formation/Michigan_processed",
        help="Folder containing processed per-cell CSVs (paper-style columns, ending with capacity)",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results of reviewer/Michigan results",
        help="Root folder to store experiment outputs",
    )
    parser.add_argument("--log_name", type=str, default="logging.txt", help="Log filename inside each experiment")

    # split controls (CELL-LEVEL)
    parser.add_argument("--test_cells", type=int, nargs="*", default=None,
                        help="Cell IDs to reserve for test (e.g., --test_cells 4 8). Overrides --test_ratio.")
    parser.add_argument("--valid_cells", type=int, nargs="*", default=None,
                        help="Cell IDs to reserve for validation. Overrides --valid_ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="If --test_cells not provided, reserve this fraction of cells for test.")
    parser.add_argument("--valid_ratio", type=float, default=0.2,
                        help="Fraction of remaining (non-test) cells reserved for validation.")
    parser.add_argument("--base_seed", type=int, default=420,
                        help="Base seed; experiment e uses seed = base_seed + e.")

    # repeat experiments
    parser.add_argument("--n_experiments", type=int, default=10, help="Number of repeated runs")
    parser.add_argument("--small_sample", type=int, default=None,
                        help="Optional: only use the first N CSV files in the folder (debug)")

    # match paper args (kept same names used by Model.py)
    parser.add_argument("--data", type=str, default="Michigan", help="dataset name label")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--normalization_method", type=str, default="min-max", help="min-max,z-score")

    # scheduler related
    parser.add_argument("--epochs", type=int, default=200, help="epoch")
    parser.add_argument("--early_stop", type=int, default=20, help="early stop")
    parser.add_argument("--warmup_epochs", type=int, default=30, help="warmup epoch")
    parser.add_argument("--warmup_lr", type=float, default=0.002, help="warmup lr")
    parser.add_argument("--lr", type=float, default=0.01, help="base lr")
    parser.add_argument("--final_lr", type=float, default=0.0002, help="final lr")
    parser.add_argument("--lr_F", type=float, default=0.001, help="lr of F")

    # model related
    parser.add_argument("--F_layers_num", type=int, default=3, help="the layers num of F")
    parser.add_argument("--F_hidden_dim", type=int, default=60, help="the hidden dim of F")

    # loss related
    parser.add_argument("--alpha", type=float, default=0.7, help="loss = l_data + alpha * l_PDE + beta * l_physics")
    parser.add_argument("--beta", type=float, default=0.2, help="loss = l_data + alpha * l_PDE + beta * l_physics")

    args = parser.parse_args()

    # Model.py expects args.seed sometimes; we provide it (set per experiment in main())
    args.seed = int(args.base_seed)

    return args


if __name__ == "__main__":
    main()
