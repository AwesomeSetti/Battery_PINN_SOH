"""main_michigan.py

Run the paper's PINN training loop on **processed Michigan per-cycle feature CSVs**.

This script is intentionally structured to mirror `main_XJTU.py`:
  - build train/valid/test dataloaders from a folder of per-battery CSVs
  - repeat multiple experiments with different random splits
  - save logs + predictions in a results folder per experiment

Expected Michigan processed CSV format (same as paper/XJTU style):
  feature columns ... , capacity
Optionally a "cycle" column may exist; it will be treated as a feature.

Notes:
  - Your `prepare_michigan_cycle_csv.py` already normalizes "capacity" to SOH (baseline=first valid cycle).
    Therefore, we set `nominal_capacity=1.0` for Michigan so the dataloader's SOH normalization is a no-op.
  - We avoid hard-coding a specific train/test split. Instead you can specify test cells explicitly
    (`--test_cells 4 8`) OR use a deterministic ratio split (`--test_ratio 0.2`).
"""
from __future__ import annotations

import os
import sys
import argparse
import re
from typing import List, Optional, Tuple

# Ensure we can import sibling files: dataloader.py, Model.py, util.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Battery_PINN_SOH/Paper
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from dataloader import DF
from Model import PINN
from util import write_to_txt

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")



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


def _split_by_cells(
    all_paths: List[str],
    test_cells: Optional[List[int]],
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Split paths into train/test by cell id.

    Priority:
      1) If test_cells provided -> those ids go to test.
      2) Else if test_ratio in (0,1) -> take last fraction (deterministic by sorted id)
    """

    # Map path -> id (or -1)
    pairs = []
    for p in all_paths:
        cid = _extract_cell_id(p)
        pairs.append((p, cid))

    # If any id is missing, keep them in train unless explicitly selected.
    if test_cells and len(test_cells) > 0:
        test_set = set(int(x) for x in test_cells)
        test = [p for (p, cid) in pairs if cid in test_set]
        train = [p for (p, cid) in pairs if cid not in test_set]
        return train, test

    # Ratio-based split (by unique ids)
    ids = sorted({cid for (_, cid) in pairs if cid is not None})
    if 0.0 < test_ratio < 1.0 and len(ids) >= 2:
        # deterministic selection by sorted ids; seed kept for future extensibility
        n_test = max(1, int(round(len(ids) * test_ratio)))
        test_ids = set(ids[-n_test:])
        test = [p for (p, cid) in pairs if cid in test_ids]
        train = [p for (p, cid) in pairs if (cid not in test_ids)]
        # If some files have no id, keep in train.
        train += [p for (p, cid) in pairs if cid is None]
        # Remove duplicates if any
        train = sorted(set(train))
        test = sorted(set(test))
        return train, test

    # Fallback: no split requested -> all in train, none in test
    return all_paths, []


def load_data(args: argparse.Namespace, small_sample: Optional[int] = None):
    root = args.root
    data = MichiganData(root=root, args=args)

    all_paths = [os.path.join(root, f) for f in data.file_list]
    if small_sample is not None:
        all_paths = all_paths[: int(small_sample)]

    train_list, test_list = _split_by_cells(
        all_paths=all_paths,
        test_cells=args.test_cells,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_loader = data.read_all(specific_path_list=train_list)
    if len(test_list) > 0:
        test_loader = data.read_all(specific_path_list=test_list)
        dataloader = {
            "train": train_loader["train_2"],
            "valid": train_loader["valid_2"],
            "test": test_loader["test_3"],
        }
    else:
        # No explicit test set: reuse the internal split (Condition 1) from DF.load_all_battery
        dataloader = {
            "train": train_loader["train"],
            "valid": train_loader["valid"],
            "test": train_loader["test"],
        }
    return dataloader


def main():
    args = get_args()

    for e in range(args.n_experiments):
        # Make each experiment deterministic (train/valid split uses a fixed random_state in dataloader)
        # We still record the experiment index in the output folder name.
        save_folder = os.path.join(
            args.results_root,
            "0-0",
            f"Experiment{e + 1}",
        )
        os.makedirs(save_folder, exist_ok=True)

        setattr(args, "save_folder", save_folder)
        setattr(args, "log_dir", args.log_name)

        dataloader = load_data(args, small_sample=args.small_sample)
        x1, y1, x2, y2 = next(iter(dataloader["train"]))
        

        pinn = PINN(args)
        pinn.Train(
            trainloader=dataloader["train"],
            validloader=dataloader["valid"],
            testloader=dataloader["test"],
        )


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

    # split controls
    parser.add_argument(
        "--test_cells",
        type=int,
        nargs="*",
        default=None,
        help="Cell IDs to reserve for test (e.g., --test_cells 4 8). If omitted, use --test_ratio.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="If --test_cells not provided, reserve the last fraction of cell IDs for test (0..1).",
    )
    parser.add_argument("--seed", type=int, default=420, help="Seed placeholder (split is deterministic by id)")

    # repeat experiments
    parser.add_argument("--n_experiments", type=int, default=10, help="Number of repeated runs")
    parser.add_argument(
        "--small_sample",
        type=int,
        default=None,
        help="Optional: only use the first N CSV files in the folder (debug)",
    )

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

    return parser.parse_args()


if __name__ == "__main__":
    main()
