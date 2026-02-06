"""
main_michigan_cell_split.py

Strict Michigan replication with CELL-LEVEL splits:
- split CSV files by cell id into train / valid / test
- then concatenate cycles WITHIN each split
- train PINN on train, early-stop on valid, report on test

This avoids leakage that happens if you concatenate all cells then split rows.
"""

from __future__ import annotations

import os
import re
import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# local imports (same folder)
from dataloader import DF
from Model import PINN


def extract_cell_id(path: str) -> Optional[int]:
    name = os.path.basename(path)
    m = re.search(r"(?:Cell[_\-\s]*)(\d+)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)(?=\.csv$)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def split_cells(
    cell_ids: List[int],
    test_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Random split by cell IDs.
    """
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(cell_ids))
    rng.shuffle(ids)

    n = len(ids)
    n_test = max(1, int(round(n * test_ratio)))
    n_valid = max(1, int(round(n * valid_ratio)))

    test_ids = ids[:n_test].tolist()
    valid_ids = ids[n_test : n_test + n_valid].tolist()
    train_ids = ids[n_test + n_valid :].tolist()

    # Safety: ensure no empty train
    if len(train_ids) == 0:
        train_ids = valid_ids[:-1]
        valid_ids = valid_ids[-1:]

    return train_ids, valid_ids, test_ids


def paths_for_ids(all_paths: List[str], wanted_ids: List[int]) -> List[str]:
    wanted = set(wanted_ids)
    out = []
    for p in all_paths:
        cid = extract_cell_id(p)
        if cid in wanted:
            out.append(p)
    return sorted(out)


def build_loader_from_paths(df_helper: DF, paths: List[str], nominal_capacity: float, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Build a DataLoader by concatenating (x1,y1),(x2,y2) across selected cell CSVs.
    Uses DF.load_one_battery() which already:
      - inserts cycle index if missing
      - applies 3-sigma filtering
      - normalizes features (min-max or z-score)
      - returns paired samples (x_t, x_{t+1}, y_t, y_{t+1})
    """
    X1, X2, Y1, Y2 = [], [], [], []

    for p in paths:
        (x1, y1), (x2, y2) = df_helper.load_one_battery(p, nominal_capacity=nominal_capacity)
        X1.append(x1)
        X2.append(x2)
        Y1.append(y1)
        Y2.append(y2)

    if len(X1) == 0:
        raise RuntimeError("No files selected for this split. Check your root path and filename patterns.")

    X1 = np.concatenate(X1, axis=0)
    X2 = np.concatenate(X2, axis=0)
    Y1 = np.concatenate(Y1, axis=0).reshape(-1, 1)
    Y2 = np.concatenate(Y2, axis=0).reshape(-1, 1)

    tX1 = torch.from_numpy(X1).float()
    tX2 = torch.from_numpy(X2).float()
    tY1 = torch.from_numpy(Y1).float()
    tY2 = torch.from_numpy(Y2).float()

    return DataLoader(
        TensorDataset(tX1, tX2, tY1, tY2),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def main():
    args = get_args()

    # list all Michigan CSVs
    all_paths = sorted([
        os.path.join(args.root, f)
        for f in os.listdir(args.root)
        if f.lower().endswith(".csv")
    ])

    # extract ids
    id_map = []
    for p in all_paths:
        cid = extract_cell_id(p)
        if cid is not None:
            id_map.append(cid)

    unique_ids = sorted(set(id_map))
    if len(unique_ids) < 3:
        raise RuntimeError("Not enough Michigan cell CSVs found. Check root folder and filenames like Michigan_Cell_01.csv")

    # Michigan processed CSV already stores capacity as SOH => keep nominal_capacity=1.0
    # (so DF won't change the label scale)
    nominal_capacity = 1.0

    # helper to reuse DF preprocessing + pairing logic
    df_helper = DF(args)

    for e in range(args.n_experiments):
        exp_seed = args.base_seed + e
        save_folder = os.path.join(args.results_root, "0-0", f"Experiment{e+1}")
        os.makedirs(save_folder, exist_ok=True)

        # wire outputs into Model.py logger + saver
        args.save_folder = save_folder
        args.log_dir = args.log_name
        args.seed = exp_seed  # recorded in logs

        # choose cell split
        if args.test_cells is not None and len(args.test_cells) > 0:
            test_ids = sorted(set(args.test_cells))
            remaining = [i for i in unique_ids if i not in test_ids]

            if args.valid_cells is not None and len(args.valid_cells) > 0:
                valid_ids = sorted(set(args.valid_cells))
                train_ids = [i for i in remaining if i not in valid_ids]
            else:
                train_ids, valid_ids, _ = split_cells(remaining, test_ratio=0.0, valid_ratio=args.valid_ratio, seed=exp_seed)
        else:
            train_ids, valid_ids, test_ids = split_cells(unique_ids, args.test_ratio, args.valid_ratio, exp_seed)

        train_paths = paths_for_ids(all_paths, train_ids)
        valid_paths = paths_for_ids(all_paths, valid_ids)
        test_paths  = paths_for_ids(all_paths, test_ids)

        # build loaders
        train_loader = build_loader_from_paths(df_helper, train_paths, nominal_capacity, args.batch_size, shuffle=True)
        valid_loader = build_loader_from_paths(df_helper, valid_paths, nominal_capacity, args.batch_size, shuffle=False)
        test_loader  = build_loader_from_paths(df_helper, test_paths,  nominal_capacity, args.batch_size, shuffle=False)

        pinn = PINN(args)
        pinn.Train(trainloader=train_loader, validloader=valid_loader, testloader=test_loader)


def get_args():
    p = argparse.ArgumentParser("Michigan CELL-level split replication")

    p.add_argument("--root", type=str, required=True, help="Folder of Michigan processed per-cell CSVs")
    p.add_argument("--results_root", type=str, required=True, help="Where to save Experiment folders")
    p.add_argument("--log_name", type=str, default="logging.txt")

    # split controls
    p.add_argument("--test_cells", type=int, nargs="*", default=None, help="Optional explicit test cell ids")
    p.add_argument("--valid_cells", type=int, nargs="*", default=None, help="Optional explicit valid cell ids")
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--valid_ratio", type=float, default=0.2)
    p.add_argument("--base_seed", type=int, default=420)

    # repeats
    p.add_argument("--n_experiments", type=int, default=10)

    # match Model.py / DF args
    p.add_argument("--data", type=str, default="Michigan")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--normalization_method", type=str, default="min-max", choices=["min-max", "z-score"])

    # training hyperparams (same as your current setup)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--early_stop", type=int, default=20)
    p.add_argument("--warmup_epochs", type=int, default=30)
    p.add_argument("--warmup_lr", type=float, default=0.002)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--final_lr", type=float, default=0.0002)
    p.add_argument("--lr_F", type=float, default=0.001)

    # model
    p.add_argument("--F_layers_num", type=int, default=3)
    p.add_argument("--F_hidden_dim", type=int, default=60)

    # loss weights
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=0.2)

    return p.parse_args()


if __name__ == "__main__":
    main()
