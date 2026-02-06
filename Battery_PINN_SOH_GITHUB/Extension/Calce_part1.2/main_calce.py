
from __future__ import annotations

import os
import sys
import argparse
from typing import List, Optional, Dict

# Ensure we can import sibling files: dataloader.py, Model.py, util.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from dataloader import DF
from Model import PINN

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


class CalceData(DF):


    def __init__(self, root: str, args: argparse.Namespace):
        super().__init__(args)
        self.root = root
        self.file_list = sorted([f for f in os.listdir(root) if f.lower().endswith(".csv")])
        self.num = len(self.file_list)
        self.nominal_capacity = 1.0 if self.normalization else None

    def read_all(self, specific_path_list: Optional[List[str]] = None):
        if specific_path_list is None:
            paths = [os.path.join(self.root, f) for f in self.file_list]
        else:
            paths = specific_path_list
        return self.load_all_battery(path_list=paths, nominal_capacity=self.nominal_capacity)


def _list_all_csvs(root: str) -> List[str]:
    files = sorted([f for f in os.listdir(root) if f.lower().endswith(".csv")])
    return [os.path.join(root, f) for f in files]


def _split_by_temp_tag(all_paths: List[str], test_temp_tag: str) -> (List[str], List[str]):
    """
    Split by filename tag, e.g.:
      test_temp_tag="T45C"  => test files contain "T45C_"
    """
    tag = test_temp_tag.strip()
    if not tag:
        raise ValueError("Empty test_temp_tag")

    # Use "T45C_" style matching to avoid accidental matches
    pattern = f"{tag}_"
    test = [p for p in all_paths if pattern in os.path.basename(p)]
    train = [p for p in all_paths if pattern not in os.path.basename(p)]

    if len(test) == 0:
        raise ValueError(
            f"No test files matched tag '{pattern}'. "
            f"Example filenames: {[os.path.basename(x) for x in all_paths[:5]]}"
        )
    if len(train) == 0:
        raise ValueError(f"All files matched tag '{pattern}' — train set would be empty.")

    return train, test


def _split_by_protocol_tag(all_paths: List[str], test_protocol_tag: str) -> (List[str], List[str]):
    """
    Split by filename tag for protocol/condition, e.g.:
      test_protocol_tag="cond16" => test files contain "cond16_"
    """
    tag = test_protocol_tag.strip()
    if not tag:
        raise ValueError("Empty test_protocol_tag")

    pattern = f"{tag}_"
    test = [p for p in all_paths if pattern in os.path.basename(p)]
    train = [p for p in all_paths if pattern not in os.path.basename(p)]

    if len(test) == 0:
        raise ValueError(
            f"No test files matched tag '{pattern}'. "
            f"Example filenames: {[os.path.basename(x) for x in all_paths[:5]]}"
        )
    if len(train) == 0:
        raise ValueError(f"All files matched tag '{pattern}' — train set would be empty.")

    return train, test


def _split_by_ratio(all_paths: List[str], test_ratio: float, seed: int) -> (List[str], List[str]):
    """
    File-level random split (fallback).
    """
    if not (0.0 < test_ratio < 1.0):
        return all_paths, []

    import random
    rng = random.Random(seed)
    paths = list(all_paths)
    rng.shuffle(paths)

    n_test = max(1, int(round(len(paths) * test_ratio)))
    test = paths[:n_test]
    train = paths[n_test:]

    return train, test


def load_data(args: argparse.Namespace) -> Dict[str, object]:
    """
    Returns dict with train/valid/test dataloaders compatible with PINN.Train().
    """
    data = CalceData(root=args.root, args=args)
    all_paths = _list_all_csvs(args.root)

    if args.small_sample is not None:
        all_paths = all_paths[: int(args.small_sample)]

    # ------------------------------------------
    # Decide split strategy (priority)
    # 1) held-out temperature tag
    # 2) held-out protocol tag
    # 3) ratio split
    # ------------------------------------------
    if args.test_temp_tag is not None and args.test_temp_tag != "":
        train_list, test_list = _split_by_temp_tag(all_paths, args.test_temp_tag)
        split_mode = f"held-out temp: {args.test_temp_tag}"
    elif args.test_protocol_tag is not None and args.test_protocol_tag != "":
        train_list, test_list = _split_by_protocol_tag(all_paths, args.test_protocol_tag)
        split_mode = f"held-out protocol: {args.test_protocol_tag}"
    else:
        train_list, test_list = _split_by_ratio(all_paths, args.test_ratio, args.seed)
        split_mode = f"ratio: {args.test_ratio}"

    # ------------------------------------------
    # Build loaders using DF.load_all_battery
    # ------------------------------------------
    train_loader_bundle = data.read_all(specific_path_list=train_list)

    if len(test_list) > 0:
        test_loader_bundle = data.read_all(specific_path_list=test_list)

        # We use the random 80/20 split inside DF for train/valid:
        dataloader = {
            "train": train_loader_bundle["train_2"],
            "valid": train_loader_bundle["valid_2"],
            "test":  test_loader_bundle["test_3"],  # test on all test files (no split)
        }
    else:
        # If you choose no test, fallback to DF's internal split:
        dataloader = {
            "train": train_loader_bundle["train"],
            "valid": train_loader_bundle["valid"],
            "test":  train_loader_bundle["test"],
        }

    # Print useful debug info
    print("\n=== CALCE DATA SPLIT ===")
    print("Root:", args.root)
    print("Total CSV files:", len(all_paths))
    print("Split mode:", split_mode)
    print("Train files:", len(train_list))
    print("Test files:", len(test_list))
    if len(test_list) > 0:
        print("Example test file:", os.path.basename(test_list[0]))
    print("========================\n")

    return dataloader


def main():
    args = get_args()

    for e in range(args.n_experiments):
        save_folder = os.path.join(args.results_root, "0-0", f"Experiment{e + 1}")
        os.makedirs(save_folder, exist_ok=True)

        setattr(args, "save_folder", save_folder)
        setattr(args, "log_dir", args.log_name)

        dataloader = load_data(args)
        pinn = PINN(args)
        pinn.Train(
            trainloader=dataloader["train"],
            validloader=dataloader["valid"],
            testloader=dataloader["test"],
        )


def get_args():
    parser = argparse.ArgumentParser("Hyper Parameters for CALCE (processed CSVs)")

    # data paths
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Folder containing processed CALCE CSVs (e.g., .../CALCE_processed_fixed3)",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results of reviewer/Calce results",
        help="Root folder to store experiment outputs",
    )
    parser.add_argument("--log_name", type=str, default="logging.txt", help="Log filename inside each experiment")

    # split controls
    parser.add_argument(
        "--test_temp_tag",
        type=str,
        default="",
        help="Held-out temperature test tag, e.g. 'T45C' to test on files containing 'T45C_'",
    )
    parser.add_argument(
        "--test_protocol_tag",
        type=str,
        default="",
        help="Held-out protocol/condition tag, e.g. 'cond16' to test on files containing 'cond16_'",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fallback: file-level random split ratio if no held-out tag is set (0..1).",
    )
    parser.add_argument("--seed", type=int, default=420, help="Random seed for ratio split")

    # repeat experiments
    parser.add_argument("--n_experiments", type=int, default=1, help="Number of repeated runs")
    parser.add_argument("--small_sample", type=int, default=None, help="Use first N CSV files (debug)")

    # match paper args (must match names used by Model.py)
    parser.add_argument("--data", type=str, default="CALCE", help="dataset name label")
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
