from __future__ import annotations
import os
import sys
import argparse
from typing import List, Optional

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
        # your processed CALCE already uses normalized capacity groups; keep nominal=1.0
        self.nominal_capacity = 1.0 if self.normalization else None

    def read_all(self, specific_path_list: Optional[List[str]] = None):
        if specific_path_list is None:
            paths = [os.path.join(self.root, f) for f in self.file_list]
        else:
            paths = specific_path_list
        return self.load_all_battery(path_list=paths, nominal_capacity=self.nominal_capacity)


def list_csv_paths(root: str, small_sample: Optional[int] = None) -> List[str]:
    files = sorted([f for f in os.listdir(root) if f.lower().endswith(".csv")])
    if small_sample is not None:
        files = files[: int(small_sample)]
    return [os.path.join(root, f) for f in files]


def split_paths(
    all_paths: List[str],
    test_temp_tag: Optional[str],
    test_protocol_tag: Optional[str],
    test_ratio: float,
    seed: int,
):
    import random
    rng = random.Random(seed)

    def match(p: str) -> bool:
        name = os.path.basename(p)
        ok = True
        if test_temp_tag:
            ok = ok and (test_temp_tag in name)
        if test_protocol_tag:
            ok = ok and (test_protocol_tag in name)
        return ok

    if test_temp_tag or test_protocol_tag:
        test_paths = [p for p in all_paths if match(p)]
        train_paths = [p for p in all_paths if p not in test_paths]
        return train_paths, test_paths, "held-out tag"

    # fallback random split
    paths = list(all_paths)
    rng.shuffle(paths)
    n_test = max(1, int(round(len(paths) * test_ratio)))
    test_paths = paths[:n_test]
    train_paths = paths[n_test:]
    return train_paths, test_paths, "random ratio"


def load_data(args: argparse.Namespace):
    data = CalceData(root=args.root, args=args)
    all_paths = list_csv_paths(args.root, small_sample=args.small_sample)

    train_list, test_list, mode = split_paths(
        all_paths=all_paths,
        test_temp_tag=args.test_temp_tag,
        test_protocol_tag=args.test_protocol_tag,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list) if len(test_list) > 0 else None

    print("=== CALCE SPLIT ===")
    print("Mode:", mode)
    print("Root:", args.root)
    print("Total CSV files:", len(all_paths))
    print("Train files:", len(train_list))
    print("Test files:", len(test_list))
    if len(test_list) > 0:
        print("Example test file:", os.path.basename(test_list[0]))
    print("===================")

    dataloader = {
        "train": train_loader["train_2"],
        "valid": train_loader["valid_2"],
        "test": test_loader["test_3"] if test_loader is not None else train_loader["test"],
    }
    return dataloader


def main():
    args = get_args()

    for e in range(args.n_experiments):
        save_folder = os.path.join(args.results_root, "0-0", f"Experiment{e + 1}")
        os.makedirs(save_folder, exist_ok=True)
        args.save_folder = save_folder
        args.log_dir = args.log_name

        dl = load_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dl["train"], validloader=dl["valid"], testloader=dl["test"])


def get_args():
    parser = argparse.ArgumentParser("Hyper Parameters for CALCE (processed CSVs)")
    parser.add_argument("--root", type=str, required=True, help="Folder with processed per-condition CSVs")
    parser.add_argument("--results_root", type=str, default="results of reviewer/Calce results")
    parser.add_argument("--log_name", type=str, default="logging.txt")

    # split
    parser.add_argument("--test_temp_tag", type=str, default=None, help="e.g. T45C (held-out temperature)")
    parser.add_argument("--test_protocol_tag", type=str, default=None, help="e.g. cond16 (held-out protocol)")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--n_experiments", type=int, default=1)
    parser.add_argument("--small_sample", type=int, default=None)

    # training
    parser.add_argument("--data", type=str, default="CALCE")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--normalization_method", type=str, default="min-max")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=30)
    parser.add_argument("--warmup_lr", type=float, default=0.002)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--final_lr", type=float, default=0.0002)
    parser.add_argument("--lr_F", type=float, default=0.001)
    parser.add_argument("--F_layers_num", type=int, default=3)
    parser.add_argument("--F_hidden_dim", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.2)

    # model dims
    parser.add_argument("--input_dim", type=int, default=20)

    # --- Arrhenius / StressNet extension flags ---
    parser.add_argument("--use_arrhenius", action="store_true", help="Enable Arrhenius k(T) in PDE residual")
    parser.add_argument("--Tref", type=float, default=298.15)
    parser.add_argument("--tempK_col", type=int, default=-3, help="column index for temp_K inside xt")
    parser.add_argument("--use_stressnet", action="store_true", help="Enable learned s(cond) in PDE residual")
    parser.add_argument("--cond_dim", type=int, default=2, help="dimension of cond vector for StressNet")
    parser.add_argument("--cond_cols", type=str, default=None, help='e.g. "-2,-3" meaning [condition_id,temp_K]')

    return parser.parse_args()


if __name__ == "__main__":
    main()
