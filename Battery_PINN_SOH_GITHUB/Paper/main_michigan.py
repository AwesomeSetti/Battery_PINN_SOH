import os
import re
import random
import argparse

from .dataloader import DF
from .Model import PINN


def extract_cell_id(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"Michigan_Cell_(\d+)\.csv", base)
    if not m:
        raise ValueError(f"Cannot parse cell id from filename: {base}")
    return int(m.group(1))


def load_data_michigan(args):
    root = args.data_root

    all_files = sorted(
        os.path.join(root, f)
        for f in os.listdir(root)
        if f.endswith(".csv") and f.startswith("Michigan_Cell_")
    )
    if not all_files:
        raise RuntimeError(f"No Michigan_Cell_*.csv found in: {root}")

    # hold out last 20% cells as test
    cell_ids = [extract_cell_id(p) for p in all_files]
    pairs = sorted(zip(cell_ids, all_files), key=lambda x: x[0])

    n = len(pairs)
    n_test = max(2, int(round(0.2 * n)))
    test_pairs = pairs[-n_test:]
    train_pairs = pairs[:-n_test]

    train_list = [p for _, p in train_pairs]
    test_list = [p for _, p in test_pairs]

    df = DF(args)

    # If your processed CSV capacity column is already SOH (~1.0 at start), keep 1.0.
    # If capacity is in Ah, use 2.36.
    nominal_capacity = 1.0
    train_loader = df.load_all_battery(train_list, nominal_capacity)
    test_loader = df.load_all_battery(test_list, nominal_capacity)

    return train_loader, test_loader, train_list, test_list


def get_args():
    parser = argparse.ArgumentParser("Hyper Parameters for Michigan dataset (paper-compatible)")
    parser.add_argument("--data", type=str, default="Michigan")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder containing Michigan_Cell_*.csv (processed)")

    # Keep paper args so dataloader/model don't crash
    parser.add_argument("--train_batch", type=int, default=0, choices=[-1, 0, 1, 2, 3, 4, 5])
    parser.add_argument("--test_batch", type=int, default=1, choices=[-1, 0, 1, 2, 3, 4, 5])
    parser.add_argument("--batch", type=str, default="2C",
                        choices=["2C", "3C", "R2.5", "R3", "RW", "satellite"])

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--normalization_method", type=str, default="min-max")

    # scheduler
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=30)
    parser.add_argument("--warmup_lr", type=float, default=0.002)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--final_lr", type=float, default=0.0002)
    parser.add_argument("--lr_F", type=float, default=0.001)

    # model
    parser.add_argument("--F_layers_num", type=int, default=3)
    parser.add_argument("--F_hidden_dim", type=int, default=60)

    # loss
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.2)

    # REQUIRED by Paper/dataloader.py
    parser.add_argument("--log_dir", type=str, default="text log.txt")
    parser.add_argument("--save_folder", type=str, default="results of reviewer/Michigan results")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = get_args()
    # make sure save folder exists (paper code assumes it)
    if args.save_folder is not None:
        os.makedirs(args.save_folder, exist_ok=True)

    random.seed(args.seed)

    train_loader, test_loader, train_list, test_list = load_data_michigan(args)

    print("Michigan processed root:", args.data_root)
    print(f"Train cells: {len(train_list)} files")
    print(f"Test  cells: {len(test_list)} files")
    print("Example train file:", os.path.basename(train_list[0]))
    print("Example test  file:", os.path.basename(test_list[0]))

    model = PINN(args)
    model.train(
        train_loader=train_loader["train_2"],
        valid_loader=train_loader["valid_2"],
        test_loader=test_loader["test_3"],
    )
    


if __name__ == "__main__":
    main()

