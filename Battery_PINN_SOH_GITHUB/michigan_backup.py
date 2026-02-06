import os
import re
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Paper.dataloader import DF


from Model import PINN
  # same as main_XJTU.py
from util import write_to_txt


def extract_cell_id(path: str) -> int:
    """
    Expect filenames like:
      Michigan_Cell_01.csv
      Michigan_Cell_11.csv
    """
    base = os.path.basename(path)
    m = re.search(r"Michigan_Cell_(\d+)\.csv", base)
    if not m:
        raise ValueError(f"Cannot parse cell id from filename: {base}")
    return int(m.group(1))


def load_data_michigan(args):
    """
    Michigan equivalent of load_data() in main_XJTU.py.
    - Lists processed per-cell CSVs
    - Splits into train/test by cell (held-out cells)
    - Lets DF do its internal train/valid split (Condition 2)
    """
    root = args.data_root  # <-- set this in main()

    all_files = sorted(
        os.path.join(root, f)
        for f in os.listdir(root)
        if f.endswith(".csv") and f.startswith("Michigan_Cell_")
    )

    if not all_files:
        raise RuntimeError(f"No Michigan_Cell_*.csv found in: {root}")

    # Decide train/test split by cell IDs (paper-style: hold out whole cells)
    # Default: last 20% cells as test.
    cell_ids = [extract_cell_id(p) for p in all_files]
    pairs = sorted(zip(cell_ids, all_files), key=lambda x: x[0])

    # Example policy: deterministic split by cell number
    n = len(pairs)
    n_test = max(2, int(round(0.2 * n)))  # at least 2 test cells
    test_pairs = pairs[-n_test:]
    train_pairs = pairs[:-n_test]

    train_list = [p for _, p in train_pairs]
    test_list = [p for _, p in test_pairs]

    df = DF(args)
    nominal_capacity = 2.36
    train_loader = df.load_all_battery(train_list, nominal_capacity)
    test_loader  = df.load_all_battery(test_list, nominal_capacity)

    #train_loader = df.load_all_battery(train_list)  # provides train_2, valid_2
    #test_loader = df.load_all_battery(test_list)    # provides test_3

    return train_loader, test_loader, train_list, test_list


def main():

import argparse

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters (XJTU/Michigan)')
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU, Michigan')

    # ✅ add this for Michigan processed folder
    parser.add_argument('--data_root', type=str, default=None,
                        help='If set, use this folder of processed CSVs (e.g., Michigan_processed)')

    # ✅ add seed (your Michigan main was using it)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--train_batch', type=int, default=0, choices=[-1,0,1,2,3,4,5])
    parser.add_argument('--test_batch', type=int, default=1, choices=[-1,0,1,2,3,4,5])
    parser.add_argument('--batch', type=str, default='2C',
                        choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--normalization_method', type=str, default='min-max')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--warmup_epochs', type=int, default=30)
    parser.add_argument('--warmup_lr', type=float, default=0.002)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--final_lr', type=float, default=0.0002)
    parser.add_argument('--lr_F', type=float, default=0.001)

    # model related
    parser.add_argument('--F_layers_num', type=int, default=3)
    parser.add_argument('--F_hidden_dim', type=int, default=60)

    # loss related
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=0.2)

    parser.add_argument('--log_dir', type=str, default='text log.txt')
    parser.add_argument('--save_folder', type=str, default='results of reviewer/XJTU results')

    args = parser.parse_args()
    return args


    train_loader, test_loader, train_list, test_list = load_data_michigan(args)

    print("Michigan processed root:", args.data_root)
    print(f"Train cells: {len(train_list)} files")
    print(f"Test  cells: {len(test_list)} files")
    print("Example train file:", os.path.basename(train_list[0]))
    print("Example test  file:", os.path.basename(test_list[0]))

    # Train
    model = PINN(args)

    # Follow the same keys main_XJTU.py uses
    model.train(
        train_loader=train_loader["train_2"],
        valid_loader=train_loader["valid_2"],
        test_loader=test_loader["test_3"],


if __name__ == "__main__":
    main()
