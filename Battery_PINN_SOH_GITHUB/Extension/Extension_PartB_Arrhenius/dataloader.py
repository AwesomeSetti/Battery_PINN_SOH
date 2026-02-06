import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from util import write_to_txt


class DF():
    def __init__(self, args):
        self.normalization = True
        self.normalization_method = args.normalization_method  # min-max, z-score
        self.args = args

    def _3_sigma(self, Ser1):
        rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.std() < Ser1)
        index = np.arange(Ser1.shape[0])[rule]
        return index

    def delete_3_sigma(self, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.reset_index(drop=True)
        out_index = []
        for col in df.columns:
            index = self._3_sigma(df[col])
            out_index.extend(index)
        out_index = list(set(out_index))
        df = df.drop(out_index, axis=0)
        df = df.reset_index(drop=True)
        return df

    def _reorder_calce_columns_if_present(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce a stable column order for CALCE processed files:

        [... feature columns ..., temp_C, temp_K, condition_id, cycle, capacity]

        This guarantees:
          - time t = last feature column (cycle)
          - y = last column (capacity)
          - temp_K is always at index -3 among features (good for Arrhenius)
        """
        must_have = ["capacity"]
        for c in must_have:
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}' in CSV.")

        # If these CALCE-specific columns exist, reorder them.
        calce_cols = ["temp_C", "temp_K", "condition_id", "cycle"]
        has_all = all(c in df.columns for c in calce_cols)

        # Also support "cycle index" if your file uses that name
        if not has_all:
            # If "cycle" not present, but "cycle index" exists, rename it to cycle
            if "cycle" not in df.columns and "cycle index" in df.columns:
                df = df.rename(columns={"cycle index": "cycle"})
                has_all = all(c in df.columns for c in calce_cols)

        if has_all:
            base_cols = [c for c in df.columns if c not in ["temp_C", "temp_K", "condition_id", "cycle", "capacity"]]
            ordered_cols = base_cols + ["temp_C", "temp_K", "condition_id", "cycle", "capacity"]
            df = df[ordered_cols]
            return df

        # Otherwise keep original order (non-CALCE datasets)
        return df

    def read_one_csv(self, file_name, nominal_capacity=None):
        df = pd.read_csv(file_name)

        # Add a time index ONLY if dataset doesn't already have one
        has_cycle = any(c.lower() in ["cycle", "cycle_index", "cycle index"] for c in df.columns)
        if not has_cycle:
            df.insert(df.shape[1] - 1, "cycle index", np.arange(df.shape[0]))

        # Remove outliers / NaN / Inf
        df = self.delete_3_sigma(df)

        # Reorder columns (CALCE)
        df = self._reorder_calce_columns_if_present(df)

        if nominal_capacity is not None:
            df["capacity"] = df["capacity"] / nominal_capacity

            f_df = df.iloc[:, :-1]
            if self.normalization_method == "min-max":
                den = (f_df.max() - f_df.min())
                den = den.replace(0, 1.0)
                f_df = 2 * (f_df - f_df.min()) / den - 1
                f_df = f_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            elif self.normalization_method == "z-score":
                std = f_df.std().replace(0, 1.0)
                f_df = (f_df - f_df.mean()) / std

            df.iloc[:, :-1] = f_df

        return df

    def load_one_battery(self, path, nominal_capacity=None):
        df = self.read_one_csv(path, nominal_capacity)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        x1 = x[:-1]
        x2 = x[1:]
        y1 = y[:-1]
        y2 = y[1:]
        return (x1, y1), (x2, y2)

    def load_all_battery(self, path_list, nominal_capacity):
        X1, X2, Y1, Y2 = [], [], [], []

        if self.args.log_dir is not None and self.args.save_folder is not None:
            save_name = os.path.join(self.args.save_folder, self.args.log_dir)
            write_to_txt(save_name, "data path:")
            write_to_txt(save_name, str(path_list))

        for path in path_list:
            (x1, y1), (x2, y2) = self.load_one_battery(path, nominal_capacity)
            X1.append(x1)
            X2.append(x2)
            Y1.append(y1)
            Y2.append(y2)

        X1 = np.concatenate(X1, axis=0)
        X2 = np.concatenate(X2, axis=0)
        Y1 = np.concatenate(Y1, axis=0)
        Y2 = np.concatenate(Y2, axis=0)

        tensor_X1 = torch.from_numpy(X1).float()
        tensor_X2 = torch.from_numpy(X2).float()
        tensor_Y1 = torch.from_numpy(Y1).float().view(-1, 1)
        tensor_Y2 = torch.from_numpy(Y2).float().view(-1, 1)

        # Condition 1: 80/20 split in time within the concatenated dataset, then 80/20 train/valid
        split = int(tensor_X1.shape[0] * 0.8)
        train_X1, test_X1 = tensor_X1[:split], tensor_X1[split:]
        train_X2, test_X2 = tensor_X2[:split], tensor_X2[split:]
        train_Y1, test_Y1 = tensor_Y1[:split], tensor_Y1[split:]
        train_Y2, test_Y2 = tensor_Y2[:split], tensor_Y2[split:]

        train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
            train_test_split(train_X1, train_X2, train_Y1, train_Y2, test_size=0.2, random_state=420)

        train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                  batch_size=self.args.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                  batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
                                 batch_size=self.args.batch_size, shuffle=False)

        # Condition 2: random 80/20 split for train/valid; no explicit test here
        train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
            train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, test_size=0.2, random_state=420)

        train_loader_2 = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                    batch_size=self.args.batch_size, shuffle=True)
        valid_loader_2 = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                    batch_size=self.args.batch_size, shuffle=True)

        # Condition 3: wrap full set as "test"
        test_loader_3 = DataLoader(TensorDataset(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2),
                                   batch_size=self.args.batch_size, shuffle=False)

        loader = {
            "train": train_loader, "valid": valid_loader, "test": test_loader,
            "train_2": train_loader_2, "valid_2": valid_loader_2,
            "test_3": test_loader_3
        }
        return loader
