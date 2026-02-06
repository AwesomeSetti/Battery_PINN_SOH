import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import random
from sklearn.model_selection import train_test_split
from util import write_to_txt

class DF():
    def __init__(self,args):
        self.normalization = True
        self.normalization_method = args.normalization_method # min-max, z-score
        self.args = args

    def _3_sigma(self, Ser1):
        '''
        :param Ser1:
        :return: index
        '''
        rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.std() < Ser1)
        index = np.arange(Ser1.shape[0])[rule]
        return index

    def delete_3_sigma(self,df):
        '''
        :param df: DataFrame
        :return: DataFrame
        '''
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

    def read_one_csv(self,file_name,nominal_capacity=None):
        '''
        read a csv file and return a DataFrame
        :param file_name: str
        :return: DataFrame
        '''
        df = pd.read_csv(file_name)

        # ------------------------------------------------------------
        # (A) Ensure a cycle/time index exists
        # ------------------------------------------------------------
        # Add a time index ONLY if dataset doesn't already have one
        has_cycle = any(c.lower() in ["cycle", "cycle_index", "cycle index"] for c in df.columns)
        if not has_cycle:
            # Insert before the last column (assumed label) if possible
            insert_at = max(df.shape[1] - 1, 0)
            df.insert(insert_at, "cycle index", np.arange(df.shape[0]))

        # ------------------------------------------------------------
        # (B) CALCE-specific column ordering fix (CRITICAL)
        # Your Model.py assumes:
        #   t = xt[:, -1]   (last feature col)
        #   x = xt[:, :-1]
        #
        # In your processed CALCE csv, the columns end like:
        #   ... cycle, temp_C, temp_K, condition_id, capacity
        # which causes Model.py to treat condition_id as time.
        #
        # We reorder to:
        #   ... temp_C, temp_K, condition_id, cycle, capacity
        # so "cycle" becomes the last feature column.
        # ------------------------------------------------------------
        required_calce = {"cycle", "temp_C", "temp_K", "condition_id", "capacity"}
        if required_calce.issubset(set(df.columns)):
            base_cols = [c for c in df.columns if c not in ["cycle", "temp_C", "temp_K", "condition_id", "capacity"]]
            ordered_cols = base_cols + ["temp_C", "temp_K", "condition_id", "cycle", "capacity"]
            df = df[ordered_cols]

        # (If not CALCE, keep original order except any inserted cycle index)

        # ------------------------------------------------------------
        # (C) Clean outliers
        # ------------------------------------------------------------
        df = self.delete_3_sigma(df)

        # ------------------------------------------------------------
        # (D) Normalize features and capacity->SOH if nominal_capacity provided
        # ------------------------------------------------------------
        if nominal_capacity is not None:
            df['capacity'] = df['capacity']/nominal_capacity

            f_df = df.iloc[:,:-1]  # all features except capacity

            if self.normalization_method == 'min-max':
                den = (f_df.max() - f_df.min())
                den = den.replace(0, 1.0)              # avoid divide-by-zero for constant columns
                f_df = 2*(f_df - f_df.min())/den - 1
                f_df = f_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)  # final safety

            elif self.normalization_method == 'z-score':
                f_df = (f_df - f_df.mean())/f_df.std()

            df.iloc[:,:-1] = f_df

        return df

    def load_one_battery(self,path,nominal_capacity=None):
        '''
        Read a csv file and divide the data into x and y
        :param path:
        :param nominal_capacity:
        :return:
        '''
        df = self.read_one_csv(path,nominal_capacity)
        x = df.iloc[:,:-1].values
        y = df.iloc[:,-1].values
        x1 = x[:-1]
        x2 = x[1:]
        y1 = y[:-1]
        y2 = y[1:]
        return (x1,y1),(x2,y2)

    def load_all_battery(self,path_list,nominal_capacity):
        '''
        Read multiple csv files, divide the data into X and Y, and then package it into a dataloader
        :param path_list: list of file paths
        :param nominal_capacity: nominal capacity, used to calculate SOH
        :param batch_size: batch size
        :return: Dataloader
        '''
        X1, X2, Y1, Y2 = [], [], [], []
        if self.args.log_dir is not None and self.args.save_folder is not None:
            save_name = os.path.join(self.args.save_folder,self.args.log_dir)
            write_to_txt(save_name,'data path:')
            write_to_txt(save_name,str(path_list))
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
        tensor_Y1 = torch.from_numpy(Y1).float().view(-1,1)
        tensor_Y2 = torch.from_numpy(Y2).float().view(-1,1)

        # Condition 1: 80/20 split then 80/20 train/valid inside train
        split = int(tensor_X1.shape[0] * 0.8)
        train_X1, test_X1 = tensor_X1[:split], tensor_X1[split:]
        train_X2, test_X2 = tensor_X2[:split], tensor_X2[split:]
        train_Y1, test_Y1 = tensor_Y1[:split], tensor_Y1[split:]
        train_Y2, test_Y2 = tensor_Y2[:split], tensor_Y2[split:]

        train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
            train_test_split(train_X1, train_X2, train_Y1, train_Y2, test_size=0.2, random_state=420)

        train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
        test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
                                 batch_size=self.args.batch_size,
                                 shuffle=False)

        # Condition 2: random 80/20 split
        train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
            train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, test_size=0.2, random_state=420)
        train_loader_2 = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
        valid_loader_2 = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)

        # Condition 3: no split (all as test)
        test_loader_3 = DataLoader(TensorDataset(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2),
                                 batch_size=self.args.batch_size,
                                 shuffle=False)

        loader = {'train': train_loader, 'valid': valid_loader, 'test': test_loader,
                  'train_2': train_loader_2,'valid_2': valid_loader_2,
                  'test_3': test_loader_3}
        return loader


class XJTUdata(DF):
    def __init__(self, root, args):
        super(XJTUdata, self).__init__(args)
        self.root = root
        self.file_list = os.listdir(root)
        self.variables = pd.read_csv(os.path.join(root, self.file_list[0])).columns
        self.num = len(self.file_list)
        self.batch_names = ['2C','3C','R2.5','R3','RW','satellite']
        self.batch_size = args.batch_size
