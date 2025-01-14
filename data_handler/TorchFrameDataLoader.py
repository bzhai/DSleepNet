"""
This script should only contain the frame to label data loaders
"""
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from utilities.utils import *
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from sleep_stage_config import Config
import numpy as np
import pandas as pd


class WindowedFrameDataLoader2D(torch.utils.data.Dataset):
    def __init__(self, data, target, idx, transform=None):
        self.data = torch.from_numpy(data).float()
        self.data = self.data.permute(0, 2, 1)  #  set it to batch_num, channel, time_dim
        self.data = self.data.unsqueeze(1)
        self.idx = torch.from_numpy(idx)
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        i = self.idx[index]
        if self.transform:
            x = self.transform(x)
        return x, y, i

    def __len__(self):
        return len(self.data)


class WindowedFrameDataLoader(torch.utils.data.Dataset):
    def __init__(self, data, target, idx, transform=None):
        self.data = torch.from_numpy(data).float()
        self.data = self.data.permute(0, 2, 1)  #  set it to batch_num, channel, time_dim
        self.idx = torch.from_numpy(idx)
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        i = self.idx[index]
        if self.transform:
            x = self.transform(x)
        return x, y, i

    def __len__(self):
        return len(self.data)


def get_windowed_apple_dataset(cfg, args, batch_size, num_classes, data_format='NCW'):
    """
    The method will read pre-windows acc and hrv data from H5PY
    """
    cache_path = cfg.APPLE_NN_ACC_HRV % args.seq_len
    import h5py as h5py
    with h5py.File(cache_path, 'r') as data:
        # df_data = data["df_values"][:]
        x = data["x"][:]
        y = data["y"][:]
        idx = data["idx"][:]
        # columns = data["columns"][:].astype(str).tolist()
        data.close()
    if data_format == 'NCW':
        x = x.transpose(0, 2, 1)
    # split_df = pd.read_csv(cfg.APPLE_LOOCV_PID_PATH)
    # all_pid = split_df['pid'].values.tolist()

    # all_idx = df[df.pid.isin(all_pid)]['window_idx'].values.astype(int)

    print("...Loading windowed cache dataset from %s" % cache_path)

    # make sure the sleep classes are casted if the not 5 stages
    if (len(y.shape) < 2) and (len(set(y))) != num_classes:
        y = cast_sleep_stages_mesa(y.astype(int), num_classes)

    ds = WindowedFrameDataLoader(x, y, idx)

    data_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return data_loader

def get_mesa_loocv_ids(cfg:Config, fold):
    split_df = pd.read_csv(cfg.MESA_LOOCV_PID_PATH)
    train_pid = split_df[(split_df['set_type']=="train") & (split_df['fold_num']==fold)]['pid'].values.tolist()
    val_pid = split_df[(split_df['set_type']=="val") & (split_df['fold_num']==fold)]['pid'].values.tolist()
    test_pid = split_df[(split_df['set_type']=="test") & (split_df['fold_num']==fold)]['pid'].values.tolist()
    return train_pid, val_pid, test_pid

# if __name__ == "__main__":
#     cfg = Config()
#     # train_loader, test_loader, val_loader = get_windowed_train_test_val_loader(cfg, 100, 100, 3, "apple")
#     train_loader = get_windowed_apple_loader(cfg, 100, 3)
#     for i, data in enumerate(train_loader):
#         print(data[0].shape)
#     print("output shape")