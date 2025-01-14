from copy import copy
import itertools
import os
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import h5py

from utilities.utils import get_all_files_include_sub, get_char_split_symbol, make_one_block, \
    standardize_df_given_feature, cast_sleep_stages_mesa
from sleep_stage_config import Config
from data_handler.sliding_window import sliding_window


def load_h5_df_dataset(path):
    """
    This function needs to be removed, it's a part of data loader
    """
    feature_name = []
    start = time.time()
    store = pd.HDFStore(path, 'r')
    dftrain = store["train"]
    dftest = store["test"]

    feature_name = store["featnames"].values.tolist()
    if type(feature_name[0]) is list:
        feature_name = list(itertools.chain.from_iterable(feature_name))
    store.close()
    print("loading dataset spend : %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() -start)))
    return dftrain, dftest, feature_name


def extract_x_y(df, seq_len, mesaid, label_posi='mid', feature=""):
    df_x = df[df["mesaid"] == mesaid][[feature, "stages"]].copy()
    y = df_x["stages"].astype(int).values  # get the ground truth for y
    del df_x["stages"]
    if label_posi == 'mid':
        for s in range(1, round(seq_len/2) + 1):
            df_x["shift_%d" % s] = df_x[feature].shift(s)
        # reverse columns
        columns = df_x.columns.tolist()
        columns = columns[::-1]  # or data_frame = data_frame.sort_index(ascending=True, axis=0)
        df_x = df_x[columns]
        for s in range(1, round(seq_len/2) + 1):
            df_x["shift_-%d" % s] = df_x[feature].shift(-s)
    else:
        for s in range(1, seq_len+1):
            df_x["shift_%d" % s] = df_x["activity"].shift(s)
    x = df_x.fillna(-1).values
    return x, y


def get_data(df, seq_len, feature_list):
    # build dataset by participant ID, extract dataset using sliding window method.
    final_x = []
    # loop all mesa_ids
    for feature in feature_list:
        mesaids = df.mesaid.unique()
        x, y = extract_x_y(df, seq_len, mesaids[0], label_posi='mid', feature=feature)
        for mid in mesaids[1:]:
            x_tmp, y_tmp = extract_x_y(df, seq_len, mid, label_posi='mid', feature=feature)
            x = np.concatenate((x, x_tmp))
            y = np.concatenate((y, y_tmp))
        x = np.expand_dims(x, -1)
        final_x.append(x)
    combined_x = np.concatenate(final_x, axis=-1)
    return combined_x, y


def get_data_with_pid(df, seq_len, feature_list):
    """

    @param df: data frame needs a column of domain.
    @param seq_len: the sliding window length
    @param feature_list:  feature list for iteration.
    @return:
    """
    # build dataset by participant ID, extract dataset using sliding window method.
    final_x = []
    # loop all mesa_ids
    d_index = []

    for feature in feature_list:
        mesaids = df.mesaid.unique()
        x, y = extract_x_y(df, seq_len, mesaids[0], label_posi='mid', feature=feature)
        if len(mesaids) > 1:
            for mid in mesaids[1:]:
                x_tmp, y_tmp = extract_x_y(df, seq_len, mid, label_posi='mid', feature=feature)
                x = np.concatenate((x, x_tmp))
                y = np.concatenate((y, y_tmp))
        x = np.expand_dims(x, -1)
        final_x.append(x)
    combined_x = np.concatenate(final_x, axis=-1)
    id_list = []
    for pid in mesaids:
        id_list.extend(df[df['mesaid'] == pid].shape[0] * [pid])
        d_index.extend(df[df['mesaid'] == pid]['domain'].values.tolist())
    d_index = np.asarray(d_index)
    return combined_x, y, d_index, id_list




def get_mesa_loocv_ids(cfg: Config, fold):
    split_df = pd.read_csv(cfg.MESA_LOOCV_PID_PATH)
    train_pid = split_df[(split_df['set_type'] == "train") & (split_df['fold_num'] == fold)]['pid'].values.tolist()
    val_pid = split_df[(split_df['set_type'] == "val") & (split_df['fold_num'] == fold)]['pid'].values.tolist()
    test_pid = split_df[(split_df['set_type'] == "test") & (split_df['fold_num'] == fold)]['pid'].values.tolist()
    return train_pid, val_pid, test_pid