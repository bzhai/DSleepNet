import os.path
from copy import copy

import h5py
import torch
from torch.utils.data import DataLoader

from data_handler.sliding_window import sliding_window
from utilities.utils import *
from sleep_stage_config import Config
import numpy as np
import pandas as pd
from data_handler.dis_data_loader_base import WindowedFrameDisDataLoader

import yaml


def sliding_norm_data(seq_len, pid_list, id_dict, pid_file_dic, feature_list, scale):
    x_list, y_list, d_list, id_list = [], [], [], []
    for pid in pid_list:
        # read csv
        df = pd.read_csv(pid_file_dic[pid])
        standardize_df_given_feature(df, feature_list, scale, df_name="df", simple_method=True)
        # normalise data
        x, y = build_sliding_window_data(df, feature_list, seq_len)
        domain = [id_dict[pid]]
        domain = domain * len(y)
        x_list.append(x)
        y_list.append(y)
        d_list.append(domain)
        id_list.extend((int(pid) * 10000 + np.arange(1, len(y) + 1)).tolist())
    x, y, d, id = np.vstack(x_list), np.concatenate(y_list), np.concatenate(d_list), np.asarray(id_list)
    return x, y, d, id


def build_sliding_window_data(df, dl_feature_list, win_len):
    # print("building cached dataset for window length: %s pid : %s" % (win_len, id))
    features = copy(dl_feature_list)
    features.extend(["stages"])
    df_array = df[features].values
    num_cols = len(dl_feature_list) + 1  # the number of columns = len(feature) + gt
    mid_pos = int(np.floor(win_len / 2))
    padding = np.ones(shape=(mid_pos, num_cols)) * -1
    df_array = np.concatenate([padding, df_array, padding])  # padding the edges of time series
    sliding_data = sliding_window(df_array, (win_len + 1, num_cols), (1, num_cols))
    return sliding_data[:, :, :-1], sliding_data[:, mid_pos, -1]


def get_mesa_unifactor_dis_dict(cfg, dis_type='sleepage5c'):
    admin_df = pd.read_csv(os.path.join(get_project_root(), cfg.ADMIN_RECORD_PATH % 'mesa'))
    admin_df['mesaid'] = admin_df['mesaid'].map(lambda x: "%04d" % x)
    if dis_type == 'gender1':
        gender_dict = {0: 0, 1: 1}
        admin_df['gender_group'] = admin_df['gender1'].map(lambda x: gender_dict[x])
        mesa_dict = dict(zip(admin_df.mesaid.values, admin_df[dis_type].values))
        cate_group_name = "gender1"

    elif dis_type == 'bmi5c':
        # admin_df = admin_df[['mesaid', dis_type, 'bmicat5c']]
        mesa_dict = dict(zip(admin_df.mesaid.values, admin_df[dis_type].values))
        cate_group_name = "bmicat5c"

    elif dis_type == "ahi4pa5":
        # admin_df = admin_df[['mesaid', dis_type, 'ahi_group']]
        mesa_dict = dict(zip(admin_df.mesaid.values, admin_df.ahi_group.values))
        cate_group_name = "ahi_group"

    elif dis_type == 'sleepage5c':
        dis_type = dis_type.split('_')[0]
        # admin_df = admin_df[['mesaid', dis_type]]
        group_interal = 10
        min_age, max_age = admin_df[dis_type].min(), admin_df[dis_type].max()
        age_dict = {}
        lower_bound_age = min_age // group_interal if min_age < 10 else 10 * (min_age // group_interal)
        # ##### we still need age group here, as the age group is used to filter the admin df
        group_num = 0
        for i in np.arange(min_age, max_age + 1):
            if not (lower_bound_age + group_interal > i >= lower_bound_age):
                group_num += 1
                lower_bound_age += group_interal
            age_dict[i] = group_num
        admin_df['age_group'] = admin_df['sleepage5c'].map(lambda x: age_dict[x])
        mesa_dict = dict(zip(admin_df.mesaid.values, admin_df.age_group.values))
        cate_group_name = "age_group"

    elif dis_type == 'mesaid':
        id_df = pd.read_csv(cfg.TRAIN_TEST_SPLIT)
        id_df['uids'] = id_df['uids'].map(lambda x: "%04d" % x)
        train_id = id_df[id_df['segment'] == 'train'].uids.values.tolist()
        mesa_dict = dict(zip(train_id, np.arange(1, len(train_id) + 1)))
        all_id = admin_df.mesaid.unique()
        non_train_id = all_id.difference(train_id)
        for id in non_train_id:
            mesa_dict.update({id: len(train_id) + 1})
        cate_group_name = "id_group"

    else:  # non disentangle
        mesa_dict = dict(zip(admin_df.mesaid, admin_df.shape[0] * [0]))
        cate_group_name = ""
    return mesa_dict, admin_df, cate_group_name


def build_dis_dataloader(batch_size, x_train, y_train, d_train, train_idx, x_val, y_val, d_val, val_idx, x_test, y_test,
                         d_test, test_idx, shuffle):
    """
    This function will build the disentangle data loader by given sample wised domain data, inputs, and targets.
    @param batch_size:
    @param x_train:
    @param y_train:
    @param d_train:
    @param train_idx:
    @param x_val:
    @param y_val:
    @param d_val:
    @param val_idx:
    @param x_test:
    @param y_test:
    @param d_test:
    @param test_idx:
    @return:
    """
    # unique_y, y_counts = np.unique(y_train, return_counts=True)
    # weights = 100.0 / torch.Tensor(y_counts)
    # weights = weights.double()
    # sample_w = get_sample_weights(y_train, weights)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w),
    #                                                          replacement=True)
    train_ds = WindowedFrameDisDataLoader(x_train, y_train, d_train, train_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        # sampler=sampler
    )
    test_ds = WindowedFrameDisDataLoader(x_test, y_test, d_test, test_idx)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    val_ds = WindowedFrameDisDataLoader(x_val, y_val, d_val, val_idx)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader, test_loader


def build_pid_file_dict(pid, all_csv_files):
    pid_file_dict = {}
    for pid in pid:
        for file_path in all_csv_files:
            if str(pid) in file_path:
                pid_file_dict[pid] = file_path
    return pid_file_dict

def post_process_tvt(x_train, y_train, x_val, y_val, x_test, y_test, seq_len, num_classes, stage_casting_func):
    if seq_len != 100:
        x_train = x_train[:, 50 - seq_len // 2:50 + seq_len // 2 + 1, :]
        x_val = x_val[:, 50 - seq_len // 2:50 + seq_len // 2 + 1, :]
        x_test = x_test[:, 50 - seq_len // 2:50 + seq_len // 2 + 1, :]
    if (len(y_train.shape) < 2) and (len(set(y_train))) != num_classes:
        y_train = stage_casting_func(y_train.astype(int), num_classes)
    if (len(y_test.shape) < 2) and (len(set(y_test))) != num_classes:
        y_test = stage_casting_func(y_test.astype(int), num_classes)
    if (len(y_val.shape) < 2) and (len(set(y_val))) != num_classes:
        y_val = stage_casting_func(y_val, num_classes)
    return x_train, y_train, x_val, y_val, x_test, y_test


def __load_pa_group_cfg_file__(cfg, dataset, num_train, dis_type, train_test_group, feature_type):
    """
    The method will load the group constrain from the yaml file
    @param cfg: config file
    @param dataset: dataset name
    @param num_train: number of training subjects if the number of subjects is less than the given number, then using
                        all subjects
    @param dis_type: disentangle type code (e.g., ahi4pa5)
    @param train_test_group: train test group list (e.g., group54)
    @param feature_type: feature type (e.g., hrv, all, full etc.)
    @return: train constrain, test constrain, cache file path
    """
    with open(os.path.join(get_project_root(), "%s_exp_group_settings.yaml" % dataset)) as f:
        exp_config = yaml.load(f, Loader=yaml.FullLoader)
        train_constrain = exp_config[train_test_group[0]]['train']
        test_constrain = exp_config[train_test_group[0]]['test']
    if feature_type in ("all", "full"):
        cache_file = cfg.NN_ACC_HRV_DIS_TRAIN_NORM_REGROUP % (dataset, num_train, str(dis_type)+str(train_test_group), feature_type)
    elif feature_type == "hrv":
        cache_file = cfg.NN_HRV_DIS_TRAIN_NORM_REGROUP % (dataset, num_train, str(dis_type)+str(train_test_group), feature_type)
    print(f"total subjects for training is {num_train}")
    return train_constrain, test_constrain, cache_file
def get_unifactor_filter_regroup_loader(cfg: Config, batch_size, seq_len, num_classes, dataset, num_train, dis_type,
                                        train_test_group: list, train_shuffle=True, feature_type="all"):
    """
    The method will build the sliding window dataset on the fly
    """
    train_constrain, test_constrain, cache_file = __load_pa_group_cfg_file__(cfg, dataset, num_train, dis_type,
                                                                             train_test_group, feature_type)
    if type(dis_type) == list:
        dis_type = dis_type[0]
    if os.path.exists(cache_file):
        print("Loading cached data...")
        with h5py.File(cache_file, 'r') as data:
            x_train, y_train, d_train, train_idx = data['x_train'][:], data['y_train'][:], \
                data['d_train'][:], data['train_idx'][:]
            x_val, y_val, d_val, val_idx = data['x_val'][:], data['y_val'][:], data['d_val'][:], \
                data['val_idx'][:]
            x_test, y_test, d_test, test_idx = data['x_test'][:], data['y_test'][:], \
                data['d_test'][:], data['test_idx'][:]
    else:
        if dataset == "mesa":
            dis_dict, admin_df, group_name = get_mesa_unifactor_dis_dict(cfg, dis_type)
            id_col_name = "mesaid"

        admin_df['file_name'] = cfg.CSV30_DATA_PATH % dataset + os.sep + admin_df["file_name"]
        pid_file_dict = dict(zip(admin_df[id_col_name].values.tolist(), admin_df['file_name'].values.tolist()))

        train_group = [int(x) for x in train_constrain[dis_type]]
        test_group = [int(x) for x in test_constrain[dis_type]]
        train_val_id = admin_df[admin_df[group_name].isin(train_group)][id_col_name].tolist()
        test_id = admin_df[admin_df[group_name].isin(test_group)][id_col_name].tolist()

        print("Total number of subjects in train and val: %s" % len(train_val_id))
        print("Total number of subjects in test: %s" % len(test_id))
        test_id.sort(reverse=False)

        # here we need do a maximum value selection between given number of subjects and the total number of subjects
        num_train = len(train_val_id) if (num_train > len(train_val_id)) or num_train <= 0 else num_train
        train_val_id = train_val_id[:num_train]
        np.random.shuffle(train_val_id)
        train_id = train_val_id[0:int(np.round(len(train_val_id) * 0.8))]
        val_id = train_val_id[int(np.round(len(train_val_id) * 0.8)):]
        # calculate the normaliser

        # the feature list depends on the dataset
        feature_list = pd.read_csv(cfg.FEATURE_LIST % (dataset, feature_type))['feature_list'].values.tolist()
        train_df = []
        for pid in train_id:
            df_tmp = pd.read_csv(pid_file_dict[pid])
            df_tmp = df_tmp[feature_list]
            train_df.append(df_tmp)
        train_df = pd.concat(train_df, ignore_index=True)
        # calculate the standard deviation of each column
        # as we randomly split the train val dataset, so we should calculate the scaler each time.
        scale = standardize_df_given_feature(train_df, feature_list, df_name='train', simple_method=True)
        del train_df

        x_train, y_train, d_train, train_idx = sliding_norm_data(100, train_id, dis_dict, pid_file_dict,
                                                                 feature_list, scale)
        x_val, y_val, d_val, val_idx = sliding_norm_data(100, val_id, dis_dict, pid_file_dict, feature_list, scale)
        x_test, y_test, d_test, test_idx = sliding_norm_data(100, test_id, dis_dict, pid_file_dict, feature_list,
                                                             scale)
        print("Building H5 dataset....")

        if not os.path.exists(cache_file):
            with h5py.File(cache_file, 'w') as data:
                data['x_train'], data['y_train'], data['d_train'], data[
                    'train_idx'] = x_train, y_train, d_train, train_idx
                data['x_val'], data['y_val'], data['d_val'], data['val_idx'] = x_val, y_val, d_val, val_idx
                data['x_test'], data['y_test'], data['d_test'], data['test_idx'] = x_test, y_test, d_test, test_idx
    # make sure the sleep classes are casted if the not 5 stages
    if dataset == "shhs1":
        stage_casting_func = cast_sleep_stages_shhs
    else:
        stage_casting_func = cast_sleep_stages_mesa
    # post process the data, 1) cut to window length, 2) cast the sleep stages if not fit
    x_train, y_train, x_val, y_val, x_test, y_test = post_process_tvt(x_train, y_train, x_val, y_val, x_test, y_test, seq_len, num_classes, stage_casting_func)
    tr, va, te = build_dis_dataloader(batch_size, x_train, y_train, d_train, train_idx, x_val, y_val, d_val, val_idx,
                                      x_test, y_test, d_test, test_idx, shuffle=train_shuffle)
    return tr, va, te

def get_ml_dataset(cfg: Config, batch_size, seq_len, num_classes, dataset, num_train, dis_type,
                                        train_test_group: list, feature_type="all"):
    train_constrain, test_constrain, cache_file = __load_pa_group_cfg_file__(cfg, dataset, num_train, dis_type,
                                                                             train_test_group, feature_type)
    if type(dis_type) == list:
        if len(dis_type) == 1:
            dis_type = dis_type[0]
    if os.path.exists(cache_file):
        print("Loading cached data...")
        with h5py.File(cache_file, 'r') as data:
            x_train, y_train, d_train, train_idx = data['x_train'][:], data['y_train'][:], \
                data['d_train'][:], data['train_idx'][:]
            x_val, y_val, d_val, val_idx = data['x_val'][:], data['y_val'][:], data['d_val'][:], \
                data['val_idx'][:]
            x_test, y_test, d_test, test_idx = data['x_test'][:], data['y_test'][:], \
                data['d_test'][:], data['test_idx'][:]
    else:
        raise ValueError("No cached data found")
    stage_casting_func = cast_sleep_stages_mesa
    x_train, y_train, x_val, y_val, x_test, y_test = post_process_tvt(x_train, y_train, x_val, y_val, x_test, y_test,
                                                                      seq_len, num_classes, stage_casting_func)
    # for traditional machine learning, we need to combin the train and val data
    x_train = np.concatenate([x_train, x_val], axis=0)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_train = np.concatenate([y_train, y_val], axis=0)
    d_train = np.concatenate([d_train, d_val], axis=0)
    train_idx = np.concatenate([train_idx, val_idx], axis=0)
    return x_train, y_train, d_train, train_idx, x_test, y_test, d_test, test_idx

def get_multifactor_filter_regroup_loader(cfg: Config, batch_size, seq_len, num_classes, dataset, num_train,
                                          dis_type, train_test_group: list, train_shuffle=True, feature_type="all",
                                          mask_att=[]):
    """
    The method will build the sliding window dataset on the fly
    """
    train_constrain, test_constrain, cache_file = __load_pa_group_cfg_file__(cfg, dataset, num_train, dis_type,
                                                                             train_test_group, feature_type)
    unmask = []
    for i in range(len(dis_type)):
        if dis_type[i] not in mask_att:
            unmask.append(i)
    unmask = np.array(unmask)
    print(f"check cacha file: {cache_file}")
    if os.path.exists(cache_file):
        print(f"Loading cached data {cache_file}")
        with h5py.File(cache_file, 'r') as data:
            x_train, y_train, d_train, train_idx = data['x_train'][:], data['y_train'][:], data['d_train'][:], data['train_idx'][:]
            x_val, y_val, d_val, val_idx = data['x_val'][:], data['y_val'][:], data['d_val'][:], data['val_idx'][:]
            x_test, y_test, d_test, test_idx = data['x_test'][:], data['y_test'][:], data['d_test'][:], data['test_idx'][:]
    else:
        print("Building H5 dataset....")
        # based on the constraints, we need to get the train, test, and validation id
        admin_df = pd.read_csv(os.path.join(get_project_root(), cfg.ADMIN_RECORD_PATH % 'mesa'))
        if dataset == "mesa":
            unifactor_dis_dict = get_mesa_unifactor_dis_dict
            id_col_name = "mesaid"
            admin_df[id_col_name] = admin_df[id_col_name].map(lambda x: "%04d" % x)
        # load the admin df
        admin_df['file_name'] = cfg.CSV30_DATA_PATH % dataset + os.sep + admin_df["file_name"]
        # build a dictionary for format {pid: [constraint value list]}
        pa_cols = list(train_constrain.keys())  # get all the constraints
        pid_file_dict = dict(zip(admin_df[id_col_name].values.tolist(), admin_df['file_name'].values.tolist()))

        # filter the admin_df into train and test
        train_admin_df, test_admin_df = admin_df.copy(deep=True), admin_df.copy(deep=True)
        for col, val in train_constrain.items():
            _, _, train_cate_group_name = unifactor_dis_dict(cfg, col)
            train_admin_df = train_admin_df[train_admin_df[train_cate_group_name].isin(val)]

        for col, val in test_constrain.items():
            _, _, test_cate_group_name = unifactor_dis_dict(cfg, col)
            test_admin_df = test_admin_df[test_admin_df[test_cate_group_name].isin(val)]

        pas = admin_df[pa_cols].values.tolist()  # get all the values of constraint
        # these two dictionaries are used for train and test.
        pid_constraint_dict = dict(zip(admin_df[id_col_name].values, pas))

        train_val_id = train_admin_df.mesaid.values.tolist()
        test_id = test_admin_df.mesaid.values.tolist()

        print("Total number of subjects in train and val: %s" % len(train_val_id))
        print("Total number of subjects in test: %s" % len(test_id))
        test_id.sort(reverse=False)

        num_train = len(train_val_id) if (num_train > len(train_val_id)) or num_train <= 0 else num_train
        train_val_id = train_val_id[:num_train]
        np.random.shuffle(train_val_id)
        train_id = train_val_id[0:int(np.round(len(train_val_id) * 0.8))]
        val_id = train_val_id[int(np.round(len(train_val_id) * 0.8)):]
        # calculate the normaliser
        all_csv_files = get_all_files_include_sub(cfg.CSV30_DATA_PATH % dataset, ".csv")
        feature_list = pd.read_csv(cfg.FEATURE_LIST % (dataset, feature_type))['feature_list'].values.tolist()
        train_df = []
        print("loading csv for building a standard scaler...")
        for file_path in all_csv_files:
            pid = os.path.basename(file_path).split('_')[0]
            if pid in train_id:
                df_tmp = pd.read_csv(file_path)
                # df_tmp = df_tmp[feature_list]
                train_df.append(df_tmp)
        train_df = pd.concat(train_df, ignore_index=True)
        scale = standardize_df_given_feature(train_df, feature_list, df_name='train',
                                             simple_method=True)
        del train_df
        x_train, y_train, d_train, train_idx = sliding_norm_data(100, train_id, pid_constraint_dict, pid_file_dict,
                                                                 feature_list, scale)
        x_val, y_val, d_val, val_idx = sliding_norm_data(100, val_id, pid_constraint_dict, pid_file_dict,
                                                         feature_list, scale)
        x_test, y_test, d_test, test_idx = sliding_norm_data(100, test_id, pid_constraint_dict, pid_file_dict,
                                                             feature_list, scale)
        print("Building H5 dataset....")
        if not os.path.exists(cache_file):
            with h5py.File(cache_file, 'w') as data:
                data['x_train'], data['y_train'], data['d_train'], data[
                    'train_idx'] = x_train, y_train, d_train, train_idx
                data['x_val'], data['y_val'], data['d_val'], data['val_idx'] = x_val, y_val, d_val, val_idx
                data['x_test'], data['y_test'], data['d_test'], data['test_idx'] = x_test, y_test, d_test, test_idx

    # post process the data
    d_train = d_train[:, unmask]
    d_val = d_val[:, unmask]
    d_test = d_test[:, unmask]
    if dataset == "shhs1":
        stage_casting_func = cast_sleep_stages_shhs
    else:
        stage_casting_func = cast_sleep_stages_mesa
    x_train, y_train, x_val, y_val, x_test, y_test = \
        post_process_tvt(x_train, y_train, x_val, y_val, x_test, y_test, seq_len, num_classes, stage_casting_func)
    tr, va, te = build_dis_dataloader(batch_size, x_train, y_train, d_train, train_idx, x_val, y_val, d_val, val_idx,
                                      x_test, y_test, d_test, test_idx, shuffle=train_shuffle)
    return tr, va, te

