# -*-coding:utf-8-*-
import os
import sys
module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
import numpy as np
import h5py
data_path = "G:\\tmp\sleep\\feature_analysis\\GSNMSE3DIS\\20220420-235014"
train_path = os.path.join(data_path, "train.h5")
with h5py.File(train_path, "r") as data:
    train_feature_y = data["zy_q"][:]
    train_feature_d = data["zd_q"][:]
    train_y = data["y"][:]
    train_uidx = data["idx"][:]

admin_df = pd.read_csv(fr"G:\issmp_dis\mesa_admin_df.csv")
def extract_sample_index(train_uidx, pids, train_y, num_subjects = 50, num_samples = 150):
    # num_subjects = 50 # how many subjects to choose
    # num_samples = 150  # how many samples per subject to choose

    train_sample_array_index_list = []
    # get the unique train id
    train_pids = list(set(train_uidx//10000))
    len(train_pids)

    # select the first num_subjects subjects
    # np.random.shuffle(train_pids)
    # filter train_pids by the given pids
    selected_pid_train = list(set(train_pids).intersection(set(pids)))
    selected_pid_train = train_pids[:num_subjects]

    for train_pid in selected_pid_train:
        # get the array indices correpond to a train id, array([ 999409,  999410,  999411, ..., 1000481, 1000482, 1000483], dtype=int64)
        tmp_sample_idx = np.where(np.isin(train_uidx // 10000, train_pid))[0]
        # select num_samples/3 of samples for each label
        for label in [0, 1, 2]:
            # get the indices of samples for each sleep stage
            label_idx = tmp_sample_idx[np.where(train_y[tmp_sample_idx]==label)]
            label_idx = list(label_idx)
            np.random.shuffle(label_idx)
            label_idx = label_idx[:int(num_samples/3)]
            train_sample_array_index_list.extend(label_idx)
    # select_pid_train_indices = np.where(np.isin(train_uidx//10000, selected_pid_train[0]))[0]
    return train_sample_array_index_list
from utilities.utils import generate_tsne
num_subjects = 50
num_samples = 150
# now let's plot the TSNE for each category of AHI from the admin_df
# first let's get the AHI category for each subject ahi4pa5, AHI training only has two categories
# now let's plot the TSNE for group 2 and 3 separately
for ahi_group in [2, 3]:
    ahi_group_pids = admin_df[admin_df['ahi_group'] == ahi_group].mesaid.unique().tolist()
    train_sample_array_index_list = extract_sample_index(train_uidx, ahi_group_pids, train_y=train_y, num_subjects = num_subjects, num_samples = num_samples)
    # this is to plot the TSNE of sleep stage's representation
    generate_tsne(data=train_feature_y[train_sample_array_index_list], num_class=3, gt=train_y[train_sample_array_index_list],
                              output_path=data_path, title=fr"train_y_{num_subjects}_subjects_{num_samples}_samples_ahi_group_{ahi_group}",percentage_num_samples=0.5, legend=False)
    # this is to plot the TSNE of ahi representation
    generate_tsne(data=train_feature_d[train_sample_array_index_list], num_class=3, gt=train_y[train_sample_array_index_list],
                              output_path=data_path, title=fr"train_d_{num_subjects}_subjects_{num_samples}_samples_ahi_group_{ahi_group}", percentage_num_samples=0.5, legend=False)

# this is to plot the TSNE of sleep stage's representation from all subjects
pids = admin_df.mesaid.unique().tolist()
train_sample_array_index_list = extract_sample_index(train_uidx, pids, train_y=train_y, num_subjects = num_subjects, num_samples = num_samples)
generate_tsne(data=train_feature_y[train_sample_array_index_list], num_class=3, gt=train_y[train_sample_array_index_list],
                          output_path=data_path, title=fr"train_y_{num_subjects}_subjects_{num_samples}_samples",percentage_num_samples=0.2)
# this is to plot the TSNE of sleep stage's representation
generate_tsne(data=train_feature_d[train_sample_array_index_list], num_class=3, gt=train_y[train_sample_array_index_list],
                          output_path=data_path, title=fr"train_d_{num_subjects}_subjects_{num_samples}_samples", percentage_num_samples=0.2)


