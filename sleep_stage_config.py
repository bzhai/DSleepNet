import os
import platform


class Config(object):
    def __init__(self):

        self.ANALYSIS_SUB_FOLDER = {'sleep_period': "sp_summary", "recording_period": "summary"}
        self.MESA_ACC_HR_STATISTIC_FEATURE_LIST = ["activity", "min_hr", "max_hr", "mean_hr", "skw_hr", "kurt_hr", "std_hr"]
        self.APPLE_ACC_HRV_FEATURE_LIST = ["activity", "mean_nni", "sdnn", "sdsd", "vlf", "lf", "hf", "lf_hf_ratio",
         "total_power"]
        self.SUMMARY_FILE_PATH = r"./exp_results.csv"
        self.SUMMARY_FOLDER_DICT = {"s": r"./sp_exp_results.csv",
                                    "r": r"./rp_exp_results.csv", }
        self.ADMIN_RECORD_PATH = r"./%s_admin_df.csv"  # the root of the repository has a file called mesa_admin_df.csv contains the subject personal attributes and their groups.
        self.FEATURE_LIST = "%s_%s_feature_list.csv"  # This file contains the features (columns) utilized in the experiments.
        self.EXP_DIR_SETTING_FILE = r"./exp_dir_setting.yaml"
        # please only change the directory part for the following setting variables e.g. G:\\tmp\\sleep\\opensource
        if platform.uname()[1] == 'BB-WIN11':  # change this value to your computer's name
            self.CSV30_DATA_PATH = "G:\\tmp\\sleep\\opensource\\%s\\Aligned_final" # This folder contains aligned ACC and HRV features for each sleep epoch of every participant.

            self.NN_ACC_HRV_DIS_TRAIN_NORM_REGROUP = "G:\\tmp\\sleep\\opensource\\%s\\HRV30s_ACC30s_H5\\dis\\nn_acc_hrv30s_%d_%s_%s_regroup_samples.h5"  #
            self.NN_HRV_DIS_TRAIN_NORM_REGROUP = "G:\\tmp\\sleep\\opensource\\%s\\HRV30s_ACC30s_H5\\dis\\nn_hrv30s_%d_%s_%s_regroup_samples.h5"

            self.TRAIN_TEST_SPLIT = "./assets/train_test_pid_split.csv"  # this table only maintain the MESA id

            self.APPLE_HRV30_ACC_STD_PATH = r"G:\tmp\sleep\opensource\applewatch_dataprocessing_disentangle\outputs\features\apple_hrv30s_acc30s_full_feat_stand.h5"

            self.APPLE_NN_ACC_HRV = r"G:\tmp\sleep\opensource\Apple_watch_disentangle\apple_loocv_windowed_%d.h5" # this is the processed AppleWatch dataset
