import pandas as pd
from pathlib import Path
import numpy as np

from sleep_stage_config import Config
from utilities.utils import make_one_block, standardize_df_given_feature, cast_sleep_stages_mesa
import os
import pickle
from tqdm import tqdm

def get_project_root() -> Path:
    return Path(r'G:\AcademicCode\applewatch_dataprocessing_disentangle')

FEATURE_FILE_PATH = r"G:\AcademicCode\applewatch_dataprocessing_disentangle\outputs\features"
OUTPUT_PATH_FILE = os.path.join(FEATURE_FILE_PATH, "apple_hrv30s_acc30s_full_feat_stand.h5")
subject_ids = [46343, 3509524, 5132496, 1066528, 5498603, 2638030, 2598705, 5383425, 1455390, 4018081, 9961348,
                    1449548, 8258170, 781756, 9106476, 8686948, 8530312, 3997827, 4314139, 1818471, 4426783,
                    8173033, 7749105, 5797046, 759667, 8000685, 6220552, 844359, 9618981, 1360686,
                    8692923]


def fix_apple_sleep_stages(data, classes=5):
    if type(data) is np.ndarray:
        data[data == 4] = 3  # non-REM 3 combined NREM4 to NREM3
        data[data == 5] = 4  # index 5 move to index 4
        return data
    else:
        # this is for a scalar
        stages_dict = {}

        # dataset=0 wake, dataset=1:non-REM, dataset=2:non-REM, dataset=3:non-REM, dataset=4:REM
        stages_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5:4}
        return stages_dict[data]
        return data


def build_hold_out_h5df(cfg: Config):
    tmp = []
    for subject_id in tqdm(subject_ids):
        # print("Processing pid: %s" % subject_id)
        # read hr features
        # read act features
        hr_features = pd.read_csv(str(get_project_root().joinpath('outputs/features/', str(subject_id) + "_hr_feature.out")), sep=' ')
        act_features = pd.read_csv(str(get_project_root().joinpath('outputs/features/', str(subject_id) + "_count_feature.out")), delimiter=' ', header=None).values
        sleep_stages = pd.read_csv(str(get_project_root().joinpath('outputs/features/', str(subject_id) + "_psg_labels.out")), delimiter=' ', header=None).values
        combined_dataset = hr_features
        combined_dataset['activity'] = act_features
        combined_dataset['stages'] = sleep_stages
        df_columns = cfg.APPLE_ACC_HRV_FEATURE_LIST + ["linetime", "stages", 'appleid', 'window_idx']
        feature_list =  cfg.APPLE_ACC_HRV_FEATURE_LIST

        # df_tmp = pd.DataFrame(combined_dataset)
        # df_tmp.columns = df_columns

        gt_true = combined_dataset[combined_dataset["stages"] > 0]
        if gt_true.empty:
            print("Ignoring subject's file %s" % subject_id)
            continue
        start_block = combined_dataset.index.get_loc(gt_true.index[0])
        end_block = combined_dataset.index.get_loc(gt_true.index[-1])
        combined_dataset["gt_sleep_block"] = make_one_block(combined_dataset["stages"], start_block, end_block)
        combined_dataset["appleid"] = subject_id
        combined_dataset["window_idx"] = subject_id*10000+combined_dataset.index
        combined_dataset = combined_dataset[df_columns]
        tmp.append(combined_dataset)
    test_proportion = 0.2
    whole_df = pd.concat(tmp)
    # sort the dataframe by "window_idx"
    whole_df = whole_df.sort_values(by="window_idx").reset_index(drop=True)

    whole_df['stages'] = whole_df['stages'].apply(lambda x: fix_apple_sleep_stages(x))
    del tmp
    print("start standardisation on df_train....")
    scaler = standardize_df_given_feature(whole_df, feature_list, df_name="whole_df", simple_method=True)

    store = pd.HDFStore(OUTPUT_PATH_FILE, 'w')
    store["data"] = whole_df
    store["featnames"] = pd.Series(feature_list)
    store.close()
    print('h5 dataset is saved to %s' % OUTPUT_PATH_FILE)
    with open(os.path.join(OUTPUT_PATH_FILE + '_std_transformer'), "wb") as f:
        pickle.dump(scaler, f)
    print("all completed")

if __name__ == '__main__':
    cfg = Config()
    build_hold_out_h5df(cfg)