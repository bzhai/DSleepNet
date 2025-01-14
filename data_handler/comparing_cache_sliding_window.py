import numpy as np
import pandas as pd

from data_handler.dis_train_norm_dataloader import build_sliding_window_data
from data_handler.dis_data_processing import get_data
from sleep_stage_config import Config
from utilities.utils import get_all_files_include_sub


def comparing_sliding_window_cache_data():
    cfg = Config()
    id_df = pd.read_csv(cfg.TRAIN_TEST_SPLIT)
    id_df['uids'] = id_df['uids'].map(lambda x: "%04d" % x)
    train_id = id_df[id_df['segment'] == 'train'].uids.values.tolist()
    val_id = id_df[id_df['segment'] == 'val'].uids.values.tolist()
    test_id = id_df[id_df['segment'] == 'test'].uids.values.tolist()
    all_h5_files = get_all_files_include_sub(cfg.CSV30_DATA_PATH, ".csv")
    dl_feature = ["_Act", "mean_nni", "sdnn", "sdsd", "vlf", "lf", "hf", "lf_hf_ratio",
                  "total_power"]
    for f in all_h5_files:
        # build sliding window using new method.
        df = pd.read_csv(f)
        x, y = build_sliding_window_data(df, dl_feature, 100)
        # build sliding window using old method
        old_x, old_y = get_data(df, 100, dl_feature)
        print("x diff is %s " % np.abs(x - old_x).sum())
        print("y diff is %s " % np.abs(y - old_y).sum())
        assert  np.abs(x - old_x).sum() + np.abs(y - old_y).sum() == 0

    return True

if __name__ == "__main__":
    results = comparing_sliding_window_cache_data()