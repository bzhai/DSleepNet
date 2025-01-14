from sklearn.preprocessing import OneHotEncoder
import h5py
from utilities.utils import *
from sleep_stage_config import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class AppleSleepDataLoader(object):
    """
    a dataset loader for actigraphy
    """

    def __init__(self, cfg: Config, modality, num_classes, seq_len):
        self.config = cfg
        self.modality = modality
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.x = []
        self.y = []
        self.dl_feature_list = []
        self.__prepare_feature_list__()

    def __prepare_feature_list__(self):
        if self.modality == "all":
            self.dl_feature_list = ["activity", "min_hr", "max_hr", "mean_hr", "skw_hr", "kurt_hr", "std_hr"]
        elif self.modality == "hr":
            self.dl_feature_list = ["min_hr", "max_hr", "mean_hr", "skw_hr", "kurt_hr", "std_hr"]
        elif self.modality == "acc":
            self.dl_feature_list = ["activity"]
        elif self.modality == "hr":
            self.dl_feature_list = ["mean_hr"]
        elif self.modality == "hrv_acc":
            self.dl_feature_list = ["activity", "mean_nni", "sdnn", "sdsd", "vlf", "lf", "hf", "lf_hf_ratio",
                                    "total_power"]

    @staticmethod
    def __check_seq_len__(seq_len):
        if seq_len not in [100, 50, 20]:
            raise Exception("seq_len i error")

    def build_windowed_cache_data(self, win_len):
        self.__check_seq_len__(win_len)
        print("Loading H5 dataset....")
        df, feat_names = load_h5_df_train_test_dataset(self.config.APPLE_HRV30_ACC_STD_PATH)
        cache_path = self.config.APPLE_NN_ACC_HRV % win_len
        print("building cached dataset for window length: %s ....." % win_len)
        x, y = get_data(df, win_len, self.dl_feature_list, pid_col_name="appleid",
                        gt_col_name="stages")
        _, idx = get_data(df, win_len, self.dl_feature_list, pid_col_name="appleid",
                        gt_col_name="window_idx")

        with h5py.File(cache_path, 'w') as data:
            data["x"] = x
            data["y"] = y
            data['idx'] = idx
            data.close()

    def _load_std_df(self):
        """
        LOOCV will concatenate train and test into a single frame. The train test index for each fold in CV
        is stored in a separate file
        """
        df, feature_name = load_h5_df_train_test_dataset(self.config.APPLE_HRV30_ACC_STD_PATH)
        df['stages'] = df['stages'].apply(lambda x: cast_sleep_stages_mesa(x, classes=self.num_classes))
        # df_test['stages'] = df_test['stages'].apply(lambda x: cast_sleep_stages_mesa(x, classes=self.num_classes))

        return df, feature_name


if __name__ == '__main__':
    config = Config()
    window_len = 100
    apple_windowed_builder = AppleSleepDataLoader(cfg=config, modality="hrv_acc", num_classes=3, seq_len=window_len)
    apple_windowed_builder.build_windowed_cache_data(window_len)

