import yaml
import os
from easydict import EasyDict as edict

from models.layer4CNN import qzd
from utilities.utils import get_project_root

def get_model_time_step_dim(model_name, seq_len):
    with open(os.path.join(get_project_root(), "model_settings.yaml")) as f:
        exp_config = edict(yaml.load(f))
    fc_dim = exp_config[model_name].FC_TEMP_STEPS[str(seq_len)]
    return fc_dim


def get_num_in_channel(dataset_name="mesa", feature_type="all"):
    if dataset_name == "mesa":
        if feature_type == "all":
            in_channel = 9
        elif feature_type == "hrv":
            in_channel = 8
        elif feature_type == "full":
            in_channel = 28
        elif feature_type == "hr_statistic":
            in_channel = 7
    elif dataset_name == "apple":
        if feature_type == "all":
            in_channel = 9
        else:
            raise Exception("feature type not existed!")
    else:
        raise ValueError("Sorry, dataset is not recognised should be mesa, mesa_hr_statistic, apple")
    return in_channel

def build_model(nn_type, num_classes, fc_dim, chs):
    if nn_type == "qzd":
        model = qzd(in_channels=chs, num_classes=num_classes, fc_dim=fc_dim)
    else:
        raise Exception("model specified not existed!")
    return model

