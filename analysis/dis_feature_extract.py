import argparse
import h5py
from data_handler.get_dataloader import build_dataloader
from utilities.utils import *
from sleep_stage_config import Config
from copy import copy
from pathlib import Path
import torch
import sys
import yaml
import os

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def get_total_folds(dataset_name):
    """
    return total folds based on the dataset name
    """
    if dataset_name == "mesa":
        total_fold = 1
    elif dataset_name == "apple":
        total_fold = 16
    elif dataset_name == "mesa_hr_statistic":
        total_fold = 1
    else:
        raise ValueError("Dataset is not recognised")
    return total_fold


def set_up_optimizers(parameters):
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    return optimizer


def main(args):
    # torch.backends.cudnn.enabled = False

    cfg = Config()
    total_folds = get_total_folds(args.dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = args.exp_path
    model_path = get_best_pth(0, model_path)
    model = torch.load(model_path)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    print("Model: %s" % args.nn_type)
    # print(" Launch TensorBoard with: tensorboard --logdir=%s" % tracker.tensorboard_path)
    print_args(args)

    for fold_num in np.arange(total_folds):
        # summary(model, (get_num_in_channel(args.dataset), int(args.seq_len)+1))  # print the model summary
        print("Loading train, val, test dataset...")
        train_loader, val_loader, test_loader = build_dataloader(cfg, args, fold_num, train_shuffle=False)
        print("Total training samples: %s" % train_loader.dataset.__len__())
        print("Total testing samples: %s" % test_loader.dataset.__len__())
        setup_seed(args.seed)
        for loader_name, data_loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            # ***************** Training ************************
            # control the accumulated samples that added to gt and pred to calculate metrics at the last batch

            train_idx_list = []
            zd_q_list = []
            zy_q_list = []
            y_list = []
            for batch_idx, (x, y, d, train_idx) in enumerate(data_loader):
                if batch_idx > 0 and args.debug == 1:
                    continue
                x, y, d = x.to(device), y, d.to(device)

                zy_q, zd_q = model.get_features(x)

                zy_q_feature = copy(zy_q.detach().cpu().numpy())
                zd_q_feature = copy(zd_q.detach().cpu().numpy())
                y_list.extend(list(y.numpy()))
                zy_q_list.append(zy_q_feature)
                zd_q_list.append(zd_q_feature)
                train_idx_list.extend(train_idx.cpu().numpy().tolist())

            export_path = os.path.join(args.export_feature_path, args.nn_type,
                                       args.exp_path.split("\\")[-1])
            Path(export_path).mkdir(parents=True, exist_ok=True)
            zy_q_list = np.vstack(zy_q_list)
            zd_q_list = np.vstack(zd_q_list)
            # y_list = np.vstack(y_list)

            with h5py.File(os.path.join(export_path, fr"{loader_name}.h5"), "w") as data:
                data['zy_q'] = zy_q_list
                data['zd_q'] = zd_q_list
                data['y'] = np.asarray(y_list)
                data['idx'] = np.asarray(train_idx_list)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # specialised parameters
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size of training')
    parser.add_argument('--dataset', type=str, default='mesa', help='name of dataset')
    parser.add_argument('--exp_path', type=str, default=fr'D:\My Drive\AllCode\issmp_dis\tfboard\mesa\GSNMSE\20220414-173956') #20220307-135648
    parser.add_argument('--n_feature', type=int, default=9, help='name of feature dimension')
    parser.add_argument('--n_class', type=int, default=3, help='number of class')
    parser.add_argument('--d_AE', type=int, default=50, help='dim of AE')
    parser.add_argument('--x-dim', type=int, default=1152, help='input size after flattening')
    parser.add_argument('--num_train', type=int, default=1500, help='number of subjects for forwarding')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes or labels')
    parser.add_argument('--target_domain', type=str, default='S1', help='the target domain, [S1, S2, S3, S4]')
    parser.add_argument('--n_domains', type=int, default=1, help='number of total domains actually')
    # parser.add_argument('--n_target_domains', type=int, default=1, help='number of target domains')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--debug', type=int, default=0, help='debug model')
    parser.add_argument('--feature_type', type=str, default="all", help="all, hrv, hr, and so on")
    parser.add_argument('--comments', type=str, default="", help="comments to append")
    parser.add_argument('--seed', type=int, default=42, help="fix seed")
    parser.add_argument('--export_feature_path', type=str, default='L:\\tmp\\sleep\\feature_analysis',
                        help="default feature analysis saving path")

    args = parser.parse_known_args(argv)[0]
    pretrained_args_df = pd.read_csv(os.path.join(args.exp_path, "args.csv"), delimiter=":", header=None, nrows=39)
    pretrained_args_df.columns = ["arg_name", "arg_value"]
    for _, row in pretrained_args_df.iterrows():
        if row["arg_name"] in ['nn_type', 'train_group', 'test_group', 'loader_type', 'seq_len', 'dis_type']:
            parser.add_argument(f'--{row["arg_name"]}', type=str, default=str(row['arg_value']).strip())
    nn_type = str(pretrained_args_df[pretrained_args_df['arg_name'] == 'nn_type']['arg_value'].values[0]).strip()
    seq_len = int(str.strip(pretrained_args_df[pretrained_args_df['arg_name'] == 'seq_len']['arg_value'].values[0]))
    with open(os.path.join(get_project_root(), "model_dis_settings.yaml")) as f:
        exp_config = yaml.load(f, Loader=yaml.FullLoader)[nn_type][seq_len]
    for k, v in exp_config.items():
        parser.add_argument(f'--{k}', type=int, default=v)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
