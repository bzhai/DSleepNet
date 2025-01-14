import argparse
import yaml
from data_handler.get_dataloader import build_dataloader
from utilities.utils import *
from sleep_stage_config import Config
import time
from copy import copy
from models.build_model import build_model, get_num_in_channel

import torch
import torch.nn as nn
from torchsummary import summary
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score
import shutil
from utilities.tracker_utils import ClassificationTracker
from scipy.special import softmax
from models.layer4CNN import qzd

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
        total_fold = 1
    else:
        raise ValueError("Dataset is not recognised")
    return total_fold

def main(args):

    cfg = Config()
    total_folds = get_total_folds(args.dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    log_root = os.path.join(r"tfboard", args.dataset, args.nn_type)
    tracker = ClassificationTracker(args, tensorboard_dir=log_root, master_kpi_path="exp_results.csv")
    sys.stdout = Logger(tracker.tensorboard_path)

    print("Model: %s" % args.nn_type)
    print(" Launch TensorBoard with: tensorboard --logdir=%s" % tracker.tensorboard_path)
    print_args(args)
    tracker.copy_main_run_file(os.path.join(os.path.abspath(os.getcwd()), os.path.basename(__file__)))
    tracker.copy_py_files(os.path.join(os.path.abspath(os.getcwd()), "models"), "models")
    tracker.copy_py_files(os.path.join(os.path.abspath(os.getcwd()), "data_handler"), "data_handler")
    test_fold_gt = []  # this list can be used for hold out and CV
    test_fold_pred = []
    test_fold_prob = []
    test_fold_feature = []
    test_fold_idx = []
    for fold_num in np.arange(total_folds):

        if args.nn_type == "qzd":
            model = qzd(in_channels=args.n_feature, num_classes=args.num_classes, fc_dim=7)
        else:
            raise Exception("model name is not found!")
        if torch.cuda.is_available():
            model.cuda()
        optims = {"SGD": torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum),
                  "ADAM": torch.optim.Adam(model.parameters(), lr=args.lr)
                  }
        optimizer = optims[args.optim]
        if args.nn_type == 'qzd':
            summary(model, (args.n_feature, int(args.seq_len)+1))  # print the model summary
        train_loader, val_loader, test_loader = build_dataloader(cfg, args, fold_num)
        print("Total training samples: %s" % train_loader.dataset.__len__())
        print("Total testing samples: %s" % test_loader.dataset.__len__())
        setup_seed(args.seed)
        tracker.reset_best_eval_metrics_models()
        for epoch in range(args.epochs):
            # ***************** Training ************************
            first_train_epoch = True # control the accumulated samples that added to gt and pred to calculate metrics at the last batch
            model.train()
            train_idx_list = []
            for batch_idx, (x, y, _, train_idx) in enumerate(train_loader):
                if batch_idx > 0 and args.debug == 1:
                    continue
                x = x.to(device)
                y = y.to(device)
                # Forward pass
                outputs = model(x)
                train_loss = criterion(outputs, y)
                # Backward and optimize
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                # Calculate the performance
                _, predicted = torch.max(outputs.data, dim=1)
                if first_train_epoch:
                    epoch_gt = copy(y)
                    epoch_pred = copy(predicted)
                else:
                    epoch_gt = torch.cat([epoch_gt, y])
                    epoch_pred = torch.cat([epoch_pred, predicted])

                if batch_idx % args.log_interval == 0:
                    tracker.log_train_fold_epoch(epoch_gt.cpu().numpy(), epoch_pred.cpu().numpy(),
                                                 {'xent': train_loss.item()}, fold_num, len(train_loader), epoch,
                                                 batch_idx)
                first_train_epoch = False
                train_idx_list.extend(train_idx)
            train_idx_csv_dir = os.path.join(tracker.tensorboard_path, "training_id.csv")
            if not os.path.exists(train_idx_csv_dir):
                pd.DataFrame.from_dict({"training_idx": train_idx_list}).to_csv(train_idx_csv_dir)
            # ************** validation ******************
            print("validation start...")
            first_val_epoch = True
            num_val_samples = 0
            total_val_loss = 0
            val_idx_list = []
            val_fc3 = []
            model.eval()
            with torch.no_grad():
                for batch_idx, (x, y, _, val_idx) in enumerate(val_loader):
                    if batch_idx > 0 and args.debug == 1:
                        continue
                    x = x.to(device)
                    y = y.to(device)
                    val_feature = y_val_pred = model(x)
                    if batch_idx < 5:
                        val_fc3.append(val_feature)
                    val_loss = criterion(y_val_pred, y)
                    total_val_loss += val_loss
                    _, y_val_pred = torch.max(y_val_pred.data, dim=1)
                    num_val_samples += y.nelement()
                    if first_val_epoch:
                        val_epoch_gt = copy(y)
                        val_epoch_pred = copy(y_val_pred)
                    else:
                        val_epoch_gt = torch.cat([val_epoch_gt, y])
                        val_epoch_pred = torch.cat([val_epoch_pred, y_val_pred])
                    first_val_epoch = False
                mean_val_loss = total_val_loss / num_val_samples
                val_fc3 = torch.cat(val_fc3, dim=0).cpu()
                tracker.log_eval_fold_epoch(val_epoch_gt.cpu().numpy(), val_epoch_pred.cpu().numpy(),
                                            {'mean_xent': mean_val_loss.cpu().numpy()}, fold_num, epoch, model)
                if args.save_eval == 1:
                    tracker.save_analysis_data(val_epoch_gt.cpu().numpy(), val_epoch_pred.cpu().numpy(),
                                               val_fc3.cpu().numpy(), epoch, run_type='eval', fold_num=fold_num)
            val_idx_list.extend(val_idx)
        val_idx_csv_dir = os.path.join(tracker.tensorboard_path, "val_id.csv")
        if not os.path.exists(val_idx_csv_dir):
            pd.DataFrame.from_dict({"val_idx": val_idx_list}).to_csv(val_idx_csv_dir)
        # ************** test ******************
        # load the best val
        print("testing start...")
        first_test_epoch = True
        num_test_samples = 0
        correct_test = 0
        total_test_loss = 0
        test_fc3_feature = []
        test_idx_epoch_list = []
        # load the best validation model
        model = tracker.load_best_eval_model(model)
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, y, _, test_idx) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                y_test_prob = test_feature = model(x)
                if batch_idx < 5:
                    test_fc3_feature.append(test_feature)
                test_loss = criterion(y_test_prob, y)
                total_test_loss += test_loss
                _, y_test_pred = torch.max(y_test_prob.data, dim=1)
                num_test_samples += y.nelement()
                correct_test += y_test_pred.eq(y.data).sum().item()
                if first_test_epoch:
                    test_epoch_gt = copy(y)
                    test_epoch_pred = copy(y_test_pred)
                    test_epoch_prob = copy(y_test_prob)
                else:
                    test_epoch_gt = torch.cat([test_epoch_gt, y])
                    test_epoch_pred = torch.cat([test_epoch_pred, y_test_pred])
                    test_epoch_prob = torch.cat([test_epoch_prob, y_test_prob])
                first_test_epoch = False
                test_idx_epoch_list.append(test_idx)
            mean_test_loss = total_test_loss / num_test_samples
            test_fc3_feature = torch.cat(test_fc3_feature, dim=0).cpu()
            test_idx_epoch_list = torch.cat(test_idx_epoch_list, dim=0).cpu()
            tracker.log_test_fold_epoch(fold_num, tracker.best_eval_epoch_idx, test_epoch_gt.cpu().numpy(),
                                        test_epoch_pred.cpu().numpy(),
                                        {'mean_xent': mean_test_loss.cpu().numpy()})
            test_fold_feature.append(test_fc3_feature.cpu().numpy())
            test_fold_gt.append(np.expand_dims(test_epoch_gt.cpu().numpy(), axis=1))
            test_fold_pred.append(np.expand_dims(test_epoch_pred.cpu().numpy(), axis=1))
            test_fold_prob.append(test_epoch_prob.cpu().numpy())
            test_fold_idx.append(np.expand_dims(test_idx_epoch_list.cpu().numpy(), axis=1))
    test_fold_idx = np.vstack(test_fold_idx).squeeze()
    test_fold_gt = np.vstack(test_fold_gt).squeeze()
    test_fold_pred = np.vstack(test_fold_pred).squeeze()
    test_fold_feature = np.vstack(test_fold_feature)
    test_fold_prob = softmax(np.vstack(test_fold_prob), axis=1)  # softmax to get the prob
    test_pid = test_fold_idx//10000
    current_df_test = pd.DataFrame({'pid': test_pid, 'stages': test_fold_gt, 'line': test_fold_idx})
    current_df_test = add_sleep_block(current_df_test, id_col_name="pid")
    current_df_test = tracker.save_test_prediction(test_fold_gt, test_fold_pred, test_fold_prob,
                                                   test_fold_idx, df_test=current_df_test)
    tracker.save_analysis_data(test_fold_gt, test_fold_pred,
                               test_fold_feature, tracker.best_eval_epoch_idx, 'test')
    tracker.register_results_to_leaderboard(y_gt=test_fold_gt, y_pred=test_fold_pred, df_test=current_df_test,
                                            summary_folder_dic=cfg.SUMMARY_FOLDER_DICT)
    print("Finished!")
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # specialised parameters
    parser.add_argument('--nn_type', type=str, default="qzd",
                        help='define the neural network type, e.g. qzd')
    # general parameters for all models
    parser.add_argument('--optim', type=str, default="ADAM", help='optimisation')
    # parser.add_argument('--n_feature', type=int, default=9, help='number of features')
    parser.add_argument('--log_interval', type=int, default=100, help='interval to log metrics')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes or labels')
    parser.add_argument('--momentum', type=float, default=0.9, help='opt momentum')
    parser.add_argument('--epochs', type=int, default=1, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--debug', type=int, default=1, help='debug model')
    parser.add_argument('--dataset', type=str, default="mesa", help="apple, mesa")
    parser.add_argument('--seq_len', type=int, default=100, help="100, 50, 20")
    parser.add_argument('--comments', type=str, default="", help="comments to append")
    parser.add_argument('--save_eval', type=int, default=0, help="not save the eval results")
    parser.add_argument('--seed', type=int, default=42, help="fix seed")
    parser.add_argument('--num_train', type=int, default=2002, help="number of training samples")
    parser.add_argument('--dis_type', type=str, nargs='+', default=['bmi5c'],
                        help="disentangle type: ahi4pa5, bmi5c, sleepage5c, ['ahi4pa5', 'bmi5c', 'sleepage5c']")
    parser.add_argument('--train_test_group', nargs='+', default=['group51'],
                        help="training group, ", type=str)
    parser.add_argument('--mask_att', type=str, nargs='+', default=[],
                        help="ablation study purpose to see a single factor effect under a joint disentangle setting"
                             ": ahi4pa5, bmi5c, sleepage5c, ahi_a0h4a")
    parser.add_argument('--loader_type', type=str, default='regroup',  help="data loader type")
    parser.add_argument('--feature_type', type=str, default="all", help="all, hrv, hr, full: full hrv+act and so on")
    args = parser.parse_known_args(argv)[0]
    with open(os.path.join(get_project_root(), "model_dis_settings.yaml")) as f:
        exp_config = yaml.load(f, Loader=yaml.FullLoader)[args.nn_type][args.seq_len]
    for k, v in exp_config.items():       parser.add_argument(f'--{k}', type=int, default=v)
    in_chs = get_num_in_channel(args.dataset, args.feature_type)
    parser.add_argument('--n_feature', type=int, default=in_chs)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
