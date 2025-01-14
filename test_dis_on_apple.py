import argparse
from models.layer4CNN import qzd
from data_handler.get_dataloader import build_dataloader
from models.build_model import get_num_in_channel
from models.gile_4layers_mse import GSNMSE3DIS
from models.gsn_no_ie import GSN_NO_IE
from data_handler.TorchFrameDataLoader import get_windowed_apple_dataset
from utilities.utils import *
from sleep_stage_config import Config
from copy import copy
from models.lstm import LSTM
import torch
import sys
import yaml
import os
from utilities.tracker_utils import ClassificationTracker
from scipy.special import softmax
import glob


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def set_up_optimizers(parameters):
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    return optimizer


def dl_prediction(model, test_loader, tracker, device, nn_type):
    print("testing start...")
    num_test_samples = 0
    test_epoch_gt, test_epoch_pred, test_epoch_prob, test_class_false = [], [], [], []
    test_fold_gt, test_fold_pred, test_fold_prob, test_fold_idx = [], [], [], []
    test_idx_epoch_list = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y, test_idx) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            if nn_type in ["qzd", "lstm"]:
                y_pred = y_prob = model(x)
            elif nn_type in ["GSNMSE3DIS", "GSN_NO_IE"]:
                d_pred, y_pred, d_prob, y_prob, d_false, y_false = model.classifier(x)
            else:
                # get Random Forest prediction in probability, flatten the input
                x = x.view(x.shape[0], -1)
                # convert to tensor
                y_pred = y_prob = torch.from_numpy(model.predict_proba(x))
            _, y_test_pred = torch.max(y_pred.data, dim=1)
            num_test_samples += y.nelement()
            # class prediction
            test_epoch_gt.append(y.cpu().numpy())
            test_epoch_pred.append(y_test_pred.cpu().numpy())
            test_epoch_prob.append(y_prob.cpu().numpy())
            # domain prediction
            # test_class_false.append(y_false.cpu().numpy())
            test_idx_epoch_list.append(test_idx.cpu().numpy())
        test_idx_epoch_list = np.concatenate(test_idx_epoch_list)
        test_epoch_gt = np.concatenate(test_epoch_gt)
        test_epoch_pred = np.concatenate(test_epoch_pred)
        test_epoch_prob = np.concatenate(test_epoch_prob)
        # test_class_false = np.concatenate(test_class_false)
        # tracker.log_test_fold_epoch(fold_num, tracker.best_eval_epoch_idx, test_epoch_gt.cpu().numpy(),
        #                             test_epoch_pred.cpu().numpy(),
        #                             {'mixed_loss': total_test_loss/num_test_samples,
        #                              "class_loss": test_class_y_loss/num_test_samples})
        tracker.log_test_fold_epoch(0, tracker.best_eval_epoch_idx, test_epoch_gt,
                                    test_epoch_pred,
                                    {'mixed_loss': 0,
                                     "class_loss": 0})
        test_fold_gt.append(np.expand_dims(test_epoch_gt, axis=1))
        test_fold_pred.append(np.expand_dims(test_epoch_pred, axis=1))
        test_fold_prob.append(test_epoch_prob)
        test_fold_idx.append(np.expand_dims(test_idx_epoch_list, axis=1))

    test_fold_idx = np.vstack(test_fold_idx).squeeze()
    test_fold_gt = np.vstack(test_fold_gt).squeeze()
    test_fold_pred = np.vstack(test_fold_pred).squeeze()
    test_fold_prob = softmax(np.vstack(test_fold_prob), axis=1)  # softmax over
    test_pid = test_fold_idx // 10000
    return test_fold_gt, test_fold_pred, test_fold_prob, test_fold_idx, test_pid
def main(args):
    cfg = Config()
    exp_cfg = cfg.EXP_DIR_SETTING_FILE
    with open(exp_cfg, 'r') as file:
        config = yaml.safe_load(file)
    # Extract the directory paths from the configuration
    pc_dir_dict = {key: value['root'] for key, value in config.items()}

    exp_id_col_name = "tf"
    pc_col_name = "machine"
    nn_type_col_name = "nn_type"
    # load the experiment results Excel file
    master_df = pd.read_csv(args.exp_file)
    all_exp_ids = args.pretrained_models

    for exp in all_exp_ids:
        print(exp)
        if len(master_df[pc_col_name][master_df[exp_id_col_name] == exp].tolist()) == 0:
            print("No such experiment")
            raise Exception("No such experiment")
        else:
            pc_name = master_df[pc_col_name][master_df[exp_id_col_name] == exp].tolist()[0]
            nn_type = master_df[nn_type_col_name][master_df[exp_id_col_name] == exp].tolist()[0]
            pc_root = pc_dir_dict[pc_name]
            exp_root = os.path.join(pc_root, nn_type, exp,)
            args.nn_type = nn_type
            args.train_test_group = master_df["train_test_group"][master_df[exp_id_col_name] == exp].tolist()[0]

        log_root = os.path.join(get_project_root(), r"tfboard", args.dataset, args.nn_type)
        tracker = ClassificationTracker(args, tensorboard_dir=log_root, master_kpi_path="./exp_results.csv")
        sys.stdout = Logger(tracker.tensorboard_path)

        print("Model: %s" % args.nn_type)
        # print(" Launch TensorBoard with: tensorboard --logdir=%s" % tracker.tensorboard_path)
        print_args(args)
        args.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        all_models = glob.glob(os.path.join(exp_root, "saved_models", "*.pth"))
        if args.nn_type == "GSNMSE3DIS":
            model = GSNMSE3DIS(args)
        elif args.nn_type == "lstm":
            model = LSTM()
        elif args.nn_type == "GSN_NO_IE":
            model = GSN_NO_IE(args)
        elif args.nn_type == "qzd":
            model = qzd(in_channels=7, num_classes=args.num_classes, fc_dim=args.fc_dim)
        else:
            raise Exception("model name is not found!")
        model = tracker.load_best_eval_model(model, all_models[0])

        if torch.cuda.is_available():
            model.cuda()
        args.pretrained = exp

        test_loader = get_windowed_apple_dataset(cfg, args, 256, 3)
        print("Total testing samples: %s" % test_loader.dataset.__len__())

        # ************** test ******************
        # load the best
        test_fold_gt, test_fold_pred, test_fold_prob, test_fold_idx, test_pid = dl_prediction(model, test_loader,
                                                                                              tracker, device, args.nn_type)
        current_df_test = pd.DataFrame({'pid': test_pid, 'stages': test_fold_gt, 'line': test_fold_idx})
        current_df_test = add_sleep_block(current_df_test, "pid")
        current_df_test = tracker.save_test_prediction(test_fold_gt, test_fold_pred, test_fold_prob,
                                                       test_fold_idx, df_test=current_df_test)
        tracker.save_analysis_data(test_fold_gt, test_fold_pred,
                                   None, tracker.best_eval_epoch_idx, 'test')
        tracker.register_results_to_leaderboard(y_gt=test_fold_gt, y_pred=test_fold_pred, df_test=current_df_test,
                                                summary_folder_dic=cfg.SUMMARY_FOLDER_DICT)
    print("Finished!")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn_type', type=str, default='GSNMSE3DIS', help='name of dataset, GILEBase,'
                                                                          'GSNMSE3DIS')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes or labels')
    parser.add_argument('--epochs', type=int, default=2, help='training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--feature_type', type=str, default="all", help="all, hrv, hr, full: full "
                                                                        "hrv+act and so on")
    parser.add_argument('--seq_len', type=int, default=100, help="100, 50, 20")
    parser.add_argument('--dataset', type=str, default='apple', help='name of dataset')
    parser.add_argument('--exp_file', type=str, default=fr"P:\sleep_disentangle_tmp\merged_results\exp_results.csv"
                        , help='the csv file stores all experimental results')
    parser.add_argument('--train_test_group', nargs='+', default=[],
                        help="leave it for blank for apple dataset testing ", type=str)
    parser.add_argument('--pretrained', type=str, default='',  help="leave this empty")
    parser.add_argument('--pretrained_models', type=str, default=["20240122-161314", "20240122-162301"],
                        help="A list of experiment codes e.g., 20220421-011827")
    args = parser.parse_known_args(argv)[0]
    with open(os.path.join(get_project_root(), "model_dis_settings.yaml")) as f:
        exp_config = yaml.load(f, Loader=yaml.FullLoader)[args.nn_type][args.seq_len]
    for k, v in exp_config.items():
        parser.add_argument(f'--{k}', type=int, default=v)
    in_chs = get_num_in_channel(args.dataset, args.feature_type)
    parser.add_argument('--n_feature', type=int, default=in_chs)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
