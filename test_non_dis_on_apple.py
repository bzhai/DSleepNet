"""
Do not use the apple dataset for now, as the AppleWatch dataset, it doesn't have RRI. it's heart rate.
"""
import argparse
from data_handler.TorchFrameDataLoader import get_windowed_apple_dataset
from models.layer4CNN import qzd
from utilities.utils import *
from sleep_stage_config import Config
from copy import copy
import glob
import torch
import sys
import os
from utilities.tracker_utils import ClassificationTracker
from scipy.special import softmax

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


def set_up_optimizers(parameters):
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    return optimizer


def main(args):
    pretrained_models = args.pretrained_models
    for pre_trained in pretrained_models:
        cfg = Config()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        log_root = os.path.join(get_project_root(), r"tfboard", args.dataset, args.nn_type)
        tracker = ClassificationTracker(args, tensorboard_dir=log_root, master_kpi_path="./exp_results.csv")
        sys.stdout = Logger(tracker.tensorboard_path)

        print("Model: %s" % args.nn_type)
        # print(" Launch TensorBoard with: tensorboard --logdir=%s" % tracker.tensorboard_path)
        print_args(args)

        tracker.copy_main_run_file(os.path.join(os.path.abspath(os.getcwd()), os.path.basename(__file__)))
        tracker.copy_py_files(os.path.join(os.path.abspath(os.getcwd()), "models"), "models")
        tracker.copy_py_files(os.path.join(os.path.abspath(os.getcwd()), "data_handler"), "data_handler")

        # for fold_num in np.arange(total_folds):
        if args.nn_type == "qzd":
            model = qzd(in_channels=args.n_feature, num_classes=args.num_classes, fc_dim=7)
        else:
            raise Exception("model name is not found!")
        if torch.cuda.is_available():
            model.cuda()

        args.pretrained = pre_trained
        model_dir = os.path.join(args.model_dir, args.nn_type,
                                 args.pretrained, "saved_models")
        # walk through the model directory and find all pth files
        all_models = glob.glob(os.path.join(model_dir, "*.pth"))
        model = tracker.load_best_eval_model(model, all_models[0])
        model.cuda()

        test_fold_gt = []  # this list can be used for hold out and CV
        test_fold_pred = []
        test_fold_prob = []
        test_fold_feature = []
        test_fold_idx = []
        print("Loading train, val, test dataset...")
        test_loader = get_windowed_apple_dataset(cfg, args, 256, 3)
        print("Total testing samples: %s" % test_loader.dataset.__len__())
        setup_seed(args.seed)
        tracker.reset_best_eval_metrics_models()
        criterion = torch.nn.CrossEntropyLoss()
        # ************** test ******************
        # load the best
        print("testing start...")
        fold_num = 0
        num_test_samples = 0
        first_test_epoch = True
        num_test_samples = 0
        correct_test = 0
        total_test_loss = 0
        test_fc3_feature = []
        test_idx_epoch_list = []
        # load the best validation model
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, y, test_idx) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                y_outputs = model(x)
                if type(y_outputs) in (tuple, list):
                    test_feature, y_test_prob = y_outputs[0], y_outputs[1]
                else:
                    y_test_prob = y_outputs
                    test_feature = y_outputs
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
            if args.save_feature == 1:
                np.savez(os.path.join(tracker.tensorboard_path, "test_savez"),
                         test_idx_epoch_list, test_epoch_gt, test_epoch_pred, test_epoch_prob)

        test_fold_idx = np.vstack(test_fold_idx).squeeze()
        test_fold_gt = np.vstack(test_fold_gt).squeeze()
        test_fold_pred = np.vstack(test_fold_pred).squeeze()
        test_fold_feature = np.vstack(test_fold_feature)
        test_fold_prob = softmax(np.vstack(test_fold_prob), axis=1)  # softmax over
        test_pid = test_fold_idx // 10000
        current_df_test = pd.DataFrame({'pid': test_pid, 'stages': test_fold_gt, 'line': test_fold_idx})
        current_df_test = add_sleep_block(current_df_test, "pid")
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
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of training')

    # parser.add_argument('--fc_dim', type=int, default=448, help='name of dataset')

    parser.add_argument('--n_feature', type=int, default=9, help='number of feature dimension')
    parser.add_argument('--save_feature', type=int, default=0, help='save features')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes or labels')
    parser.add_argument('--seed', type=int,  default=42, help="fix seed")
    parser.add_argument('--dataset', type=str, default='apple', help='name of dataset')
    parser.add_argument('--model_dir', type=str, default=r"P:\sleep_disentangle_tmp\Ultron\tfboard\mesa", help="saved model directory")
    parser.add_argument('--pretrained_models', type=str, default=["20220421-025541"], help="a tf code to load pretrained model e.g., 20220421-011827")
    parser.add_argument('--pretrained', type=str, default='',  help="Please leave it blank")
    parser.add_argument('--nn_type', type=str, default='qzd', help='name of model, GSNMSE3DIS')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
