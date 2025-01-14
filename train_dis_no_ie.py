import argparse

from data_handler.get_dataloader import build_dataloader
from models.build_model import get_num_in_channel
from models.gsn_no_ie import GSN_NO_IE
from utilities.utils import *
from sleep_stage_config import Config
from copy import copy

import torch
import sys
import yaml
import os
from easydict import EasyDict as edict
from utilities.tracker_utils import ClassificationTracker
from scipy.special import softmax

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def set_up_optimizers(parameters):
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    return optimizer

def main(args):
    # torch.backends.cudnn.enabled = False

    cfg = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    log_root = os.path.join(r"tfboard", args.dataset, args.nn_type)
    tracker = ClassificationTracker(args, tensorboard_dir=log_root, master_kpi_path="./exp_results.csv")
    sys.stdout = Logger(tracker.tensorboard_path)

    print("Model: %s" % args.nn_type)
    # print(" Launch TensorBoard with: tensorboard --logdir=%s" % tracker.tensorboard_path)
    print_args(args)

    tracker.copy_main_run_file(os.path.join(os.path.abspath(os.getcwd()), os.path.basename(__file__)))
    tracker.copy_py_files(os.path.join(os.path.abspath(os.getcwd()), "models"), "models")
    tracker.copy_py_files(os.path.join(os.path.abspath(os.getcwd()), "data_handler"), "data_handler")
    test_fold_gt = []  # this list can be used for hold out and CV
    test_fold_pred = []
    test_fold_prob = []
    test_fold_idx = []
    total_folds = 1
    for fold_num in np.arange(total_folds):
        if args.nn_type == "GSN_NO_IE":
            model = GSN_NO_IE(args)
        else:
            raise Exception("model name is not found!")
        if torch.cuda.is_available():
            model.cuda()
        optimizer = set_up_optimizers(model.parameters())

        # summary(model, (get_num_in_channel(args.dataset), int(args.seq_len)+1))  # print the model summary
        print("Loading train, val, test dataset...")
        train_loader, val_loader, test_loader = build_dataloader(cfg, args, fold_num)
        print("Total training samples: %s" % train_loader.dataset.__len__())
        print("Total testing samples: %s" % test_loader.dataset.__len__())
        setup_seed(args.seed)
        tracker.reset_best_eval_metrics_models()
        for epoch in range(args.epochs):
            # ***************** Training ************************
            # control the accumulated samples that added to gt and pred to calculate metrics at the last batch
            first_train_epoch = True
            train_loss = 0
            train_class_y_loss = 0
            total_train = 0
            train_pa_loss = 0
            train_idx_list = []
            for batch_idx, (x, y, d, train_idx) in enumerate(train_loader):
                model.train()

                if batch_idx > 0 and args.debug == 1:
                    continue
                x, y, d = x.to(device), y.to(device), d.to(device)
                optimizer.zero_grad()
                # Forward pass
                loss_origin, class_y_loss, y_prob, MSE_d = model.loss_function(d, x, y)
                loss_origin.backward()
                optimizer.step()
                train_loss += loss_origin
                train_class_y_loss += class_y_loss
                train_pa_loss += MSE_d
                total_train += y.size(0)
                # Calculate the performance
                d_pred, y_pred, d_prob, y_prob, d_false, y_false = model.classifier(x)
                _, predicted = torch.max(y_prob.data, dim=1)
                if first_train_epoch:
                    epoch_gt = copy(y)
                    epoch_pred = copy(predicted)
                else:
                    epoch_gt = torch.cat([epoch_gt, y])
                    epoch_pred = torch.cat([epoch_pred, predicted])

                if batch_idx % args.log_interval == 0:
                    tracker.log_train_fold_epoch(epoch_gt.cpu().numpy(), epoch_pred.cpu().numpy(),
                                                 {'mixed_loss': train_loss/total_train,
                                                  "class_loss": train_class_y_loss/total_train,
                                                  "pa_loss": train_pa_loss/total_train}, fold_num,
                                                 len(train_loader), epoch, batch_idx)

                first_train_epoch = False
                train_idx_list.extend(train_idx.cpu().numpy().tolist())
            train_idx_csv_dir = os.path.join(tracker.tensorboard_path, "training_id.csv")
            if not os.path.exists(train_idx_csv_dir):
                pd.DataFrame.from_dict({"training_idx": train_idx_list}).to_csv(train_idx_csv_dir)
            # ************** validation ******************
            print("validating start...")
            first_val_epoch = True
            num_val_samples = 0
            total_val_loss = 0
            val_class_y_loss = 0
            val_pa_loss = 0
            val_idx_list = []
            model.eval()
            with torch.no_grad():
                for batch_idx, (x, y, d, val_idx) in enumerate(val_loader):
                    if batch_idx > 0 and args.debug == 1:
                        continue
                    x, y, d = x.to(device), y.to(device), d.to(device)
                    loss_origin, class_y_loss, _, MSE_d = model.loss_function(d, x, y)
                    total_val_loss += loss_origin
                    val_class_y_loss += class_y_loss
                    val_pa_loss += MSE_d
                    d_pred, y_pred, d_prob, y_prob, d_false, y_false = model.classifier(x)
                    _, y_val_pred = torch.max(y_prob.data, dim=1)
                    num_val_samples += y.nelement()
                    if first_val_epoch:
                        val_epoch_gt = copy(y)
                        val_epoch_pred = copy(y_val_pred)
                    else:
                        val_epoch_gt = torch.cat([val_epoch_gt, y])
                        val_epoch_pred = torch.cat([val_epoch_pred, y_val_pred])
                    first_val_epoch = False
                # val_fc3 = torch.cat(val_fc3, dim=0).cpu()
                tracker.log_eval_fold_epoch(val_epoch_gt.cpu().numpy(), val_epoch_pred.cpu().numpy(),
                                            {'mixed_loss': total_val_loss/num_val_samples,
                                             "class_loss": val_class_y_loss/num_val_samples,
                                             "pa_loss": val_pa_loss / num_val_samples
                                             }, fold_num, epoch, model)
                if args.save_eval == 1:
                    tracker.save_analysis_data(val_epoch_gt.cpu().numpy(), val_epoch_pred.cpu().numpy(),
                                               None, epoch, 'eval', fold_num=fold_num)
            val_idx_list.extend(val_idx)
        val_idx_csv_dir = os.path.join(tracker.tensorboard_path, "val_id.csv")
        if not os.path.exists(val_idx_csv_dir):
            pd.DataFrame.from_dict({"val_idx": val_idx_list}).to_csv(val_idx_csv_dir)
        # ************** test ******************
        # load the best
        print("testing start...")
        num_test_samples = 0
        test_epoch_gt, test_epoch_pred, test_epoch_prob, test_class_false = [], [], [], []
        test_domain, test_domain_prob, test_domain_gt, test_domain_false = [], [], [], []
        test_idx_epoch_list = []
        # load the best validation model
        model = tracker.load_best_eval_model(model)
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, y, d, test_idx) in enumerate(test_loader):
                x, y, d = x.to(device), y.to(device), d.to(device)
                # loss_origin, class_y_loss, y_prob = model.loss_function(d, x, y)
                # total_test_loss += loss_origin
                # test_class_y_loss += class_y_loss
                d_pred, y_pred, d_prob, y_prob, d_false, y_false = model.classifier(x)
                _, y_test_pred = torch.max(y_pred.data, dim=1)
                num_test_samples += y.nelement()
                # class prediction
                test_epoch_gt.append(y.cpu().numpy())
                test_epoch_pred.append(y_test_pred.cpu().numpy())
                test_epoch_prob.append(y_prob.cpu().numpy())
                # domain prediction
                test_domain_gt.append(d.cpu().numpy())
                test_domain.append(d_pred.cpu().numpy())
                test_domain_prob.append(d_prob.cpu().numpy())
                test_domain_false.append(d_false.cpu().numpy())
                test_class_false.append(y_false.cpu().numpy())
                test_idx_epoch_list.append(test_idx.cpu().numpy())
            test_idx_epoch_list = np.concatenate(test_idx_epoch_list)
            test_epoch_gt = np.concatenate(test_epoch_gt)
            test_epoch_pred = np.concatenate(test_epoch_pred)
            test_epoch_prob = np.concatenate(test_epoch_prob)
            test_domain_gt = np.concatenate(test_domain_gt)
            test_domain = np.concatenate(test_domain)
            test_domain_prob = np.concatenate(test_domain_prob)
            test_domain_false = np.concatenate(test_domain_false)
            test_class_false = np.concatenate(test_class_false)
            # tracker.log_test_fold_epoch(fold_num, tracker.best_eval_epoch_idx, test_epoch_gt.cpu().numpy(),
            #                             test_epoch_pred.cpu().numpy(),
            #                             {'mixed_loss': total_test_loss/num_test_samples,
            #                              "class_loss": test_class_y_loss/num_test_samples})
            tracker.log_test_fold_epoch(fold_num, tracker.best_eval_epoch_idx, test_epoch_gt,
                                        test_epoch_pred,
                                        {'mixed_loss': 0,
                                         "class_loss": 0})
            test_fold_gt.append(np.expand_dims(test_epoch_gt, axis=1))
            test_fold_pred.append(np.expand_dims(test_epoch_pred, axis=1))
            test_fold_prob.append(test_epoch_prob)
            test_fold_idx.append(np.expand_dims(test_idx_epoch_list, axis=1))
            if args.save_feature == 1:
                np.savez(os.path.join(tracker.tensorboard_path, "test_savez"),
                         test_idx_epoch_list, test_epoch_gt, test_epoch_pred, test_epoch_prob, test_domain_gt,
                         test_domain, test_domain_prob, test_domain_false, test_class_false)
    test_fold_idx = np.vstack(test_fold_idx).squeeze()
    test_fold_gt = np.vstack(test_fold_gt).squeeze()
    test_fold_pred = np.vstack(test_fold_pred).squeeze()
    test_fold_prob = softmax(np.vstack(test_fold_prob), axis=1)  # softmax over
    test_pid = test_fold_idx//10000
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
    # specialised parameters
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size of training')
    parser.add_argument('--dataset', type=str, default='mesa', help='name of dataset')
    parser.add_argument('--nn_type', type=str, default='GSN_NO_IE', help='name of dataset, GILEBase, GSNMSE3DIS')
    # parser.add_argument('--fc_dim', type=int, default=448, help='name of dataset')

    # parser.add_argument('--n_feature', type=int, default=9, help='name of feature dimension')
    parser.add_argument('--n_class', type=int, default=3, help='number of class')
    parser.add_argument('--d_AE', type=int, default=50, help='dim of AE')
    parser.add_argument('--sigma', type=float, default=0, help='parameter of mmd')
    parser.add_argument('--weight_mmd', type=float, default=0, help='weight of mmd loss')
    parser.add_argument('--save_feature', type=int, default=0, help='save features')
    parser.add_argument('--target_domain', type=str, default='S1', help='the target domain, [S1, S2, S3, S4]')
    parser.add_argument('--n_domains', type=int, default=1, help='number of total domains actually')
    # parser.add_argument('--n_target_domains', type=int, default=1, help='number of target domains')
    parser.add_argument('--beta', type=float, default=1., help='multiplier for KL')
    parser.add_argument('--x-dim', type=int, default=1152, help='input size after flattening')
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=1, help='multiplier for y classifier')
    parser.add_argument('--aux_loss_multiplier_d', type=float, default=1e-5, help='multiplier for d classifier')
    parser.add_argument('--beta_d', type=float, default=1e-5, help='multiplier for KL d')
    parser.add_argument('--beta_x', type=float, default=0., help='multiplier for KL x')
    parser.add_argument('--beta_y', type=float, default=1e-3, help='multiplier for KL y')
    parser.add_argument('--num_train', type=int, default=2002, help='number of training subjects')
    parser.add_argument('--weight_true', type=float, default=0, help='weights for classifier true')
    parser.add_argument('--weight_false', type=float, default=0, help='weights for classifier false')
    parser.add_argument('--log_interval', type=int, default=100, help='interval to log metrics')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes or labels')
    parser.add_argument('--momentum', type=float, default=0.9, help='opt momentum')
    parser.add_argument('--epochs', type=int, default=2, help='training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--debug', type=int, default=0, help='debug model')
    parser.add_argument('--feature_type', type=str, default="full", help="all, hrv, hr, and so on")
    parser.add_argument('--seq_len', type=int, default=100, help="100, 50, 20")
    parser.add_argument('--comments', type=str, default="", help="comments to append")
    parser.add_argument('--save_eval', type=int, default=0, help="not save the eval results")
    parser.add_argument('--seed', type=int,  default=42, help="fix seed")
    parser.add_argument('--dis_type', type=str, nargs='+', default=['bmi5c'],
                        help="disentangle type: ahi4pa5, bmi5c, sleepage5c")
    parser.add_argument('--train_test_group', nargs='+', default=['group51'],
                        help="training group, ", type=str)
    parser.add_argument('--mask_att', type=str, nargs='+', default=[],
                        help="abbliation study purpose to see a single factor effect: ahi4pa5, bmi5c, sleepage5c, ahi_a0h4a")
    parser.add_argument('--loader_type', type=str, default='regroup',  help="data loader type")
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
