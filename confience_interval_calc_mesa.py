# -*-coding:utf-8-*-
import multiprocessing

import pandas as pd
import numpy as np
import os
from confidenceinterval import roc_auc_score
from confidenceinterval import precision_score, recall_score, f1_score, accuracy_score
import scipy
import sklearn
import random
from confidenceinterval.bootstrap import bootstrap_ci
from joblib import Parallel, delayed



def calculate_ci_cohen_kappa(y_pred, y_true):
    """
    Calculate the confidence interval for Cohen's Kappa
    :param y_pred: the predicted labels
    :param y_true: the ground truth labels
    :return: the confidence interval of Cohen's Kappa
    """
    # Calculate the confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    n = np.sum(cm)
    n_classes = cm.shape[0]
    agreement = 0
    for i in range(n_classes):
        agreement += cm[i, i]
    # agreement due to chance
    judge1_totals = np.sum(cm, axis=0)
    judge2_totals = np.sum(cm, axis=1)
    judge1_totals_prop = np.sum(cm, axis=0) / n
    judge2_totals_prop = np.sum(cm, axis=1) / n
    by_chance = np.sum(judge1_totals_prop * judge2_totals_prop * n)
    # Calculate the kappa using non-weighted Cohen's Kappa formula
    # kappa = (agreement - by_chance) / (n - by_chance)
    # calculate the kappa using the sklearn.metrics.cohen_kappa_score
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")
    # Calculate the confidence interval
    sum0 = np.sum(cm, axis=0)
    sum1 = np.sum(cm, axis=1)
    expected = np.outer(sum0, sum1) / n
    n_classes = cm.shape[0]
    identity = np.identity(n_classes)
    p_o = np.sum((identity * cm) / n)
    p_e = np.sum((identity * expected) / n)
    # Calculate a
    ones = np.ones([n_classes, n_classes])
    row_sums = np.inner(cm, ones)
    col_sums = np.inner(cm.T, ones).T
    sums = row_sums + col_sums
    a_mat = cm / n * (1 - sums / n * (1 - kappa)) ** 2
    identity = np.identity(n_classes)
    a = np.sum(identity * a_mat)
    # Calculate b
    b_mat = cm / n * (sums / n) ** 2
    b_mat = b_mat * (ones - identity)
    b = (1 - kappa) ** 2 * np.sum(b_mat)
    # Calculate c
    c = (kappa - p_e * (1 - kappa)) ** 2
    # Standard error
    se = np.sqrt((a + b - c) / n) / (1 - p_e)
    # Two-tailed statistical test
    alpha = 0.05
    z_crit = scipy.stats.norm.ppf(1 - alpha / 2)
    ci = se * z_crit * 2
    lower = kappa - se * z_crit
    upper = kappa + se * z_crit
    return kappa, (lower, upper)


def bootstrap_cqk(y_true, y_pred):
    num_resamples = 100
    Y = np.array([y_true, y_pred]).T
    weighted_kappas = []
    for i in range(num_resamples):
        Y_resample = np.array(random.choices(Y, k=len(Y)))
        y_true_resample = Y_resample[:, 0]
        y_pred_resample = Y_resample[:, 1]
        weighted_kappas.append(sklearn.metrics.cohen_kappa_score(y_true_resample, y_pred_resample, weights="quadratic"))
    sek = np.std(weighted_kappas)
    alpha = 0.95
    margin = (1 - alpha) / 2  # two-tailed test
    x = scipy.stats.norm.ppf(1 - margin)
    return np.mean(weighted_kappas), np.mean(weighted_kappas) - x * sek, np.mean(weighted_kappas) + x * sek

def bootstrap_cqk_parallel(y_true, y_pred):
    num_resamples = 100
    Y = np.array([y_true, y_pred]).T
    weighted_kappas = []
    # make the resamples for loop parallel
    def __calc_resample(Y):
        Y_resample = np.array(random.choices(Y, k=len(Y)))
        y_true_resample = Y_resample[:, 0]
        y_pred_resample = Y_resample[:, 1]
        return sklearn.metrics.cohen_kappa_score(y_true_resample, y_pred_resample, weights="quadratic")
    num_cores = multiprocessing.cpu_count()
    weighted_kappas = Parallel(n_jobs=num_cores)(delayed(__calc_resample)(Y) for i in range(num_resamples))
    sek = np.std(weighted_kappas)
    alpha = 0.95
    margin = (1 - alpha) / 2  # two-tailed test
    x = scipy.stats.norm.ppf(1 - margin)
    return np.mean(weighted_kappas), np.mean(weighted_kappas) - x * sek, np.mean(weighted_kappas) + x * sek

def update_metrics_with_ci(latex_df, master_df, pc_dir_dict, save_to_file, exp_id_col_name='tf', gt_col_name="stages",
                           nn_type_col_name='nn_type', pc_col_name='machine', exp_root_col_name='exp_root',
                           metrics=["Accuracy", "Precision", "Recall", "MF1", "Cohen"], avg_method = "macro",
                           exp_pred_file=r"3_stages_30s_%s_100_all.csv"):
    """
    Update the metrics in the latex_df with confidence intervals
    some example data for latex_df:
    Model Name	MF1	Cohen	Accuracy	Wake	REM sleep	Non-REM sleep	tf
    CNN	25.4	-12.0	47.7	-50.5 $\pm$ 8.8	-6.5 $\pm$ 5.2	56.9 $\pm$ 9.2	20220420-013800
    DisSleepNet	33.1	11.8	45.7	-41.7 $\pm$ 8.9	-10.8 $\pm$ 4.8	52.5 $\pm$ 9.2	20220420-065804
    some expample data for master_df:
    macro_specificity	macro_recall	macro_precision	macro_f1	macro_cohen	macro_accuracy	machine	tf
    80.61314771	56.14935332	60.86974534	56.98652397	44.45027455	68.41552734	18scompd253	20220418-131013
    80.85485371	56.18970333	61.59242763	57.37753468	45.95561161	69.24316406	18scompd253	20220418-132548
    81.12985823	54.88431068	62.31074165	56.24950964	47.07222696	70.41748047	18scompd253	20220418-145922

    @param latex_df:  the latex table dataframe for generate the latex table
    @param master_df:  the master dataframe from the experiment results with exp_id, pc, nn_type, exp_root
    @param pc_dir_dict:  the dictionary of experiment root directory for each pc, tensorboard root directory
    @param save_to_file: the file to save the updated latex table
    @param exp_id_col_name:  the column name of the experiment id
    @param gt_col_name:  the column name of the ground truth
    @param nn_type_col_name:  the column name of the neural network type
    @param pc_col_name:  the column name of the pc name
    @param exp_root_col_name:  the column name of the experiment root directory
    @param metrics:  the list of metrics to be calculated, e.g., ["Accuracy", "Precision", "Recall", "MF1", "Cohen"]
    @param avg_method: the average method for calculating the metrics, e.g., "macro", "weighted"
    @param exp_pred_file: the prediction file name for each experiment, e.g., "3_stages_30s_%s_100_all.csv"
    @return:
    """
    all_exp_ids = latex_df[exp_id_col_name].tolist()
    for metric in metrics:
        if metric not in latex_df.columns:
            latex_df[metric] = np.nan
    # exp = "20220420-164554"
    for exp in all_exp_ids:
        print(exp)
        if len(master_df[pc_col_name][master_df[exp_id_col_name] == exp].tolist()) == 0:
            print("No such experiment")
            raise Exception("No such experiment")
        else:
            pc_name = master_df[pc_col_name][master_df[exp_id_col_name] == exp].tolist()[0]
            nn_type = master_df[nn_type_col_name][master_df[exp_id_col_name] == exp].tolist()[0]
            pc_root = pc_dir_dict[pc_name]

            exp_root = os.path.join(pc_root, nn_type, exp, exp_pred_file % nn_type)
            if os.path.exists(exp_root):
                exp_df = pd.read_csv(exp_root)
                # make a dictionary of the metrics
                metrics_dict = {}
                for metric in metrics:
                    if metric == "Cohen":
                        # metrics_dict[metric] = bootstrap_cqk(exp_df[gt_col_name].values, exp_df[nn_type].values)
                        # metrics_dict[metric] = bootstrap_cqk_parallel(exp_df[gt_col_name].values, exp_df[nn_type].values)
                        metrics_dict[metric] = calculate_ci_cohen_kappa(exp_df[gt_col_name].values, exp_df[nn_type].values)

                    elif metric == "Recall":
                        metrics_dict[metric] = recall_score(exp_df[gt_col_name], exp_df[nn_type], confidence_level=0.95,
                                                            average=avg_method,  method="bootstrap_basic",
                                                            n_resamples=999)
                    elif metric == "MF1":
                        metrics_dict[metric] = f1_score(exp_df[gt_col_name], exp_df[nn_type], confidence_level=0.95, average=avg_method,
                                      n_resamples=999)
                    elif metric == "Accuracy":
                        metrics_dict[metric] = accuracy_score(exp_df[gt_col_name], exp_df[nn_type], confidence_level=0.95,
                                          n_resamples=999)
                    elif metric == "Precision":
                        metrics_dict[metric] = precision_score(exp_df[gt_col_name], exp_df[nn_type],
                                                               confidence_level=0.95, average=avg_method, n_resamples=999)
                # update the latex_df
                for metric in metrics:
                    # latex_df.loc[latex_df['tf'] == exp, metric] = (f"\makecell[c]{{{metrics_dict[metric][0] * 100:.1f} "
                    #                                                f"\\\\ {{[{metrics_dict[metric][1][0] * 100:.1f}, "
                    #                                                f"{metrics_dict[metric][1][1] * 100:.1f}]}}}}")
                    latex_df.loc[latex_df['tf'] == exp, metric] = (f"{metrics_dict[metric][0] * 100:.1f} [{metrics_dict[metric][1][0] * 100:.1f}, {metrics_dict[metric][1][1] * 100:.1f}]")
            else:
                print("Cannot find the prediction file for %s" % exp_root)
                raise Exception("Cannot find the prediction file for %s" % exp_root)
    latex_df.to_excel(save_to_file, sheet_name="OriginalTable_100_8hrv", index=False)


if __name__ == "__main__":
    pc_UbuntuVM1_root = r"P:\sleep_disentangle_tmp\Ultron\tfboard\mesa"
    pc_dir_dict = {'UbuntuVM1': pc_UbuntuVM1_root}
    # Load the jbhi latext table excel file
    jbhi_df = pd.read_excel(fr"P:\sleep_disentangle_tmp\JBHI\Final Table MF1 Cohen Acc (major revision).xlsx",
                            sheet_name="OriginalTable_100_8hrv", header=0)
    # load the experiment results Excel file
    master_df = pd.read_csv(fr"P:\sleep_disentangle_tmp\merged_results\exp_results.csv")
    save_to_file = fr"c:\tmp\jbhi_with_cis.xlsx"
    update_metrics_with_ci(jbhi_df, master_df, pc_dir_dict, save_to_file)
