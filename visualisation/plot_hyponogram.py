# -*-coding:utf-8-*-
import glob
import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from concurrent import futures
import numexpr
from sklearn import metrics

def plot_sleep_hyponogram(hypnogram, gt_col_name, pred_col_name, path_to_save, ln_color="black", lw=2,
                          a4_dims=(10, 3), stage_order=["Wake", "NREM", "REM"], save_fig=True, mark_error=True,
                          font_size=16, show=True):
    # Make mask to draw the highlighted stage
    # Draw main hypnogram line, highlighted error, and Artefact/Unscored line
    # get the incorrect prediction mask
    plt.rcParams["font.family"] = "Times New Roman"
    mask = hypnogram[gt_col_name] != hypnogram[pred_col_name]
    fig, ax = plt.subplots(2, 1, squeeze=True, sharex='col', sharey=True, figsize=a4_dims)
    #
    #########################
    # plot hypnogram
    tmp_data = []
    indicator_label = ["(a)", "(b)", "(c)", "(d)"]
    tmp_data.append(hypnogram[gt_col_name].values)
    tmp_data.append(hypnogram[pred_col_name].values)
    f1 = metrics.f1_score(hypnogram[gt_col_name].values, hypnogram[pred_col_name].values, average='macro') * 100.0
    xlabel = np.arange(0, len(hypnogram), 100)
    # ######## plot hypnograms
    for i in [0, 1]:
        ax[i].plot(hypnogram.index, tmp_data[i], color=ln_color, linewidth=lw)
        # Aesthetics
        ax[i].use_sticky_edges = False
        ax[i].grid(True)
        ax[i].margins(x=0, y=1 / len(stage_order) / 2)  # 1/n_epochs/2 gives half-unit margins
        ax[i].set_yticks(range(len(stage_order)))
        ax[i].set_yticklabels(stage_order, fontsize=font_size)
        ax[i].set_ylabel("")
        # set the xticks and tick size
        ax[i].set_xticks(xlabel)
        ax[i].set_xticklabels(xlabel, fontsize=font_size)
        ax[i].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
        # ax[i].spines[["right", "top"]].set_visible(False)
        for spine in ax[i].spines.values():
            spine.set_edgecolor('grey')
        if mark_error and i == 1:
            ax[i].plot(hypnogram.index[mask], tmp_data[i][mask], 'rx', clip_on=False, mew=2, color="red")
        ax[i].xaxis.labelpad = 25
        ax[i].yaxis.labelpad = 25
        ax[i].annotate(indicator_label[i],
                       xy=(0.5, 0.5), xycoords='axes fraction',
                       xytext=(1.05, 0.5), textcoords='axes fraction',
                       fontsize=font_size, ha='left', va='center')
    plt.suptitle("F1 score: {:.2f}%".format(f1), fontsize=font_size)
    # if hyp.start is not None:
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    #     ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # Revert font-size
    if show:
        plt.show()
    # save figure
    if save_fig:
        # save image to svg
        pdf_file = os.path.splitext(path_to_save)[0] + ".pdf"
        fig.savefig(pdf_file, dpi=300, bbox_inches='tight', pad_inches=1, format="pdf")
        fig.savefig(path_to_save, dpi=300, bbox_inches='tight', pad_inches=1)
    plt.close()
    return

def plot_sleep_hyponogram_two_seperate_figs(hypnogram, gt_col_name, pred_col_name, path_to_save, y_hyp_axis_label,
                                            ln_color="black", lw=2, a4_dims=(10, 2),
                                            stage_order=["Wake", "NREM", "REM"], save_fig=True, mark_error=True,
                                            font_size=16, show_f1=True):
    # Make mask to draw the highlighted stage
    # Draw main hypnogram line, highlighted error, and Artefact/Unscored line
    # get the incorrect prediction mask
    plt.rcParams["font.family"] = "Times New Roman"
    mask = hypnogram[gt_col_name] != hypnogram[pred_col_name]
    indicator_label = ["(a)", y_hyp_axis_label]
    #
    #########################
    # plot hypnogram
    tmp_data = []

    tmp_data.append(hypnogram[gt_col_name].values)
    tmp_data.append(hypnogram[pred_col_name].values)
    f1 = metrics.f1_score(hypnogram[gt_col_name].values, hypnogram[pred_col_name].values, average='macro') * 100.0
    xlabel = np.arange(0, len(hypnogram), 200)
    # ######## plot hypnograms
    fig1, ax1 = plt.subplots(1, 1, squeeze=True, figsize=a4_dims)
    fig2, ax2 = plt.subplots(1, 1, squeeze=True, figsize=a4_dims)
    i = 0
    for ax in [ax1, ax2]:
        if i == 0:
            ax.plot(hypnogram.index, hypnogram[gt_col_name].values, color=ln_color, linewidth=lw)
        else:
            ax.plot(hypnogram.index, hypnogram[pred_col_name].values, color=ln_color, linewidth=lw)
        # Aesthetics
        ax.use_sticky_edges = False
        ax.grid(True)
        ax.margins(x=0, y=1 / len(stage_order) / 2)  # 1/n_epochs/2 gives half-unit margins
        # set y limits
        ax.set_ylim([-0.2, len(stage_order) - 0.8])
        ax.set_yticks(range(len(stage_order)))
        ax.set_yticklabels(stage_order, fontsize=font_size)
        ax.set_ylabel("")
        # set the xticks and tick size
        # if i == 1:
        ax.set_xticks(xlabel)
        ax.set_xticklabels(xlabel, fontsize=font_size)
        ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
        # ax[i].spines[["right", "top"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')
        if mark_error and i == 1:
            ax.plot(hypnogram.index[mask], tmp_data[i][mask], 'rx', clip_on=False, mew=2, color="red")

        ax.annotate(indicator_label[i],
                       xy=(0.5, 0.5), xycoords='axes fraction',
                       xytext=(1.05, 0.5), textcoords='axes fraction',
                       fontsize=font_size, ha='left', va='center')
        if i == 1 and show_f1:
            ax.set_title("F1 score: {:.2f}%".format(f1), fontsize=font_size)
        i += 1
    # if hyp.start is not None:
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    #     ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # Revert font-size
    # if show:
    #     plt.show()
    # save figure
    if save_fig:
        # save image to svg
        pdf_gt_file = os.path.splitext(path_to_save)[0] + f"_gt.pdf"
        png_gt_file = os.path.splitext(path_to_save)[0] + f"_gt.png"
        fig1.savefig(pdf_gt_file, dpi=300, bbox_inches='tight', pad_inches=0.1, format="pdf")
        fig1.savefig(png_gt_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
        pdf_gt_file = os.path.splitext(path_to_save)[0] + f"_pred.pdf"
        png_gt_file = os.path.splitext(path_to_save)[0] + f"_pred.png"
        fig2.savefig(pdf_gt_file, dpi=300, bbox_inches='tight', pad_inches=0.1, format="pdf")
        fig2.savefig(png_gt_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return

def plot_sleep_hyponogram_prob(hypnogram, path_to_save, y_prob_axis_label, a4_dims=(10, 2), stage_order=["Wake", "NREM", "REM"],
                               save_fig=True, stage_columns=["0", "1", "2"], font_size=20,
                               palette=["xkcd:sunflower", "xkcd:twilight blue", "#99d7f1"], show=True):
    plt.rcParams["font.family"] = "Times New Roman"
    tmp_df = hypnogram[stage_columns]
    xlabel = np.arange(0, len(hypnogram), 200)
    # ######## Draw the probability distribution using area plot
    fig, ax = plt.subplots(1, 1, figsize=a4_dims)
    tmp_df.plot(kind="area", color=palette[:len(stage_order)], figsize=a4_dims, alpha=0.8, stacked=True, lw=0, ax=ax)
    # Aesthetics
    ax.use_sticky_edges = False
    # ax.grid(True)
    ax.margins(x=0, y=1 / len(stage_order) / 2)  # 1/n_epochs/2 gives half-unit margins
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability", fontsize=font_size)
    # set y tick size only
    ax.tick_params(axis='y', which='major', labelsize=font_size)

    # set the xticks and tick size
    ax.set_xticks(xlabel)
    ax.set_xticklabels(xlabel, fontsize=font_size)
    ax.set_xlabel("Sleep Epoch Index", fontsize=font_size)
    for spine in ax.spines.values():
        spine.set_edgecolor('grey')
    # if hyp.start is not None:
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    #     ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.annotate(y_prob_axis_label,
                xy=(0.5, 0.5), xycoords='axes fraction',
                xytext=(1.05, 0.5), textcoords='axes fraction',
                fontsize=font_size, ha='left', va='center')
    # change legend label to stage name
    handles, labels = ax.get_legend_handles_labels()
    # change the location of the legend to bottom left
    ax.legend(handles, stage_order, loc='lower left', fontsize=font_size-6)
    if show:
        plt.show()
    # save ax to file
    if save_fig:
        # save image to svg
        pdf_file = os.path.splitext(path_to_save)[0] + f".pdf"
        fig.savefig(pdf_file, dpi=300, bbox_inches='tight', pad_inches=0.1, format="pdf")
        fig.savefig(path_to_save, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return

def wrapper_method(output_path, pred_df, model, pid, y_hyp_axis_label, y_prob_axis_label, show_f1=True):
    tmp_df = pred_df[pred_df["pid"] == pid].copy(deep=True).reset_index(drop=True)
    font_size = 26
    pid_output = os.path.join(output_path, model)
    Path(pid_output).mkdir(parents=True, exist_ok=True)
    # plot_sleep_hyponogram(tmp_df, "stages", model, os.path.join(pid_output, f"{pid}_hyponogram.png"), show=False)
    plot_sleep_hyponogram_two_seperate_figs(tmp_df, "stages", model,
                                            os.path.join(pid_output, f"{pid}_{model}_hyponogram.png"),
                                            y_hyp_axis_label, show_f1=show_f1, font_size=font_size)

    plot_sleep_hyponogram_prob(tmp_df, os.path.join(pid_output, f"{pid}_{model}_hyponogram_prob.png"), y_prob_axis_label,
                               stage_columns=["0", "1", "2"], show=False, font_size=font_size)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--win_type', type=str, default="centered_at", help='window type')
    # parser.add_argument('--overwrite', type=int, default=0, help='Over write existing files.')
    parser.add_argument('--parallel', type=int, default=0, help='Parallel Processing')
    # parser.add_argument('--s', type=int, default=5, help='mean svm aggregation window length')
    # parser.add_argument('--d', type=str, default=r'P:\IDEA-FAST\_data\S-8921602b', help='directory of the sleep study folder')

    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    output_path = f"P:\sleep_disentangle_tmp\Disentangle_sleep_plot"
    exp_to_plot = [r"P:\sleep_disentangle_tmp\18scompd253\issmp_dis\tfboard\mesa\GSNMSE3DIS\20220418-150503",
                   r"P:\sleep_disentangle_tmp\18scompd253\issmp_dis\tfboard\mesa\qzd\20220418-155716"]
    y_hyp_axis_label = {
        r"GSNMSE3DIS": "(d)",
        r"qzd": "(b)",
    }
    y_prob_axis_label = {
        r"GSNMSE3DIS": "(e)",
        r"qzd": "(c)",
    }

    for exp_path in exp_to_plot[::-1]:
        model = exp_path.split(os.sep)[-2]
        # load data
        all_csvs = glob.glob(os.path.join(exp_path, "*.csv"))
        pred_data = [x for x in all_csvs if f"3_stages_30s_{model}" in x][0]
        pred_df = pd.read_csv(pred_data)
        pred_df['pred'] = pred_df[model]
        # rename columns
        pred_df = pred_df.dropna()
        pred_df = pred_df.reset_index(drop=True)
        # replace the string with number
        pids = pred_df["pid"].unique()
        pids = [3399]
        if args.parallel == 1:
            with futures.ProcessPoolExecutor(max_workers=numexpr.detect_number_of_cores() - 2) as pool:
                for idx in range(len(pids)):
                    future_result = pool.submit(
                        wrapper_method, output_path=output_path, pred_df=pred_df, model=model, pid=pids[idx],
                        y_hyp_axis_label=y_hyp_axis_label[model], y_prob_axis_label=y_prob_axis_label[model], show_f1=False)
        else:
            for idx in range(len(pids)):
                wrapper_method(output_path=output_path, pred_df=pred_df, model=model, pid=pids[idx],
                               y_hyp_axis_label=y_hyp_axis_label[model], y_prob_axis_label=y_prob_axis_label[model],
                               show_f1=False)


