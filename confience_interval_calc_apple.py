# -*-coding:utf-8-*-
import multiprocessing
import pandas as pd
from confience_interval_calc_mesa import update_metrics_with_ci


if __name__ == "__main__":
    pc_UbuntuVM1_root = r"P:\sleep_disentangle_tmp\Ultron\tfboard\mesa"
    pc_dir_dict = {'UbuntuVM1': pc_UbuntuVM1_root}
    # Load the jbhi latext table excel file
    jbhi_df = pd.read_excel(fr"P:\sleep_disentangle_tmp\merged_results\JBHI\Final Table MF1 Cohen Acc (major revision).xlsx",
                            sheet_name="OriginalTable_100_8hrv_transfer", header=0)
    # load the experiment results Excel file
    master_df = pd.read_csv(fr"P:\sleep_disentangle_tmp\merged_results\exp_results.csv")
    save_to_file = fr"c:\tmp\jbhi_with_cis_apple.xlsx"
    # filter_exp = ["20240122-233411",
    #               # "20240122-233425", "20240122-233428", "20240122-233430", "20240122-233433",
    #               # "20240122-233436", "20240122-233439"
    #               ]
    # jbhi_df = jbhi_df[jbhi_df["tf"].isin(filter_exp)]
    update_metrics_with_ci(jbhi_df, master_df, pc_dir_dict, save_to_file)
