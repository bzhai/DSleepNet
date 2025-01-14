# -*-coding:utf-8-*-
import numpy as np
import pandas as pd
from utilities.utils import get_all_files_include_sub


root = r"G:\tmp\sleep\opensource\mesa"
mesa_admin_df = pd.read_csv(r"../mesa_admin_df.csv")
hrv_root = r"G:\tmp\sleep\opensource\mesa\Aligned_final"
hrv_files = get_all_files_include_sub(hrv_root, ".csv")
# add leading four zeros to the mesaid
mesa_admin_df['mesaid'] = mesa_admin_df['mesaid'].astype(str).apply(lambda x: x.zfill(4))
unique_pid = mesa_admin_df['mesaid'].unique()
mesa_admin_df['mesaid'] = mesa_admin_df['mesaid'].astype(str)
unique_pid = [str(p) for p in unique_pid]
mesa_intersected_pids_files = []
mesa_intersected_pids = []
mesa_admin_df['file_name'] = ""
for f in hrv_files:
    hrv_pid = f.split("\\")[-1]
    hrv_pid = hrv_pid.split("_")[0]
    if hrv_pid in unique_pid:
        mesa_intersected_pids_files.append(f)
        mesa_intersected_pids.append(hrv_pid)
        mesa_admin_df.loc[mesa_admin_df["mesaid"] == hrv_pid, "file_name"] = f.split("\\")[-1]

mesa_admin_df.to_csv(r"../mesa_admin_df.csv", index=False)