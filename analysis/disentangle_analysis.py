# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import seaborn as sns
import os

data_dir = rf"G:\tmp\sleep\feature_analysis"
dis_df = pd.read_csv(os.path.join(data_dir, "disentangle_analysis_result_test.csv"))
dis_df["F1_improvement_%"]=dis_df["F1_improvement"]/(dis_df["y_true_f1"]*100-dis_df["F1_improvement"]) *100
dis_df["y_true_y_false_diff_%"] = (dis_df["y_true_f1"] - dis_df["y_false_f1"]) # /dis_df["y_true_f1"]
# dis_df["abs_diff_%"] = (dis_df["d_true_mse"]- dis_df["d_false_mse"]).abs()/np.max(dis_df["d_true_mse"], dis_df["d_false_mse"])


f, ax = plt.subplots(figsize=(6, 4))
# sns.lineplot(x="MMD_y_d", y="F1_improvement_%", markers="*", linestyle="dotted", markersize=18, color="blue",
#              linewidth=3, data=dis_df, ax=ax)
p = sns.regplot(x="y_true_y_false_diff_%", y="F1_improvement_%", color="navy", ci=None,
                data=dis_df, ax=ax)
#
ax.set_ylabel("F1-score improvement (%) \n compared to CNN", labelpad=1, fontsize=16)
ax.set_xlabel("F1-score difference between $D_y(\mathbf{z}_s)$ \n and $D_y(\mathbf{z}_{\\tau})$ on sleep stage classification",  fontsize=16) # labelpad=1,
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),y=p.get_lines()[0].get_ydata())
slope = np.round(slope/100, 2)
ax.annotate(r"$\beta$: {:.2f}".format(slope), xy=(330, 150), xycoords='axes points',
            size=16, ha='right', va='top',
            ) # bbox=dict(boxstyle='round', fc='w')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "disentangle_analysis_plot.pdf"), format="pdf") # , dpi=300
plt.show()

# display the slop in legend using beta symbol
# ax.legend([f"R$^2$={r_value**2:.2f}"], loc="lower right", fontsize=16)
# legend_entries = [r"$\beta$: {:.2f}".format(slope)]
# ax.legend(legend_entries, loc="upper left", markerscale=0,  fancybox=False, fontsize=16)  # title="Legend Title",

# #### The following code uses MMD as the measure of disentanglement which is not correct, because MMD is not a measure of disentanglement
# ax.plot(dis_df["MMD_y_d"], dis_df["F1_improvement_%"], marker="o", linestyle="-", linewidth=3, color="blue",)
import matplotlib.lines as mlines
sns.lineplot(x="MMD_y_d", y="y_true_y_false_diff_%", linestyle="dotted", color="orange", linewidth=3, data=dis_df, ax=ax)
ax.set_ylabel("%", labelpad=1, fontsize=20)
ax.set_xlabel("Maximum Mean Discrepancy between $\mathbf{z}_s$ and $\mathbf{z}_{\\tau}$ ",  fontsize=16) # labelpad=1,
ax.tick_params(axis='both', which='major', labelsize=16)
for x, y in zip(dis_df["MMD_y_d"], dis_df["F1_improvement_%"]):
    ax.plot(x, y, marker="*", markersize=15, color="blue")
for x, y in zip(dis_df["MMD_y_d"], dis_df["y_true_y_false_diff_%"]):
    ax.plot(x, y, marker="^", markersize=15, color="orange")
# add legend to the middle right of the plot
# legend_entries = [("F1 improvement %", "*"), ("IE classification difference %", "^")]
# ax.legend(legend_entries, prop="markers", loc="center right",  fontsize=20)  # title="Legend Title",
star_line = mlines.Line2D([], [], marker="*", markersize=16, linestyle='None', color="blue", label="F1 improvement compared to CNN")
triangle_line = mlines.Line2D([], [], marker="^", markersize=16, linestyle='None', color="orange",
                              label="F1 difference between $D_y(\mathbf{z}_s)$ and $D_y(\mathbf{z}_{\\tau})$")

ax.legend(handles=[star_line, triangle_line],  loc="center right", prop={'size': 14}) #prop="markers",
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "disentangle_analysis_plot.png"), dpi=300)
plt.show()