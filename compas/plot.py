# %%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import itertools

# %%
# Flags
experiment_name = "base"
show_f=False
save=True
show=False

# Read data
experiment_folder = "./.cache/" + experiment_name + "/"
df = pd.concat([
    pd.read_pickle(experiment_folder + f) for f in os.listdir(experiment_folder)
], ignore_index=True)
df['norm of avg grad'] = df['average gradient'].apply(np.linalg.norm, ord=np.inf)

# Create a color scale for the lr
lr_cmap_log = plt.colormaps["Oranges"]
lr_cmap = LinearSegmentedColormap.from_list("Custom", lr_cmap_log(np.logspace(-0.5, 1, 100, base=10)))

os.makedirs('./plots/' + experiment_name, exist_ok=True)

sns.set_style("white")
sns.set_context("poster")
sns.set_palette("pastel")

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30,5), sharex=True, sharey=False)

# First plot
sns.lineplot(ax=axs[0], data=df[df.lr == 0], x="race count", y="y", color="#888888", estimator=None, n_boot=0)
_lp = sns.lineplot(ax=axs[0], data=df, x="race count", y="yhat", hue="lr", palette=lr_cmap, estimator=None, n_boot=0)
if show_f:
    axs[0].plot(np.arange(len(df)), df.f, color="#880000")
_lp.get_legend().remove()

# Second plot
_lp = sns.lineplot(ax=axs[1], data=df, x="race count", y="norm of avg grad", hue="lr", palette=lr_cmap, estimator=None, n_boot=0)
_lp.get_legend().remove()

# Third plot
_lp = sns.lineplot(ax=axs[2], data=df, x="race count", y="average loss", hue="lr", palette=lr_cmap, estimator=None, n_boot=0)
_lp.get_legend().set_loc('upper right')

for tick in axs[0].get_xticklabels() + axs[1].get_xticklabels() + axs[2].get_xticklabels():
    tick.set_rotation(45)  # adjust the rotation angle as needed
sns.despine(top=True, right=True)
for ax in axs: 
    ax.set_xlabel("# individuals")

ylims_avg_grad_1 = axs[1].get_ylim()

plt.tight_layout()

if save:
    plt.savefig('./plots/' + experiment_name + "/" + "series.pdf")
if show:
    plt.show()
# %%
print(df.columns)
# %%
# Next, look at plots of the bias over sensitive categories
categorical_cols = ["race"]
uniques = [ "African-American", "Caucasian", "Hispanic" ]
df = df.dropna()

# %%
# Create a subplot per sensitive category, and plot the bias in each
dfs_to_plot = {
    "lr=0": df[df.lr == 0],
    f"lr={0.05}": df[df.lr == 0.05],
}

fig, axs = plt.subplots(nrows=1, ncols=len(dfs_to_plot), figsize=(10*len(dfs_to_plot), 5*len(categorical_cols)), sharey=True, sharex=True)

for i in range(len(dfs_to_plot)):
    _df = list(dfs_to_plot.values())[i]
    for j in range(len(uniques)): 
        _df_subset = _df[_df.race == uniques[j]]
        gradients = np.array(_df_subset.gradient.to_list())
        time = np.arange(len(gradients))+1
        average_gradient = gradients.cumsum(axis=0)/time[:,None]
        _lp = sns.lineplot(ax=axs[i], x=time, y=average_gradient[:,j], label=uniques[j], estimator=None, n_boot=0)
        axs[i].set_xlabel("# individuals")
        for tick in axs[i].get_xticklabels():
            tick.set_rotation(45)  # adjust the rotation angle as needed
axs[0].set_ylabel("Bias")
axs[0].set_title("Raw COMPAS Model")
axs[1].set_title("Caliboosted COMPAS Model")
# Horizontal line at zero
axs[0].axhline(y=0, color='#888888', linestyle='--')
axs[1].axhline(y=0, color='#888888', linestyle='--')
# remove legend from axs[0]
axs[0].get_legend().remove()
# Move axs[1] legend to top right
axs[1].get_legend().set_loc('upper right')
plt.ylim([-0.5,0.5])
sns.despine(top=True, right=True)
plt.tight_layout()
if save:
    plt.savefig('./plots/' + experiment_name + "/" + "stratified.pdf")
if show:
    plt.show()
# %%

# %%

# %%
