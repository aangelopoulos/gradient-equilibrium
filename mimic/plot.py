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
experiment_name = "gradient_boosting_ethnicity_marital"
show_f=False
show_viscosity=False
save=True
show=True

# Read data
experiment_folder = "./.cache/" + experiment_name + "/"
df = pd.concat([
    pd.read_pickle(experiment_folder + f) for f in os.listdir(experiment_folder)
], ignore_index=True)
df['norm of avg grad'] = df['average gradient'].apply(np.linalg.norm, ord=np.inf)
# Find index of first nonzero gradient
df = df[df.admittime > df[df['norm of avg grad'] != 0].admittime.min()]

if not show_viscosity:
    df = df[df.viscosity==0]

# Create a color scale for the lr
lr_cmap_log = plt.colormaps["Oranges"]
lr_cmap = LinearSegmentedColormap.from_list("Custom", lr_cmap_log(np.logspace(-0.5, 1, 100, base=10)))

os.makedirs('./plots/' + experiment_name, exist_ok=True)

sns.set_style("white")
sns.set_context("poster")
sns.set_palette("pastel")

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30,5), sharex=True, sharey=False)

# First, plot the ys, fs, and yhats on the same plot, to visually check the predictions.
sns.lineplot(ax=axs[0], data=df[(df.lr == 0) & (df.viscosity == 0)], x="admittime", y="y", color="#888888")
_lp = sns.lineplot(ax=axs[0], data=df, x="admittime", y="yhat", hue="lr", palette=lr_cmap)
if show_f:
    axs[0].plot(df.f, color="#880000")
_lp.get_legend().remove()

# Next, plot the average gradient over time
_lp = sns.lineplot(ax=axs[1], data=df, x="admittime", y="norm of avg grad", hue="lr", palette=lr_cmap)
_lp.get_legend().remove()
# Next, plot the average loss over time
_lp = sns.lineplot(ax=axs[2], data=df, x="admittime", y="average loss", hue="lr", palette=lr_cmap)
_lp.get_legend().set_loc('upper right')

for tick in axs[0].get_xticklabels() + axs[1].get_xticklabels() + axs[2].get_xticklabels():
    tick.set_rotation(45)  # adjust the rotation angle as needed
sns.despine(top=True, right=True)
for ax in axs: 
    ax.set_xlabel("patient admission date")

plt.tight_layout()

if save:
    plt.savefig('./plots/' + experiment_name + "/" + "series.pdf")
if show:
    plt.show()

# %%
# Next, look at plots of the bias over sensitive categories
categorical_cols = [col for col in df.columns if col in ["ethnicity", "marital_status",  "insurance", "language"]]

# Drop nan
for col in categorical_cols:
    df = df[~df[col].isna()]
uniques = [ df[col].unique() for col in categorical_cols ]

# %%
# Create a subplot per sensitive category, and plot the bias in each
dfs_to_plot = {
    "lr=0": df[(df.lr == df.lr.min()) & (df.viscosity == 0)],
    f"lr={df.lr.max()}": df[(df.lr == df.lr.max()) & (df.viscosity == 0)],
}

fig, axs = plt.subplots(nrows=len(dfs_to_plot), ncols=len(categorical_cols), figsize=(10*len(categorical_cols), 5*len(dfs_to_plot)), sharey=True, sharex=True)

for i in range(len(dfs_to_plot)):
    _df = list(dfs_to_plot.values())[i]
    for j in range(len(categorical_cols)):
        for u in uniques[j]: 
            _df_subset = _df[_df[categorical_cols[j]] == u]
            gradients = np.array(_df_subset.gradient.to_list())
            average_gradient = gradients.cumsum(axis=0)/(np.arange(len(gradients))+1)[:,None]
            norm_average_gradient = np.linalg.norm(average_gradient, axis=1)
            _lp = sns.lineplot(ax=axs[i,j], x=_df_subset.admittime, y=norm_average_gradient, label=u.lower().split('/')[0])
            axs[0,j].set_title(categorical_cols[j].lower().replace("_"," "))
            axs[-1,j].set_xlabel("patient admission date")
            axs[i,0].set_ylabel(f"norm of avg grad\n ({list(dfs_to_plot.keys())[i]})")
            for tick in axs[i,j].get_xticklabels():
                tick.set_rotation(45)  # adjust the rotation angle as needed
sns.despine(top=True, right=True)
plt.tight_layout()
if save:
    plt.savefig('./plots/' + experiment_name + "/" + "stratified.pdf")
if show:
    plt.show()