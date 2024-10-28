import marimo

__generated_with = "0.3.4"
app = marimo.App()


@app.cell
def __():
    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import pandas as pd
    import seaborn as sns
    return glob, mcolors, np, os, pd, plt, sns


@app.cell
def __(T, df, np, plt, sns):
    # Make plot of the norms
    etas_to_plot = [0.1, 0.2, 0.5]
    plot_every = 10
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _t = np.arange(T)+1
    min_eta = min(etas_to_plot)

    _to_plot = df[
        df.lr.isin(etas_to_plot) &
        (df.time % plot_every == 0)
    ].sort_values(["lr"], ascending=False)
    _lp = sns.lineplot(
        ax=_ax, 
        data=_to_plot,
        x="time", 
        y="norm_avg_grad", 
        hue="parameters", 
        linewidth=3, 
        alpha=1
    )
    B_theta = np.nan_to_num(_to_plot.oracle_norm_thetaT_theta0,100).max()
    print(B_theta)
    # Also plot the theoretical bound
    bound = B_theta/(min_eta * _t)
    _ax.plot(_t, bound, 'k--', linewidth=3, label=r'Bound')
    _ax.set_ylabel("Norm avg grad")
    _ax.set_xlabel("Time (T)")
    _ax.set_ylim([0,2])

    plt.tight_layout()
    plt.gca()
    return (
        B_theta,
        bound,
        etas_to_plot,
        min_eta,
        plot_every,
    )


@app.cell
def __(
    T,
    df,
    etas_to_plot,
    np,
    plot_every,
    plt,
    sns,
):
    # Make plot of the signed average gradient
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _t = np.arange(T)+1

    _to_plot = df[
        df.lr.isin(etas_to_plot) &
        (df.time % plot_every == 0)
    ].sort_values(["lr"], ascending=False)
    _lp = sns.lineplot(
        ax=_ax, 
        data=_to_plot,
        x="time", 
        y="g_1", 
        hue="parameters", 
        linewidth=1, 
        alpha=0.5
    )

    _ax.set_ylabel("Avg grad")
    _ax.set_xlabel("Time (T)")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def __(
    T,
    df,
    etas_to_plot,
    np,
    plot_every,
    plt,
    sns,
):
    # Now plot the iterates and y_t values for the first d=2 dimensions
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _to_plot = df[
        df.lr.isin(etas_to_plot) & 
        (df.time % plot_every == 0)
    ].sort_values("lr", ascending=False)
    normalized_time = np.arange(T//plot_every)/(T//plot_every)
    _lp = sns.scatterplot(
        ax=_ax, 
        data=_to_plot, 
        x='theta_1', 
        y='theta_2', 
        hue="parameters", 
        alpha=0.5
    )
    sns.scatterplot(
        ax=_ax, 
        data=_to_plot[_to_plot.lr == etas_to_plot[0]], 
        x="y_1", 
        y="y_2", 
        c=normalized_time, 
        cmap="Grays", 
        alpha=0.1, 
        zorder=-1
    )
    _ax.set_ylabel(r'$\theta_2$')
    _ax.set_xlabel(r'$\theta_1$')
    _ax.legend(loc='upper left')

    plt.tight_layout()
    plt.gca()
    return normalized_time,


if __name__ == "__main__":
    app.run()
