import marimo

__generated_with = "0.3.8"
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
    # Make plot of the gradient norms
    etas_to_plot = [0, 0.01, 0.05, 0.1]
    shifts_to_plot = [1, 5, 10]
    plot_every = 10
    _fig, _axs = plt.subplots(nrows=len(shifts_to_plot), ncols=1, figsize=(10, 5*len(shifts_to_plot)), sharex=True, sharey=True)
    _t = np.arange(T)+1
    min_eta = min(etas_to_plot[1:])

    for _i in range(len(shifts_to_plot)):
        _shift = shifts_to_plot[_i]
        _to_plot = df[
            df.lr.isin(etas_to_plot) &
            (df.By == _shift) &
            (df.time % plot_every == 0)
        ].sort_values(["lr"], ascending=False)
        _lp = sns.lineplot(
            ax=_axs[_i] if len(shifts_to_plot) > 1 else _axs, 
            data=_to_plot,
            x="time", 
            y="norm_avg_grad", 
            hue="parameters", 
            linewidth=3, 
            alpha=1
        )
        B_theta = np.nan_to_num(_to_plot.oracle_norm_thetaT_theta0,100).max()
        # Also plot the theoretical bound
        bound = B_theta/(min_eta * _t)
        _axs[_i].plot(_t[100:], bound[100:], 'k--', linewidth=3, label=r'bound') if len(shifts_to_plot) > 1 else _axs.plot(_t[100:], bound[100:], 'k--', linewidth=3, label=r'bound')
        _axs[_i].set_ylabel("Norm avg grad (By=" + str(shifts_to_plot[_i]) + ')') if len(shifts_to_plot) > 1 else _axs.set_ylabel("Norm avg grad (By=" + str(shifts_to_plot[_i]) + ')')
        _axs[_i].set_xlabel("Time (T)") if len(shifts_to_plot) > 1 else _axs.set_xlabel("Time (T)")
        _axs[_i].legend(loc='upper right') if len(shifts_to_plot) > 1 else _axs.legend(loc='upper right')
    plt.tight_layout()
    plt.gca()
    return (
        B_theta,
        bound,
        etas_to_plot,
        plot_every,
        shifts_to_plot,
    )


@app.cell
def __(
    T,
    df,
    etas_to_plot,
    np,
    plot_every,
    plt,
    shifts_to_plot,
    sns,
):
    # Make plot of the iterate norms
    _fig, _axs = plt.subplots(nrows=len(shifts_to_plot), ncols=1, figsize=(10, 5*len(shifts_to_plot)), sharex=True, sharey=True)
    _t = np.arange(T)+1

    for _i in range(len(shifts_to_plot)):
        _shift = shifts_to_plot[_i]
        _to_plot = df[
            df.lr.isin(etas_to_plot) &
            (df.By == _shift) &
            (df.time % plot_every == 0)
        ].sort_values(["lr"], ascending=False)
        _lp = sns.lineplot(
            ax=_axs[_i] if len(shifts_to_plot) > 1 else _axs, 
            data=_to_plot,
            x="time", 
            y="norm_iterate", 
            hue="parameters", 
            linewidth=3, 
            alpha=1
        )
        _axs[_i].set_ylabel("Norm iterate (By=" + str(shifts_to_plot[_i]) + ')') if len(shifts_to_plot) > 1 else _axs.set_ylabel("Norm iterate (By=" + str(shifts_to_plot[_i]) + ')')
        _axs[_i].set_xlabel("Time (T)") if len(shifts_to_plot) > 1 else _axs.set_xlabel("Time (T)")
        _axs[_i].legend(loc='upper right') if len(shifts_to_plot) > 1 else _axs.legend(loc='upper right')
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
    shifts_to_plot,
    sns,
):
    # Now plot the iterates and y_t values for the first d=2 dimensions
    _fig, _axs = plt.subplots(nrows=len(shifts_to_plot), ncols=1, figsize=(10, 5*len(shifts_to_plot)), sharex=True, sharey=True)
    for _i in range(len(shifts_to_plot)):
        _to_plot = df[
            df.lr.isin(etas_to_plot) & 
            (df.By==shifts_to_plot[_i]) &
            (df.time % plot_every == 0)
        ].sort_values("lr", ascending=False)
        normalized_time = np.arange(T//plot_every)/(T//plot_every)
        _lp = sns.scatterplot(
            ax=_axs[_i] if len(shifts_to_plot) > 1 else _axs, 
            data=_to_plot, 
            x='theta_1', 
            y='theta_2', 
            hue="parameters", 
            #c=normalized_time,
            alpha=0.5
        )
        _axs[_i].set_ylabel(r'$\theta_2$ (By$=$' + str(shifts_to_plot[_i]) + ')') if len(shifts_to_plot) > 1 else _axs.set_ylabel(r'$\theta_2$ (By$=$' + str(shifts_to_plot[_i]) + ')')
        _axs[_i].set_xlabel(r'$\theta_1$') if len(shifts_to_plot) > 1 else _axs.set_xlabel(r'$\theta_1$')
        _axs[_i].legend(loc='upper left') if len(shifts_to_plot) > 1 else _axs.legend(loc='upper left')

    plt.tight_layout()
    plt.gca()
    return normalized_time,

if __name__ == "__main__":
    app.run()
