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
def __(np, os, pd):
    # Read in the data
    experiment = 'basic'
    dfs = [ pd.read_pickle('.cache/' + experiment + "/" + f) for f in os.listdir('.cache/' + experiment) ]
    df = pd.concat(dfs, ignore_index=False)
    df = df.reset_index(names="time")
    df['parameters'] = [r'$\eta=$' + str(row['lr']) + r', $\kappa=$' + str(row['viscosity']) for _, row in df.iterrows()]
    df['norm_avg_grad'] = df.average_gradient.apply(np.linalg.norm)
    df['theta_1'] = df.theta.apply(lambda th : th[0])
    df['theta_2'] = df.theta.apply(lambda th : th[1])
    df['y_1'] = df.y.apply(lambda y : y[0])
    df['y_2'] = df.y.apply(lambda y : y[1])
    df['yhat_1'] = df.y.apply(lambda yhat : yhat[0])
    df['yhat_2'] = df.y.apply(lambda yhat : yhat[1])
    theta0 = np.array(df.theta.iloc[0])
    df['thetaT_theta0'] = df.theta.apply(lambda th : th - theta0)
    df['oracle_norm_thetaT_theta0'] = df.theta.apply(lambda th : np.linalg.norm(th - theta0))
    T = 2000
    df = df[df.time < T]
    df['normalized_time'] = df['time']/(T-1)
    df.norm_avg_grad = df.norm_avg_grad.replace([-np.inf, np.inf], np.nan)
    return T, df, dfs, experiment, theta0


@app.cell
def __(sns):
    # Plotting setup
    sns.set() 
    sns.set_style("whitegrid")
    sns.set_context("talk")
    return


@app.cell
def __(T, df, np, plt, sns):
    # Make plot of the norms
    kappas_to_plot = [0, 0.1, 0.5]
    etas_to_plot = [0, 0.01, 0.05, 0.1]
    shifts_to_plot = [0, 500, 1000]
    plot_every = 10
    _fig, _axs = plt.subplots(nrows=len(shifts_to_plot), ncols=len(kappas_to_plot), figsize=(len(kappas_to_plot)*10,5*len(shifts_to_plot)), sharex="col", sharey="row")
    _t = np.arange(T)+1
    min_eta = min(etas_to_plot[1:])

    for _i in range(len(shifts_to_plot)):
        for _j in range(len(kappas_to_plot)):
            shift = shifts_to_plot[_i]
            kappa = kappas_to_plot[_j]
            _to_plot = df[
                df.lr.isin(etas_to_plot) &
                (df.viscosity==kappa) & 
                (df.trend_period == shift) &
                (df.time % plot_every == 0)
            ].sort_values(["lr", "viscosity"], ascending=False)
            _lp = sns.lineplot(
                ax=_axs[_i,_j], 
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
            bound= B_theta/(min_eta * _t)
            _lp.plot(_t[100:], bound[100:], 'k--', linewidth=3, label=r'Bound')
            _axs[_i,_j].set_ylabel("Norm avg grad (drift=" + str(shifts_to_plot[_i]) + ')')
            _axs[_i,_j].set_xlabel("Time (T)")
            _axs[_i,_j].legend(loc='upper right')

    plt.tight_layout()
    plt.gca()
    return (
        B_theta,
        bound,
        etas_to_plot,
        kappa,
        kappas_to_plot,
        min_eta,
        plot_every,
        shift,
        shifts_to_plot,
    )


@app.cell
def __(
    T,
    df,
    etas_to_plot,
    kappas_to_plot,
    np,
    plot_every,
    plt,
    shifts_to_plot,
    sns,
):
    # Make plot of the signed average gradient
    _fig, _axs = plt.subplots(nrows=len(shifts_to_plot), ncols=len(kappas_to_plot), figsize=(len(kappas_to_plot)*10,5*len(shifts_to_plot)), sharex="col", sharey="row")
    _t = np.arange(T)+1

    for _i in range(len(shifts_to_plot)):
        for _j in range(len(kappas_to_plot)):
            _to_plot = df[
                df.lr.isin(etas_to_plot) &
                (df.viscosity==kappas_to_plot[_j]) & 
                (df.distribution_shift_speed==shifts_to_plot[_i]) & 
                (df.time % plot_every == 0)
            ].sort_values(["lr", "viscosity"], ascending=False)
            _lp = sns.lineplot(
                ax=_axs[_i,_j], 
                data=_to_plot,
                x="time", 
                y="g_1", 
                hue="parameters", 
                linewidth=1, 
                alpha=0.5
            )

            # Also plot the theoretical bound
            #_lp.plot(_t, 1/np.sqrt(_t), 'k--', linewidth=3, label=r'$\frac{1}{\sqrt{T}}$')
            _axs[_i,_j].set_ylabel("Avg grad (drift=" + str(shifts_to_plot[_i]) + ')')
            _axs[_i,_j].set_xlabel("Time (T)")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def __(
    T,
    df,
    etas_to_plot,
    kappas_to_plot,
    np,
    plot_every,
    plt,
    shifts_to_plot,
    sns,
):
    # Now plot the iterates and y_t values for the first d=2 dimensions
    _fig, _axs = plt.subplots(nrows=len(shifts_to_plot), ncols=len(kappas_to_plot), figsize=(len(kappas_to_plot)*10,5*len(shifts_to_plot)), sharex="col", sharey="row")
    for _i in range(len(shifts_to_plot)):
        for _j in range(len(kappas_to_plot)):
            _to_plot = df[
                df.lr.isin(etas_to_plot) & 
                (df.viscosity==kappas_to_plot[_j]) & 
                (df.distribution_shift_speed==shifts_to_plot[_i]) &
                (df.time % plot_every == 0)
            ].sort_values("lr", ascending=False)
            normalized_time = np.arange(T//plot_every)/(T//plot_every)
            _lp = sns.scatterplot(
                ax=_axs[_i,_j], 
                data=_to_plot, 
                x='theta_1', 
                y='theta_2', 
                hue="parameters", 
                #c=normalized_time,
                alpha=0.5
            )
            sns.scatterplot(
                ax=_axs[_i,_j], 
                data=_to_plot[_to_plot.lr == etas_to_plot[0]], 
                x="y_1", 
                y="y_2", 
                c=normalized_time, 
                cmap="Grays", 
                alpha=0.1, 
                zorder=-1
            )
            if _j == 0:
                _axs[_i,_j].set_ylabel(r'$\theta_2$ (drift$=$' + str(shifts_to_plot[_i]) + ')')
            _axs[_i,_j].set_xlabel(r'$\theta_1$')
            _axs[_i,_j].legend(loc='upper left')

    plt.tight_layout()
    plt.gca()
    return normalized_time,


if __name__ == "__main__":
    app.run()
