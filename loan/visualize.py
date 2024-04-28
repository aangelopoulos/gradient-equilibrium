import marimo

__generated_with = "0.4.7"
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
def __(__file__, np, os, pd):
    # Read in the data
    experiment = 'gradient_boosting'
    # Get path of current file
    base_path = os.path.dirname(os.path.abspath(__file__))
    dfs = [ pd.read_pickle(base_path + '/.cache/' + experiment + "/" + f) for f in os.listdir(base_path + '/.cache/' + experiment) ]
    df = pd.concat(dfs, ignore_index=False)
    df = df.reset_index(names="time")
    df['parameters'] = [r'$\eta=$' + str(row['lr']) + r', $\kappa=$' + str(row['viscosity']) for _, row in df.iterrows()]
    df['norm_avg_grad'] = df.average_gradient.apply(np.linalg.norm, ord=np.inf)
    theta0 = np.array(df.theta.iloc[0])
    d = len(theta0)
    for j in range(d):
        df['avg_grad_' + str(j+1)] = df.average_gradient.apply(lambda x : x[j])
    df['norm_iterate'] = df.theta.apply(np.linalg.norm, ord=np.inf)
    df['theta_1'] = df.theta.apply(lambda th : th[0])
    df['theta_2'] = df.theta.apply(lambda th : th[1])
    df['y_1'] = df.y.apply(lambda y : y[0])
    df['y_2'] = df.y.apply(lambda y : y[1])
    df['yhat_1'] = df.y.apply(lambda yhat : yhat[0])
    df['yhat_2'] = df.y.apply(lambda yhat : yhat[1])
    df['thetaT_theta0'] = df.theta.apply(lambda th : th - theta0)
    df['oracle_norm_thetaT_theta0'] = df.theta.apply(lambda th : np.linalg.norm(th - theta0))
    T = len(df)-1
    df = df[df.time < T]
    df['normalized_time'] = df['time']/(T-1)
    df.norm_avg_grad = df.norm_avg_grad.replace([-np.inf, np.inf], np.nan)
    return T, base_path, d, df, dfs, experiment, j, theta0


@app.cell
def __(sns):
    # Plotting setup
    sns.set()
    sns.set_style("whitegrid")
    sns.set_context("talk")
    return


@app.cell
def __(T, df, np, plt, sns):
    # Make plot of the gradient norms
    kappas_to_plot = [0, 0.1, 0.5, 0.9]
    etas_to_plot = [0, 0.05, 0.1, 0.5]
    plot_every = 10
    _fig, _axs = plt.subplots(nrows=1, ncols=len(kappas_to_plot), figsize=(len(kappas_to_plot)*10,5), sharex="col", sharey="row")
    _t = np.arange(T)+1
    min_eta = min(etas_to_plot[1:])

    for _j in range(len(kappas_to_plot)):
        _kappa = kappas_to_plot[_j]
        _to_plot = df[
            df.lr.isin(etas_to_plot) &
            (df.viscosity==_kappa) & 
            (df.time % plot_every == 0)
        ].sort_values(["lr", "viscosity"], ascending=False)
        _lp = sns.lineplot(
            ax=_axs[_j], 
            data=_to_plot,
            x="time", 
            y="norm_avg_grad", 
            hue="parameters", 
            linewidth=3, 
            alpha=1
        )
        _axs[_j].set_ylabel("Norm avg grad")
        _axs[_j].set_xlabel("Time (T)")
        _axs[_j].legend(loc='upper right')
    plt.tight_layout()
    plt.gca()
    return etas_to_plot, kappas_to_plot, min_eta, plot_every


@app.cell
def __(T, d, df, etas_to_plot, kappas_to_plot, np, plot_every, plt, sns):
    # Plot the per-group bias
    # Make plot of the gradient norms
    _fig, _axs = plt.subplots(nrows=d, ncols=len(kappas_to_plot), figsize=(len(kappas_to_plot)*10,5*d), sharex="col", sharey=True)
    _t = np.arange(T)+1

    for _i in range(d):
        for _j in range(len(kappas_to_plot)):
            _kappa = kappas_to_plot[_j]
            _to_plot = df[
                df.lr.isin(etas_to_plot) &
                (df.viscosity==_kappa) & 
                (df.time % plot_every == 0)
            ].sort_values(["lr", "viscosity"], ascending=False)
            _lp = sns.lineplot(
                ax=_axs[_i,_j], 
                data=_to_plot,
                x="time", 
                y="avg_grad_" + str(_i+1), 
                hue="parameters", 
                linewidth=3, 
                alpha=1
            )
            _axs[_i,_j].set_ylabel("Avg grad (group " + str(_i+1) + ")")
            _axs[_i,_j].set_xlabel("Time (T)")
            _axs[_i,_j].legend(loc='upper right')
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def __(T, df, etas_to_plot, kappas_to_plot, np, plot_every, plt, sns):
    # Make plot of the iterate norms
    _fig, _axs = plt.subplots(nrows=1,ncols=len(kappas_to_plot), figsize=(len(kappas_to_plot)*10,5), sharex="col", sharey="row")
    _t = np.arange(T)+1

    for _j in range(len(kappas_to_plot)):
        kappa = kappas_to_plot[_j]
        _to_plot = df[
            df.lr.isin(etas_to_plot) &
            (df.viscosity==kappa) & 
            (df.time % plot_every == 0)
        ].sort_values(["lr", "viscosity"], ascending=False)
        _lp = sns.lineplot(
            ax=_axs[_j], 
            data=_to_plot,
            x="time", 
            y="norm_iterate", 
            hue="parameters", 
            linewidth=3, 
            alpha=1
        )
        _axs[_j].set_ylabel("Norm iterate")
        _axs[_j].set_xlabel("Time (T)")
        _axs[_j].legend(loc='upper right')
    plt.tight_layout()
    plt.gca()
    return kappa,


@app.cell
def __(T, df, etas_to_plot, kappas_to_plot, np, plot_every, plt, sns):
    # Now plot the iterates and y_t values for the first d=2 dimensions
    _fig, _axs = plt.subplots(nrows=1, ncols=len(kappas_to_plot), figsize=(len(kappas_to_plot)*10,5), sharex="col", sharey="row")
    for _j in range(len(kappas_to_plot)):
        _to_plot = df[
            df.lr.isin(etas_to_plot) & 
            (df.viscosity==kappas_to_plot[_j]) & 
            (df.time % plot_every == 0)
        ].sort_values("lr", ascending=False)
        normalized_time = np.arange(T//plot_every)/(T//plot_every)
        _lp = sns.scatterplot(
            ax=_axs[_j], 
            data=_to_plot, 
            x='theta_1', 
            y='theta_2', 
            hue="parameters", 
            #c=normalized_time,
            alpha=0.5
        )
        _axs[0].set_ylabel(r'$\theta_2$')
        _axs[_j].set_xlabel(r'$\theta_1$')
        _axs[_j].legend(loc='upper left')

    plt.tight_layout()
    plt.gca()
    return normalized_time,


if __name__ == "__main__":
    app.run()
