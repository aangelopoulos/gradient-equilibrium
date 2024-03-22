import marimo

__generated_with = "0.3.3"
app = marimo.App()


@app.cell
def __():
    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    return glob, np, os, pd, plt, sns


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
    T = 10000
    df = df[df.time < T]
    return T, df, dfs, experiment


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
    kappas_to_plot = [0, 0.1, 0.9]
    etas_to_plot = [0.001, 0.01, 0.1]
    shifts_to_plot = [0.0001, 0.0002, 0.0005]
    _fig, _axs = plt.subplots(nrows=len(shifts_to_plot), ncols=len(kappas_to_plot), figsize=(len(kappas_to_plot)*10,5*len(shifts_to_plot)), sharex="col", sharey="row")
    _t = np.arange(T)+1

    for _i in range(len(shifts_to_plot)):
        for _j in range(len(kappas_to_plot)):
            _to_plot = df[df.lr.isin(etas_to_plot) & (df.viscosity==kappas_to_plot[_j])].sort_values(["lr", "viscosity"], ascending=False)
            _lp = sns.lineplot(ax=_axs[_i,_j], data=_to_plot[(_to_plot.distribution_shift_speed==shifts_to_plot[_i]) & (_to_plot.init_norm == 0)], x="time", y="norm_avg_grad", hue="parameters", linewidth=1, alpha=0.5)
            
            # Also plot the theoretical bound
            #_lp.plot(_t, 1/np.sqrt(_t), 'k--', linewidth=3, label=r'$\frac{1}{\sqrt{T}}$')
            _axs[_i,_j].set_ylabel("Norm avg grad (drift=" + str(shifts_to_plot[_i]) + ')')
            _axs[_i,_j].set_xlabel("Time (T)")

    plt.tight_layout()
    plt.show()
    #os.makedirs('./results/' + cfg.experiment_name, exist_ok=True)
    #plt.savefig('./results/' + cfg.experiment_name + '/norms.pdf')
    return etas_to_plot, kappas_to_plot, shifts_to_plot


@app.cell
def __(T, df, etas_to_plot, kappas_to_plot, np, plt, shifts_to_plot, sns):
    # Now plot the iterates and y_t values for the first d=2 dimensions
    _fig, _axs = plt.subplots(nrows=len(shifts_to_plot), ncols=len(kappas_to_plot), figsize=(len(kappas_to_plot)*10,5*len(shifts_to_plot)), sharex="col", sharey="row")
    for _i in range(len(shifts_to_plot)):
        y_1 = df[df.distribution_shift_speed==shifts_to_plot[_i]].y_1[:T]
        y_2 = df[df.distribution_shift_speed==shifts_to_plot[_i]].y_2[:T]
        for _j in range(len(kappas_to_plot)):
            _to_plot = df[df.lr.isin(etas_to_plot) & (df.viscosity==kappas_to_plot[_j])].sort_values("lr", ascending=False)
            _lp = sns.lineplot(ax=_axs[_i,_j], data=_to_plot[(_to_plot.distribution_shift_speed==shifts_to_plot[_i]) & (_to_plot.init_norm == 0)], x='theta_1', y='theta_2', hue="parameters", sort=False, alpha=0.7)
            _lp.scatter(y_1, y_2, c=np.arange(T)/(T), cmap="Grays", alpha=0.1, zorder=-1)
            if _j == 0:
                _axs[_i,_j].set_ylabel(r'$\theta_2$ (drift$=$' + str(shifts_to_plot[_i]) + ')')
            _axs[_i,_j].set_xlabel(r'$\theta_1$')
            _axs[_i,_j].legend(loc='lower left')

    plt.tight_layout()
    plt.show()
    return y_1, y_2


if __name__ == "__main__":
    app.run()
