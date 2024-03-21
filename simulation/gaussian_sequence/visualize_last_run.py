import marimo

__generated_with = "0.3.3"
app = marimo.App()


@app.cell
def __():
    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    return glob, np, os, plt, sns


@app.cell
def __(sns):
    # Plot the results
    sns.set() 
    sns.set_style("whitegrid")
    sns.set_context("talk")
    return


@app.cell
def __(cfg, norms, np, os, plt, thetas, ys):
    # Make plot of the norms
    t = np.arange(cfg.experiment.dataset.size) + 1
    plt.plot(t, norms, linewidth=1, label=r'VGD ($\eta = {}$, $\kappa = {}$)'.format(cfg.experiment.optimizer.lr, cfg.experiment.optimizer. viscosity))

    # Also plot the theoretical bound
    plt.plot(t, norms.max()/np.sqrt(t), 'r--', linewidth=1, label='O(1/sqrt(t))')
    plt.xlabel('Time step')
    plt.ylabel('Norm of the gradient')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.legend()
    os.makedirs('./results/' + cfg.experiment_name, exist_ok=True)
    plt.savefig('./results/' + cfg.experiment_name + '/norms.pdf')
         
    # Now plot the iterates and y_t values for the first d=2 dimensions
    plt.figure()
    plt.plot(thetas[:,0].numpy(), thetas[:,1].numpy(), linewidth=1, label=r'$\theta_t$', alpha=0.5)
    plt.plot(ys[:,0].numpy(), ys[:,1].numpy(), linewidth=1, label=r'$y_t$', alpha=0.5)
    plt.xlabel(r'$\theta_{t,1}$')
    plt.ylabel(r'$\theta_{t,2}$')
    plt.tight_layout()
    plt.legend()
    plt.savefig('./results/' + cfg.experiment_name + '/iterates.pdf')

    return t,


if __name__ == "__main__":
    app.run()
