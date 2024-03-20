import os
# Import from ../core
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
import numpy as np
import torch
import torch.nn as nn
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from algorithms import VGD
import hydra
from omegaconf import DictConfig, OmegaConf

# Create a synthetic dataset
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, size, distribution_shift_speed, d):
        self.size = size
        self.distribution_shift_speed = distribution_shift_speed
        self.current_trend = 0.0
        self.d = d

        # Create synthetic data with a slow distribution shift
        self.data = []
        for i in range(size):
            # Simulate the distribution shift over time
            self.current_trend += distribution_shift_speed
            y_t = torch.randn(d) + self.current_trend
            self.data.append(y_t)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        y_t = self.data[idx]
        return idx, y_t

# Simple model that predicts y_t
class SimpleModel(nn.Module):
    def __init__(self, theta0):
        super(SimpleModel, self).__init__()
        self.d = theta0.shape[0]
        self.theta = nn.Parameter(theta0)

    def forward(self, x_t):
        # Model prediction is just the value of theta
        return self.theta[None,:].expand(x_t.shape[0],-1)

@hydra.main(config_path='configs', config_name='basic')
def main(cfg):
    pdb.set_trace()
# Create the synthetic dataset
    dataset = SyntheticDataset(size=cfg.experiment.dataset.size, distribution_shift_speed=cfg.experiment.dataset.distribution_shift_speed, d=cfg.experiment.dataset.d)

# Initialize the simple model
    model = SimpleModel(torch.tensor(cfg.experiment.model.theta0))

# Define the mean squared error loss
    loss_fn = nn.MSELoss()

# Initialize the Viscosity Gradient Descent optimizer
    optimizer = VGD(model.parameters(), lr=cfg.experiment.optimizer.lr, viscosity=cfg.experiment.optimizer.viscosity)

# Prepare the experiment
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Training loop (simplified for demonstration)
    thetas = torch.zeros(cfg.experiment.dataset.size, cfg.experiment.dataset.d)
    ys = torch.zeros(cfg.experiment.dataset.size, cfg.experiment.dataset.d)
    gradients = torch.zeros(cfg.experiment.dataset.size, cfg.experiment.dataset.d)
    norms = torch.zeros(cfg.experiment.dataset.size)
    for t, y_t in loader:
        optimizer.zero_grad()
        prediction = model(t.float())
        loss = 0.5*loss_fn(prediction, y_t)
        loss.backward()
        thetas[t] = model.theta.detach()
        ys[t] = y_t
        gradients[t] = model.theta.grad
        norms[t] = torch.norm(gradients[:t+1].mean(dim=0))
        optimizer.step()

# Cache the results
    os.makedirs('.cache/' + cfg.experiment_name, exist_ok=True)
    torch.save(thetas, '.cache/' + cfg.experiment_name + '/thetas.pt')
    torch.save(ys, '.cache/' + cfg.experiment_name + '/ys.pt')
    torch.save(gradients, '.cache/' + cfg.experiment_name + '/gradients.pt')
    torch.save(norms, '.cache/' + cfg.experiment_name + '/norms.pt')

# Plot the results
    sns.set()
    sns.set_style("whitegrid")
    sns.set_context("talk")

# Make plot of the norms
    t = np.arange(cfg.experiment.dataset.size) + 1
    plt.plot(t, norms, linewidth=1, label=r'VGD ($\eta = {}$, $\kappa = {}$)'.format(cfg.experiment.optimizer.lr, cfg.experiment.optimizer.viscosity))

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


if __name__ == "__main__":
    main()
