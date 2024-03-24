import os
# Import from ../core
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'core'))
import numpy as np
import torch
import torch.nn as nn
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from algorithms import VGD
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

def set_randomness(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Create a synthetic dataset
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, size, trend, d, seed):
        self.size = size
        self.trend = trend # Trend is size x d
        self.d = d
        set_randomness(seed=seed)

        # Create synthetic data with a slow distribution shift
        self.data = []
        for i in range(size):
            # Simulate the distribution shift over time
            y_t = torch.randn(d) + self.trend[i]
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

@hydra.main(config_path='configs', config_name='basic', version_base="1.3.2")
def main(cfg):
# Get job ID
    hydra_cfg = HydraConfig.get()
    job_id = hydra_cfg.job.id

# Create the synthetic dataset
    if cfg.experiment.dataset.trend_type == 'sinusoidal':
        rate = torch.arange(cfg.experiment.dataset.size)*2*np.pi * cfg.experiment.dataset.distribution_shift_speed
        trend = 5*torch.stack([torch.sin(rate), torch.cos(rate), torch.sin(rate)], axis=1) + 20
    elif cfg.experiment.dataset.trend_type == 'linear':
        trend = np.log(np.arange(cfg.experiment.dataset.size)+1) * cfg.experiment.dataset.distribution_shift_speed + 2
    else:
        raise ValueError('Invalid trend type')
    dataset = SyntheticDataset(size=cfg.experiment.dataset.size, trend=trend, d=cfg.experiment.dataset.d, seed=0)

# Initialize the simple model
    model = SimpleModel(torch.tensor(cfg.experiment.model.theta0))

# Define the mean squared error loss
    loss_fn = nn.MSELoss()

# Initialize the Viscosity Gradient Descent optimizer
    optimizer = VGD(model.parameters(), lr=cfg.experiment.optimizer.lr, viscosity=cfg.experiment.optimizer.viscosity)

# Prepare the experiment
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Training loop (simplified for demonstration)
    thetas = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)
    ys = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)
    gradients = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)
    average_gradients = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)
    thetas[0] = model.theta.detach()

    for t, y_t in loader:
        optimizer.zero_grad()
        prediction = model(t.float())
        loss = 0.5*loss_fn(prediction, y_t)
        loss.backward()
        optimizer.step()
        thetas[t+1] = model.theta.detach()
        ys[t] = y_t
        gradients[t] = model.theta.grad
        average_gradients[t] = gradients[:t+1].mean(dim=0)

# Cache the thetas, ys, gradients, and norms in a pandas dictionary
    os.makedirs('.cache/' + cfg.experiment_name, exist_ok=True)
    df = pd.DataFrame({'theta': thetas.tolist(), 'y': ys.tolist(), 'gradient': gradients.tolist(), 'average_gradient': average_gradients.tolist()})
    df['lr'] = cfg.experiment.optimizer.lr
    df['viscosity'] = cfg.experiment.optimizer.viscosity
    df['distribution_shift_speed'] = cfg.experiment.dataset.distribution_shift_speed
    df['d'] = cfg.experiment.dataset.d
    df['size'] = cfg.experiment.dataset.size
    df['init_norm'] = torch.norm(torch.tensor(cfg.experiment.model.theta0)).item()
    df['trend_type'] = cfg.experiment.dataset.trend_type
    df.to_pickle('.cache/' + cfg.experiment_name + '/' + job_id + '.pkl')

if __name__ == "__main__":
    main()
