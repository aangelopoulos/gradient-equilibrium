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
from torch.optim import SGD
from algorithms import GD, OLSModel
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
    def __init__(self, size, trends, d, seed):
        """
        Args:
            size (int): the size of the dataset
            trends (torch.Tensor): the trends to be added to the yhats. It is a tensor with shape (size, d)
            d (int): the dimension of the one-hot vector; i.e., the number of groups
            seed (int): the random seed
        """
        self.size = size
        self.d = d
        self.trends = trends
        set_randomness(seed=seed)

        # Create synthetic data with a slow distribution shift
        self.g = []
        self.y = []
        self.yhat = []
        for i in range(size):
            # Randomly sample g_t, a d-dimensional one-hot vector
            g_t = torch.zeros(d)
            g_t[np.random.randint(d)] = 1
            self.g.append(g_t)
            y_t = torch.clip(torch.randn(1),-5,5)
            yhat_t = trends[i].dot(g_t)
            self.y.append(y_t)
            self.yhat.append(yhat_t)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        y_t = self.y[idx]
        g_t = self.g[idx]
        yhat_t = self.yhat[idx]
        return idx, g_t, y_t, yhat_t

@hydra.main(config_path='configs', config_name='basic', version_base="1.3.2")
def main(cfg):
# Get job ID
    hydra_cfg = HydraConfig.get()
    job_id = hydra_cfg.job.id
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")

# Create the synthetic dataset
    trends = torch.ones(cfg.experiment.dataset.size, cfg.experiment.dataset.d)*cfg.experiment.dataset.trend_amplitude
    if cfg.experiment.dataset.trend_period > 0:
        for _d in range(cfg.experiment.dataset.d):
            trends[:,_d] = _d*cfg.experiment.dataset.trend_amplitude*torch.sin(torch.linspace(0, cfg.experiment.dataset.size, cfg.experiment.dataset.size)*(2*np.pi)/cfg.experiment.dataset.trend_period) + _d*cfg.experiment.dataset.trend_amplitude
    dataset = SyntheticDataset(size=cfg.experiment.dataset.size, trends=trends, d=cfg.experiment.dataset.d, seed=0)

# Initialize the simple model
    model = OLSModel(torch.tensor(cfg.experiment.model.theta0))

# Define the mean squared error loss
    loss_fn = nn.MSELoss(reduction='sum')

# Initialize the Gradient Descent optimizer
    optimizer = GD(model.parameters(), lr=cfg.experiment.optimizer.lr)

# Prepare the experiment
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Training loop (simplified for demonstration)
    thetas = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)
    ys = torch.zeros(cfg.experiment.dataset.size+1)
    yhats = torch.zeros(cfg.experiment.dataset.size+1)
    gs = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)
    gradients = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)
    average_gradients = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)
    thetas[0] = model.theta.detach()

    model = model.to(device)

    for t, g_t, y_t, yhat_t in loader:
        optimizer.zero_grad()
        prediction = model(g_t.to(device))
        loss = 0.5*loss_fn(prediction.squeeze(), (y_t-yhat_t).to(device).squeeze())
        loss.backward()
        optimizer.step()
        thetas[t+1] = model.theta.detach().cpu()
        ys[t] = y_t.detach().cpu()
        yhats[t] = yhat_t.detach().cpu()
        gs[t] = g_t.detach().cpu()
        gradients[t] = model.theta.grad.detach().cpu()
        average_gradients[t] = gradients[:t+1].mean(dim=0)

# Cache the thetas, ys, gradients, and norms in a pandas dictionary
    os.makedirs('.cache/' + cfg.experiment_name, exist_ok=True)
    df = pd.DataFrame({'theta': thetas.tolist(), 'y': ys.tolist(), 'yhat': yhats.tolist(), 'g': gs.tolist(), 'gradient': gradients.tolist(), 'average_gradient': average_gradients.tolist()})
    df['lr'] = cfg.experiment.optimizer.lr
    df['d'] = cfg.experiment.dataset.d
    df['size'] = cfg.experiment.dataset.size
    df['trend_period'] = cfg.experiment.dataset.trend_period
    df['trend_amplitude'] = cfg.experiment.dataset.trend_amplitude
    df.to_pickle('.cache/' + cfg.experiment_name + '/' + job_id + '.pkl')

if __name__ == "__main__":
    main()
