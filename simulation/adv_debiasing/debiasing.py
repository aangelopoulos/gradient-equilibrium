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

def get_adversarial_gradient(theta, By):
    # Goal: pick g and y to Try to make the iterates as large as possible
    # Maximize one coordinate of the gradient, g(g^Ttheta-y).
    # If theta[i] is positive, then we want to increase it. If theta[i] is negative, then we want to decrease it.
    wc_g = torch.zeros(len(theta))
    wc_y = 0
    wc_grad = 0
    for i in range(len(theta)):
        _g = torch.zeros(len(theta))
        _g[i] = 1
        _y = By * torch.sign(theta[i])
        _grad = torch.dot(_g, theta) - _y
        if _grad < wc_grad:
            wc_g = _g
            wc_y = _y
            wc_grad = _grad
    return wc_g, torch.tensor([wc_y]).float()

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

# Initialize the simple model
    model = OLSModel(torch.tensor(cfg.experiment.model.theta0))

# Define the mean squared error loss
    loss_fn = nn.MSELoss(reduction='sum')

# Initialize the Gradient Descent optimizer
    optimizer = GD(model.parameters(), lr=cfg.experiment.optimizer.lr)

# Training loop
    thetas = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)
    ys = torch.zeros(cfg.experiment.dataset.size+1)
    gs = torch.zeros(cfg.experiment.dataset.size+1)
    gradients = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)
    average_gradients = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)

    model = model.to(device)

    for t in range(cfg.experiment.dataset.size):
        optimizer.zero_grad()
        thetas[t] = model.theta.detach().cpu()
        g_t, y_t = get_adversarial_gradient(thetas[t], cfg.experiment.dataset.By)
        prediction = model(g_t.to(device))
        loss = 0.5*loss_fn(prediction.squeeze(), y_t.to(device).squeeze())
        loss.backward()
        optimizer.step()
        ys[t] = y_t.detach().cpu()
        gs[t] = g_t.detach().cpu()
        gradients[t] = model.theta.grad.detach().cpu()
        average_gradients[t] = gradients[:t+1].mean(dim=0)

# Cache the thetas, ys, gradients, and norms in a pandas dictionary
    os.makedirs('.cache/' + cfg.experiment_name, exist_ok=True)
    df = pd.DataFrame({'theta': thetas.tolist(), 'y': ys.tolist(), 'g': gs.tolist(), 'gradient': gradients.tolist(), 'average_gradient': average_gradients.tolist()})
    df['lr'] = cfg.experiment.optimizer.lr
    df['By'] = cfg.experiment.dataset.By
    df['d'] = cfg.experiment.dataset.d
    df['size'] = cfg.experiment.dataset.size
    df.to_pickle('.cache/' + cfg.experiment_name + '/' + job_id + '.pkl')

if __name__ == "__main__":
    main()
