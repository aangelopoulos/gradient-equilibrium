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
from algorithms import VGD
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

def set_randomness(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_adversarial_gradient(theta, Bg, By):
    # Goal: pick g and y to Try to make the iterates as large as possible
    # Maximize one coordinate of the gradient, g(g^Ttheta-y).
    # If theta[i] is positive, then we want to increase it. If theta[i] is negative, then we want to decrease it.
    g = torch.zeros(len(theta))
    if theta[0] >= 0:
        # Set g to be a vector orthogonal to theta
        g[0] = Bg
        g[1] = -Bg
        y = By
        # Resulting gradient: -BgBy. Thus, theta[0] will become more positive.
    else:
        g[0] = Bg
        g[1] = -Bg
        y = -By
        # Resulting gradient: BgBy. Thus, theta[0] will become more negative.
    return g, torch.tensor([y]).float()

# OLS model
class OLSModel(nn.Module):
    def __init__(self, theta0):
        super(OLSModel, self).__init__()
        self.d = theta0.shape[0]
        self.theta = nn.Parameter(theta0)

    def forward(self, x_t):
        return torch.matmul(x_t, self.theta)

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

# Initialize the Viscosity Gradient Descent optimizer
    optimizer = VGD(model.parameters(), lr=cfg.experiment.optimizer.lr, viscosity=cfg.experiment.optimizer.viscosity)

# Training loop
    thetas = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)
    ys = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)
    gs = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)
    gradients = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)
    average_gradients = torch.zeros(cfg.experiment.dataset.size+1, cfg.experiment.dataset.d)

    model = model.to(device)

    for t in range(cfg.experiment.dataset.size):
        optimizer.zero_grad()
        thetas[t] = model.theta.detach().cpu()
        g_t, y_t = get_adversarial_gradient(thetas[t], cfg.experiment.dataset.Bg, cfg.experiment.dataset.By)
        prediction = model(g_t.to(device))
        loss = 0.5*loss_fn(prediction.squeeze(), y_t.to(device).squeeze())
        loss.backward()
        optimizer.step()
        ys[t] = y_t.detach().cpu()
        gs[t] = g_t.detach().cpu()
        gradients[t] = model.theta.grad.detach().cpu()
        average_gradients[t] = gradients[:t+1].mean(dim=0)
        #print(f"lr={cfg.experiment.optimizer.lr}, viscosity={cfg.experiment.optimizer.viscosity}, t={t}, loss={loss.item()}, theta={model.theta.detach().cpu().numpy()}, gradient={model.theta.grad.detach().cpu().numpy()}")

# Cache the thetas, ys, gradients, and norms in a pandas dictionary
    os.makedirs('.cache/' + cfg.experiment_name, exist_ok=True)
    df = pd.DataFrame({'theta': thetas.tolist(), 'y': ys.tolist(), 'g': gs.tolist(), 'gradient': gradients.tolist(), 'average_gradient': average_gradients.tolist()})
    df['lr'] = cfg.experiment.optimizer.lr
    df['By'] = cfg.experiment.dataset.By
    df['Bg'] = cfg.experiment.dataset.Bg
    df['viscosity'] = cfg.experiment.optimizer.viscosity
    df['d'] = cfg.experiment.dataset.d
    df['size'] = cfg.experiment.dataset.size
    df.to_pickle('.cache/' + cfg.experiment_name + '/' + job_id + '.pkl')

if __name__ == "__main__":
    main()
