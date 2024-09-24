import os
# Import from ../core
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../', 'core'))
import numpy as np
import torch
import torch.nn as nn
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.optim import SGD
from algorithms import GD, LogisticModel
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

def set_randomness(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@hydra.main(config_path='configs', config_name='gradient_boosting_sex_familystatus', version_base="1.3.2")
def main(cfg):
# Get job ID
    hydra_cfg = HydraConfig.get()
    job_id = hydra_cfg.job.id
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

# Load the data
    data = pd.read_pickle(f"./.cache/{cfg.model_type}.pkl")
    groups = torch.tensor(pd.get_dummies(data[cfg.experiment.dataset.columns]).values, dtype=torch.float32)
    data['residuals'] = data['target'] - data['prediction']
    d = groups.shape[1]
    n = len(data)

# Initialize the simple model
    model = LogisticModel(torch.zeros((d,)))

# Define the mean squared error loss
    loss_fn = nn.MSELoss(reduction='sum')

# Initialize the Gradient Descent optimizer
    optimizer = GD(model.parameters(), lr=cfg.experiment.optimizer.lr)

# Training loop
    thetas = torch.zeros(n+1, d)
    ys = torch.zeros(n+1)
    gs = torch.zeros(n+1, d)
    fs = torch.zeros(n+1)
    gradients = torch.zeros(n+1, d)
    losses = torch.zeros(n+1)
    average_gradients = torch.zeros(n+1, d)
    average_loss = torch.zeros(n+1, d)

    model = model.to(device)

    for t in range(len(data)):
        # Set up data
        g_t = groups[t]
        y_t = torch.tensor(data['residuals'].iloc[t], dtype=torch.float32)

        optimizer.zero_grad()
        thetas[t+1] = model.theta.detach().cpu()
        prediction = model(g_t.to(device))
        fs[t+1] = prediction.detach().cpu()
        loss = 0.5*loss_fn(prediction, y_t.to(device).squeeze())
        loss.backward()
        optimizer.step()
        ys[t+1] = y_t.detach().cpu()
        gs[t+1] = g_t.detach().cpu()
        gradients[t+1] = model.theta.grad.detach().cpu()
        average_gradients[t+1] = gradients[:t+1].mean(dim=0)

# Cache the thetas, ys, gradients, and norms in a pandas dictionary
    os.makedirs('.cache/' + cfg.experiment_name, exist_ok=True)
    df = pd.DataFrame({'theta': thetas.tolist(), 'y': ys.tolist(), 'g': gs.tolist(), 'gradient': gradients.tolist(), 'average_gradient': average_gradients.tolist()})
    df['lr'] = cfg.experiment.optimizer.lr
    df['d'] = d
    df.to_pickle('.cache/' + cfg.experiment_name + '/' + job_id + '.pkl')

if __name__ == "__main__":
    main()
