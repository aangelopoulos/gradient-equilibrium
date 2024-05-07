import os
# Import from ../core
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../', 'core'))
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.optim import SGD
from algorithms import GD, OLSModel
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import pdb

def set_randomness(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path='configs', config_name='nomodel_ethnicity_marital', version_base="1.3.2")
def main(cfg):
# Get job ID
    hydra_cfg = HydraConfig.get()
    job_id = hydra_cfg.job.id
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

# Load the data
    data = pd.read_pickle(f"./.cache/{cfg.model_type}.pkl")
    if cfg.experiment.model.nomodel:
        data["f"] = 0
    if len(cfg.experiment.dataset.columns) > 0:
        xs = torch.tensor(pd.get_dummies(data[cfg.experiment.dataset.columns]).values.astype(float), dtype=torch.float32)
    else:
        xs = torch.ones(len(data),1)
    data['residuals'] = data['length_of_stay_float'] - data['f']
    d = xs.shape[1]
    #burnin_end = data[data.f > 0].index.min()
    data = data.tail(100000)#loc[burnin_end:]
    n = len(data)

# Initialize the simple model
    model = OLSModel(torch.zeros((d,)))

# Define the mean squared error loss
    loss_fn = nn.MSELoss(reduction='sum')

# Initialize the Gradient Descent optimizer
    optimizer = GD(model.parameters(), lr=cfg.experiment.optimizer.lr, viscosity=cfg.experiment.optimizer.viscosity)

# Training loop
    thetas = torch.zeros(n+1, d, dtype=torch.float32)
    ys = torch.zeros(n+1, dtype=torch.float32)
    fs = torch.zeros(n+1, dtype=torch.float32)
    yhats = torch.zeros(n+1, dtype=torch.float32)
    losses = torch.zeros(n+1, dtype=torch.float32)
    gradients = torch.zeros(n+1, d, dtype=torch.float32)
    average_losses = torch.zeros(n+1, dtype=torch.float32)
    average_gradients = torch.zeros(n+1, d, dtype=torch.float32)

    model = model.to(device)

    for t in tqdm(range(len(data))):
        # Set up data
        x_t = xs[t]
        y_t = torch.tensor(data['length_of_stay_float'].iloc[t], dtype=torch.float32)
        f_t = torch.tensor(data['f'].iloc[t], dtype=torch.float32)
        r_t = torch.tensor(data['residuals'].iloc[t], dtype=torch.float32)

        # Perform optimization
        optimizer.zero_grad()
        thetas[t+1] = model.theta.detach().cpu()
        prediction = model(x_t.to(device))
        loss = 0.5*loss_fn(prediction.squeeze(), r_t.to(device).squeeze())
        loss.backward()
        optimizer.step()

        # Store results
        ys[t+1] = y_t.detach().cpu()
        fs[t+1] = f_t
        yhats[t+1] = f_t + prediction
        losses[t+1] = loss.detach().cpu().item()
        gradients[t+1] = model.theta.grad.detach().cpu()
        average_gradients[t+1] = gradients[:t+1].mean(dim=0)
        average_losses[t+1] = losses[:t+1].mean()
        
# Cache the thetas, ys, gradients, and norms in a pandas dictionary
    os.makedirs('.cache/' + cfg.experiment_name, exist_ok=True)
    df = pd.DataFrame({
        'theta': thetas.tolist(), 
        'y': ys.tolist(), 
        'f': fs.tolist(), 
        'yhat': yhats.tolist(), 
        'loss' : losses.tolist(), 
        'gradient': gradients.tolist(), 
        'average gradient': average_gradients.tolist(), 
        'average loss' : average_losses.tolist()
    })
    df.loc[1:,"admittime"] = data.admittime.values
    for col in cfg.experiment.dataset.columns + cfg.covariates_for_plotting:
        if col not in df.columns:
            df.loc[1:,col] = data[col].values

    df['lr'] = float(cfg.experiment.optimizer.lr)
    df['viscosity'] = cfg.experiment.optimizer.viscosity
    df['d'] = d
    df.to_pickle('.cache/' + cfg.experiment_name + '/' + job_id + '.pkl')

if __name__ == "__main__":
    main()
