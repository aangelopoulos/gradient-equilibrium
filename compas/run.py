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


@hydra.main(config_path='configs', config_name='base', version_base="1.3.2")
def main(cfg):
# Get job ID
    hydra_cfg = HydraConfig.get()
    job_id = hydra_cfg.job.id
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

# Load the data
    df = pd.read_csv('./raw_data/compas-scores-two-years.csv')
    df = df[['compas_screening_date', 'sex', 'race', 'v_decile_score', 'is_recid']]
    df = df[df.race.isin(["African-American", "Caucasian", "Hispanic"])]
    df['phat'] = (df['v_decile_score']-1)/9.0
    df.compas_screening_date = pd.to_datetime(df.compas_screening_date)
    df = df.sort_values(by='compas_screening_date')

# Run debiasing
    y = torch.tensor(df.is_recid.to_numpy()).float()
    yhat = torch.tensor(df.phat.to_numpy()).float()
    dummy_df = pd.get_dummies(df.race)
    x = torch.tensor(dummy_df.values).float()
    order = dummy_df.columns.values.tolist()

# Initialize the simple model
    n = len(x)
    d = x.shape[1]
    model = OLSModel(torch.zeros((d,)))

    # Initialize the Gradient Descent optimizer
    optimizer = GD(model.parameters(), lr=cfg.experiment.optimizer.lr, viscosity=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
    thetas = torch.zeros(n+1, d, dtype=torch.float32)
    xs = torch.zeros(n+1, dtype=int)
    ys = torch.zeros(n+1, dtype=torch.float32)
    fs = torch.zeros(n+1, dtype=torch.float32)
    yhats = torch.zeros(n+1, dtype=torch.float32)
    losses = torch.zeros(n+1, dtype=torch.float32)
    gradients = torch.zeros(n+1, d, dtype=torch.float32)
    average_losses = torch.zeros(n+1, dtype=torch.float32)
    average_gradients = torch.zeros(n+1, d, dtype=torch.float32)
    race = [None]
    race_count = [0]

    model = model.to(device)

    loss_fn = nn.MSELoss()

    for t in tqdm(range(n)):
        # Set up data
        x_t = x[t]
        y_t = y[t]
        f_t = yhat[t]
        r_t =y_t-f_t

        # Perform optimization
        optimizer.zero_grad()
        thetas[t+1] = model.theta.detach().cpu()
        prediction = model(x_t.to(device))
        r_t = r_t.to(device)
        prediction.squeeze()
        loss = loss_fn(prediction, r_t)
        loss.backward()
        optimizer.step()

        # Store results
        race_idx = x_t.argmax().detach().cpu()
        ys[t+1] = y_t.detach().cpu()
        fs[t+1] = f_t
        yhats[t+1] = f_t + prediction
        losses[t+1] = loss.detach().cpu().item()
        gradients[t+1] = model.theta.grad.detach().cpu()
        average_losses[t+1] = losses[:t+1].mean()
        average_gradients[t+1] = gradients[:t+1].mean(dim=0)
        thetas[t+1] = model.theta.detach().cpu()
        race += [order[race_idx]]
        race_count += [sum(np.array(race) == order[race_idx]) + 1]
    
# Cache the thetas, ys, gradients, and norms in a pandas dictionary
    os.makedirs('.cache/' + cfg.experiment_name, exist_ok=True)
    # Print the length of each list

    save_df = pd.DataFrame({
        'theta': thetas.tolist(), 
        'y': ys.tolist(), 
        'f': fs.tolist(), 
        'yhat': yhats.tolist(), 
        'loss' : losses.tolist(), 
        'gradient': gradients.tolist(), 
        'average gradient': average_gradients.tolist(), 
        'average loss': average_losses.tolist(),
        'race': race,
        'race count': race_count
    })

    save_df['lr'] = float(cfg.experiment.optimizer.lr)
    save_df['viscosity'] = cfg.experiment.optimizer.viscosity
    save_df['d'] = d
    save_df.to_pickle('.cache/' + cfg.experiment_name + '/' + job_id + '.pkl')

if __name__ == "__main__":
    main()
