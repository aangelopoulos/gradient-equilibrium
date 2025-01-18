import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
curr_dir = os.getcwd()
parent_dir = os.path.dirname(curr_dir)
sys.path.append(os.path.join(parent_dir, 'core'))
from algorithms import DebiasingModel, GD

# Set the seed for reproducibility
np.random.seed(3)

# Parameters
T = 1000
mu = 0
y = mu + torch.tensor(np.random.normal(size=T))
theta = np.zeros(T+1)
Sum_loss_sequence = np.zeros(T)
Sum_bias_sequence = np.zeros(T)
Sum_y_sequence = np.zeros(T)
Sum_y_2_sequence = np.zeros(T)
Regret_sequence = np.zeros(T)
eta = 0.2

loss_fn = torch.nn.MSELoss()
model = DebiasingModel(torch.tensor([0.]))
optimizer = GD(model.parameters(), lr=eta)

# Main loop
for t in range(T):
    optimizer.zero_grad()
    loss = loss_fn(y[t:t+1].float(), model())
    # Update tracking variables
    Sum_loss_sequence[t] = loss.item() + Sum_loss_sequence[t-1]
    Sum_bias_sequence[t] = y[t] - model.theta.item() + Sum_bias_sequence[t-1]
    Sum_y_sequence[t] = y[t].item() + Sum_y_sequence[t-1]
    Sum_y_2_sequence[t] = y[t].item()**2 + Sum_y_2_sequence[t-1]
    Regret_sequence[t] = Sum_loss_sequence[t] - 0.5 * (Sum_y_2_sequence[t]-Sum_y_sequence[t])

    # Update theta
    loss.backward()
    optimizer.step()

# Plotting
sns.set_style("whitegrid")
plt.figure(figsize=(9.5, 3.5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, T+1), Regret_sequence / np.arange(1, T+1), label='Regret / T', color='k')
plt.plot(np.arange(1, T+1), np.zeros(T), color='#EBB901', linestyle='dashed', label='Zero line')
plt.xlabel('Iteration T')
plt.ylabel('Regret / T')

plt.subplot(1, 2, 2)
plt.plot(np.arange(1, T+1), np.abs(Sum_bias_sequence / np.arange(1, T+1)), label='Norm of bias', color='k')
plt.plot(np.arange(1, T+1), np.zeros(T), color='#EBB901', linestyle='dashed', label='Zero line')
plt.xlabel('Iteration T')
plt.ylabel('Norm of bias')

plt.tight_layout()
os.makedirs('./figures', exist_ok=True)
plt.savefig('figures/regret_and_bias.pdf')

# Second plot
x = np.linspace(-1.1, 1.1, 1000)
theta = (2/3)**np.arange(10)

phi = np.linspace(np.pi/2, np.pi, 100)
xx = np.array([])
yy = np.array([])
for i in range(len(theta)-1):
    xx = np.concatenate((xx, theta[i] + np.cos(phi) * (theta[i] - theta[i+1])))
    yy = np.concatenate((yy, theta[i+1] + np.sin(phi) * (theta[i] - theta[i+1])))

plt.figure()
plt.plot(x, np.abs(x), color='black')
plt.plot(xx, yy, color='gray', zorder=1)
plt.scatter(theta, np.abs(theta), edgecolor='#EBB901', facecolor='#FED332B3', zorder=2)

# Style the plot
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['left'].set_position('center')
plt.gca().spines['bottom'].set_color('gray')
plt.gca().spines['bottom'].set_position('zero')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().set_aspect('equal')

# Remove ticks
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.tight_layout()
plt.savefig('figures/nr_notimplies_lrs.pdf', bbox_inches='tight')

# Third plot
x = np.linspace(-1.1, 1.1, 1000)
theta = -0.5*(-1)**np.arange(1, 11) - (-2/3)**np.arange(3, 13)

plt.figure()
plt.plot(x, np.abs(x), color='black')
plt.plot(theta, np.abs(theta), color='gray', zorder=1)
plt.scatter(theta, np.abs(theta), edgecolor='#EBB901', facecolor='#FED332B3', zorder=2)

plt.gca().spines['left'].set_color('gray')
plt.gca().spines['left'].set_position('center')
plt.gca().spines['bottom'].set_color('gray')
plt.gca().spines['bottom'].set_position('zero')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().set_aspect('equal')

# Remove ticks
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.tight_layout()
plt.savefig('figures/lrs_notimplies_nr.pdf', bbox_inches='tight')
