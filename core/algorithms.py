import os
import numpy as np
import torch
import torch.nn as nn
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

# Define the gradient descent optimizer
class GD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, penalty_type=None, lambda_=0.0, alpha=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if penalty_type not in (None, 'L1', 'L2'):
            raise ValueError("Invalid penalty type: {}".format(penalty_type))
        if lambda_ < 0.0:
            raise ValueError("Invalid penalty level (lambda): {}".format(lambda_))
        if not (0 <= alpha <= 1.0):
            raise ValueError("Alpha must be in the range [0, 1.0].")

        # Store initial learning rate separately to apply decay formula
        self.global_step = 0
        defaults = dict(lr=lr, initial_lr=lr, penalty_type=penalty_type, lambda_=lambda_, alpha=alpha)
        super(GD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['initial_lr']
            alpha = group['alpha']

            # Decay the learning rate proportional to 1 / t^alpha
            self.global_step += 1
            decayed_lr = lr / (self.global_step ** alpha)
            group['lr'] = decayed_lr

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                # L1 or L2 regularization
                if group['penalty_type'] == 'L2':
                    d_p = d_p + group['lambda_'] * p.data
                elif group['penalty_type'] == 'L1':
                    d_p = d_p + group['lambda_'] * p.data.sign()

                p.data = p.data - group['lr'] * d_p

        return loss
    
# Define the gradient descent optimizer
class ExpGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, penalty_type=None, lambda_=0.0, alpha=0, norm=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if penalty_type not in (None, 'L1', 'L2'):
            raise ValueError("Invalid penalty type: {}".format(penalty_type))
        if lambda_ < 0.0:
            raise ValueError("Invalid penalty level (lambda): {}".format(lambda_))
        if not (0 <= alpha <= 1.0):
            raise ValueError("Alpha must be in the range [0, 1.0].")

        # Store initial learning rate separately to apply decay formula
        self.global_step = 0
        defaults = dict(lr=lr, initial_lr=lr, penalty_type=penalty_type, lambda_=lambda_, alpha=alpha, norm=norm)
        super(ExpGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['initial_lr']
            alpha = group['alpha']

            # Decay the learning rate proportional to 1 / t^alpha
            self.global_step += 1
            decayed_lr = lr / (self.global_step ** alpha)
            group['lr'] = decayed_lr

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                # L1 or L2 regularization
                if group['penalty_type'] == 'L2':
                    d_p = d_p + group['lambda_'] * p.data
                elif group['penalty_type'] == 'L1':
                    d_p = d_p + group['lambda_'] * p.data.sign()
                p.data *= torch.exp(-group['lr'] * d_p)
                if group['norm']:
                    p.data /= p.data.sum()

        return loss

# Simple debiasing model
class DebiasingModel(nn.Module):
    def __init__(self, theta0):
        super(DebiasingModel, self).__init__()
        self.d = theta0.shape[0]
        self.theta = nn.Parameter(theta0)

    def forward(self):
        return self.theta

# OLS model
class OLSModel(nn.Module):
    def __init__(self, theta0):
        super(OLSModel, self).__init__()
        self.d = theta0.shape[0]
        self.theta = nn.Parameter(theta0)

    def forward(self, x_t):
        return torch.matmul(x_t, self.theta)
    
# Logistic model
class LogisticModel(nn.Module):
    def __init__(self, theta0):
        super(LogisticModel, self).__init__()
        self.d = theta0.shape[0]
        self.theta = nn.Parameter(theta0)

    def forward(self, x_t):
        return torch.sigmoid(x_t @ self.theta)

# Ensembling quantile model
class EnsemblingModel(nn.Module):
    def __init__(self, init_weights):
        super(EnsemblingModel, self).__init__()
        self.weights = nn.Parameter(init_weights)

    def forward(self, predictors):
        return self.weights @ predictors

# Quantile tracker
class QuantileTracker(nn.Module):
    def __init__(self, init_q):
        super(QuantileTracker, self).__init__()
        self.q = nn.Parameter(torch.tensor(init_q))

    def forward(self):
        return self.q