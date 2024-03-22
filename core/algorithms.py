import os
import numpy as np
import torch
import torch.nn as nn
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

# Define the Viscosity Gradient Descent (VGD) optimizer
class VGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, viscosity=0.9):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if viscosity < 0 or viscosity > 1:
            raise ValueError("Invalid viscosity value: {}".format(viscosity))

        defaults = dict(lr=lr, viscosity=viscosity)
        super(VGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if 'prev_iterate' not in self.state[p]:
                    self.state[p]['prev_iterate'] = p.data

                prev_iterate = self.state[p]['prev_iterate']
                self.state[p]['prev_iterate'] = p.data
                p.data = p.data * (1 - group['viscosity']) + prev_iterate * group['viscosity'] - group['lr'] * d_p

        return loss
