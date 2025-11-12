from typing import Optional, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GLiPLoss(nn.Module):
    def __init__(self, gradient_penalty: float = 1., reduction: Optional[str] = 'mean'):
        super(GLiPLoss, self).__init__()
        self.gradient_penalty = gradient_penalty
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size,3:n_channels,W,H,D)
            targets: (batch_size,3:n_channels,W,H,D)
        """
        shapes = inputs.shape[2:]

        # compute gradient penalty
        gradient_penalty = 0
        for i, s in enumerate(shapes):
            gradient_penalty += self.gradient_penalty * (torch.abs(torch.narrow(inputs, i+2, 1, s-1) - torch.narrow(inputs, i+2, 0, s-1)) - 1/math.sqrt(len(shapes))).pow(2).mean([j+2 for j in range(len(shapes))]) #(bs,nc)

        # compute wasserstein loss
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1) #(bs,nc,W*H*D)
        targets = targets.reshape(targets.shape[0], targets.shape[1], -1) #(bs,nc,W*H*D)
        wasserstein_loss = - (inputs * targets).sum(2) / targets.sum([0,2]) \
            + (inputs * (1 - targets)).sum(2) / (1 - targets).sum([0,2]) #(bs,nc)

        # compute loss
        loss = wasserstein_loss + gradient_penalty
        
        # reduce loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss