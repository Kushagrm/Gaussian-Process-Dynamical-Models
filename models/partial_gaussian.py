#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Matt Ashman for encoding missing data likelihoods

"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

class Likelihood(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, z):
        raise NotImplementedError

    def log_prob(self, z, x):
        px = self(z)

        return px.log_prob(x)


class LinearNN(nn.Module):
    """A fully connected neural network.

    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param hidden_dims (list, optional): dimensions of hidden layers.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64),
                 nonlinearity=F.relu):
        super().__init__()

        self.nonlinearity = nonlinearity

        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                self.layers.append(nn.Linear(in_dim, hidden_dims[i]))
            elif i == len(hidden_dims):
                self.layers.append(nn.Linear(hidden_dims[i-1], out_dim))
            else:
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))

        # Weight initialisation.
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias.data is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """Returns output of the network."""
        for layer in self.layers[:-1]:
            x = self.nonlinearity(layer(x))

        x = self.layers[-1](x)
        return x

class NNHeteroGaussian(Likelihood):
    """A fully connected neural network for parameterising a diagonal
    Gaussian distribution with heteroscedastic noise.
    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param hidden_dims (list, optional): dimensions of hidden layers.
    :param sigma (float, optional): if not None, sets the initial
    homoscedastic output sigma.
    :param sigma_grad (bool, optional): whether to train the homoscedastic
    output sigma.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param init_sigma (float, optional): sets the initial output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64),
                 min_sigma=1e-3, init_sigma=None, nonlinearity=F.relu):
        super().__init__()

        self.out_dim = out_dim
        self.min_sigma = min_sigma
        self.network = LinearNN(in_dim, 2 * out_dim, hidden_dims, nonlinearity)

        if init_sigma is not None:
            self.network.layers[-1].bias.data[out_dim:] = torch.log(
                torch.exp(torch.tensor(init_sigma)) - 1)

    def forward(self, z, *args, **kwargs):
        output = self.network(z)
        mu = output[..., :self.out_dim]
        raw_sigma = output[..., self.out_dim:]
        sigma = F.softplus(raw_sigma) + self.min_sigma
        px = Normal(mu, sigma)

        return px
    

class PointNet(Likelihood):
    """PointNet from Sparse Gaussian Process Variational Autoencoders.

    :param out_dim (int): dimension of the output variable.
    :param inter_dim (int): dimension of intermediate representation.
    :param h_dims (list, optional): dimension of the encoding function.
    :param rho_dims (list, optional): dimension of the shared function.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param init_sigma (float, optional): sets the initial output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, out_dim, inter_dim, h_dims=(64, 64), rho_dims=(64, 64),
                 min_sigma=1e-3, init_sigma=None, nonlinearity=F.relu):
        super().__init__()

        self.out_dim = out_dim
        self.inter_dim = inter_dim

        # Takes the index of the observation dimension and it's value.
        self.h = LinearNN(
            2, inter_dim, h_dims, nonlinearity)

        # Takes the aggregation of the outputs from self.h.
        self.rho = NNHeteroGaussian(
            inter_dim, out_dim, rho_dims, min_sigma, init_sigma, nonlinearity)

    def forward(self, z, mask=None):
        """Returns parameters of a diagonal Gaussian distribution."""
        out = torch.zeros(z.shape[0], z.shape[1], self.inter_dim)

        # Pass through first network.
        for dim, z_dim in enumerate(z.transpose(0, 1)):
            if mask is not None:
                idx = torch.where(mask[:, dim])[0]
                z_in = z_dim[idx].unsqueeze(1)
                z_in = torch.cat([z_in, torch.ones_like(z_in)*dim], 1)
                out[idx, dim, :] = self.h(z_in)
            else:
                z_in = z_dim.unsqueeze(1)
                z_in = torch.cat([z_in, torch.ones_like(z_in)*dim], 1)
                out[:, dim, :] = self.h(z_in)

        # Aggregation layer.
        out = torch.sum(out, 1)

        # Pass through second network.
        pz = self.rho(out)

        return pz
