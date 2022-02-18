#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Variable class with sub-classes that determine type of inference for the latent variable

"""
import gpytorch
import torch
from torch import nn
from torch.distributions import kl_divergence
from gpytorch.mlls.added_loss_term import AddedLossTerm
import torch.nn.functional as F
import numpy as np
from models.partial_gaussian import PointNet

class LatentVariable(gpytorch.Module):
    
    """
    :param n (int): Size of the latent space.
    :param latent_dim (int): Dimensionality of latent space.

    """

    def __init__(self, n, dim):
        super().__init__()
        self.n = n
        self.latent_dim = dim
        
    def forward(self, x):
        raise NotImplementedError
        
    def reset(self):
         raise NotImplementedError
        
class PointLatentVariable(LatentVariable):
    def __init__(self, X_init):
        n, latent_dim = X_init.shape
        super().__init__(n, latent_dim)
        self.register_parameter('X', X_init)

    def forward(self, batch_idx=None):        
        if batch_idx is None:
            batch_idx = np.arange(self.n)
        return self.X[batch_idx, :]
    
    def reset(self, X_init_test):
        self.__init__(X_init_test)
        
class MAPLatentVariable(LatentVariable):
    
    def __init__(self, X_init, prior_x):
        n, latent_dim = X_init.shape
        super().__init__(n, latent_dim)
        self.prior_x = prior_x
        
        if torch.cuda.is_available():
            self.prior_x = prior_x
            
        self.register_parameter('X', X_init)
        self.register_prior('prior_x', self.prior_x, 'X')

    def forward(self, batch_idx=None):        
        if batch_idx is None:
            batch_idx = np.arange(self.n)
        return self.X[batch_idx, :]
    
    def reset(self, X_init_test, prior_x_test):
        self.__init__(X_init_test, prior_x_test)

class NNEncoder(LatentVariable):    
    def __init__(self, n, latent_dim, prior_x, data_dim, layers):
        super().__init__(n, latent_dim)
        self.prior_x = prior_x
        self.data_dim = data_dim
        self.latent_dim = latent_dim

        self._init_mu_nnet(layers)
        self._init_sg_nnet(len(layers))
        self.register_added_loss_term("x_kl")

        jitter = torch.eye(latent_dim).unsqueeze(0)*1e-5
        self.jitter = torch.cat([jitter for i in range(n)], axis=0)
        
        #if torch.cuda.is_available():
            
        #    self.jitter = self.jitter.cuda()

    def _get_mu_layers(self, layers):
        return (self.data_dim,) + layers + (self.latent_dim,)

    def _init_mu_nnet(self, layers):
        layers = self._get_mu_layers(layers)
        n_layers = len(layers)

        self.mu_layers = nn.ModuleList([ \
            nn.Linear(layers[i], layers[i + 1]) \
            for i in range(n_layers - 1)])

    def _get_sg_layers(self, n_layers):
        n_sg_out = self.latent_dim**2
        n_sg_nodes = (self.data_dim + n_sg_out)//2
        sg_layers = (self.data_dim,) + (n_sg_nodes,)*n_layers + (n_sg_out,)
        return sg_layers

    def _init_sg_nnet(self, n_layers):
        layers = self._get_sg_layers(n_layers)
        n_layers = len(layers)

        self.sg_layers = nn.ModuleList([ \
            nn.Linear(layers[i], layers[i + 1]) \
            for i in range(n_layers - 1)])

    def mu(self, Y):
        mu = torch.tanh(self.mu_layers[0](Y))
        for i in range(1, len(self.mu_layers)):
            mu = torch.tanh(self.mu_layers[i](mu))
            if i == (len(self.mu_layers) - 1): mu = mu * 5
        return mu        

    def sigma(self, Y):
        sg = torch.tanh(self.sg_layers[0](Y))
        for i in range(1, len(self.sg_layers)):
            sg = torch.tanh(self.sg_layers[i](sg))
            if i == (len(self.sg_layers) - 1): sg = sg * 5

        sg = sg.reshape(len(sg), self.latent_dim, self.latent_dim)
        sg = torch.einsum('aij,akj->aik', sg, sg)
        return sg + self.jitter[0:sg.shape[0],:,:]

    def forward(self, Y, batch_idx=None):
        mu = self.mu(Y)
        sg = self.sigma(Y)

        if batch_idx is None:
            batch_idx = np.arange(self.n)

        mu = mu[batch_idx, ...]
        sg = sg[batch_idx, ...]

        q_x = torch.distributions.MultivariateNormal(mu, sg)

        prior_x = self.prior_x
        prior_x.loc = prior_x.loc[:len(batch_idx), ...]
        prior_x.covariance_matrix = prior_x.covariance_matrix[:len(batch_idx), ...]

        x_kl = kl_gaussian_loss_term(q_x, self.prior_x, len(batch_idx), self.data_dim)
        self.update_added_loss_term('x_kl', x_kl)
        return q_x.rsample()

class VariationalLatentVariable(LatentVariable):
    
    def __init__(self, X_init, prior_x, data_dim):
        n, latent_dim = X_init.shape
        super().__init__(n, latent_dim)
        
        self.data_dim = data_dim
        self.prior_x = prior_x
        # G: there might be some issues here if someone calls .cuda() on their BayesianGPLVM
        # after initializing on the CPU

        # Local variational params per latent point with dimensionality latent_dim
        self.q_mu = torch.nn.Parameter(X_init) # (.cuda())
        self.q_log_sigma = torch.nn.Parameter(torch.randn(n, latent_dim))    # .cuda()
        
        #if torch.cuda.is_available():
        #    
        #    self.q_mu = self.q_mu.cuda()
        #    self.q_log_sigma = self.q_log_sigma.cuda()
        
        # This will add the KL divergence KL(q(X) || p(X)) to the loss
        self.register_added_loss_term("x_kl")

    def forward(self, batch_idx=None):
        
        if batch_idx is None:
            batch_idx = np.arange(self.n) 
        
        q_mu_batch = self.q_mu[batch_idx, ...]
        q_log_sigma_batch = self.q_log_sigma[batch_idx, ...]

        q_x = torch.distributions.Normal(q_mu_batch, q_log_sigma_batch.exp())

        self.prior_x.loc = self.prior_x.loc[:len(batch_idx), ...]
        self.prior_x.scale = self.prior_x.scale[:len(batch_idx), ...]
        x_kl = kl_gaussian_loss_term(q_x, self.prior_x, len(batch_idx), self.data_dim)        
        self.update_added_loss_term('x_kl', x_kl)
        return q_x.rsample()
    
    def reset(self, X_init_test, prior_x_test, data_dim):
        self.__init__(X_init_test, prior_x_test, data_dim)
        
class VariationalDenseLatentVariable(LatentVariable):
    
    def __init__(self, X_init, prior_x, data_dim):
        n, latent_dim = X_init.shape
        super().__init__(n, latent_dim)
        
        self.data_dim = data_dim
        self.prior_x = prior_x
        # G: there might be some issues here if someone calls .cuda() on their BayesianGPLVM
        # after initializing on the CPU

        # Local variational params per latent point with dimensionality latent_dim
        self.q_mu = torch.nn.Parameter(X_init.cuda()) # (.cuda())
        self.q_log_sigma = torch.nn.Parameter(torch.randn(n, latent_dim**2).cuda())    # .cuda()
        
        jitter = torch.eye(latent_dim).unsqueeze(0)*1e-5
        
        if torch.cuda.is_available():
            
            self.q_mu = self.q_mu.cuda()
            self.q_log_sigma = self.q_log_sigma.cuda()
            self.jitter = torch.cat([jitter for i in range(n)], axis=0).cuda()
        
        # This will add the KL divergence KL(q(X) || p(X)) to the loss
        self.register_added_loss_term("x_kl")
        
        #jitter = torch.eye(latent_dim).unsqueeze(0)*1e-5
        #self.jitter = torch.cat([jitter for i in range(n)], axis=0)
        
    def sigma(self):
       sg = self.q_log_sigma
       sg = sg.reshape(len(sg), self.latent_dim, self.latent_dim)
       sg = torch.einsum('aij,akj->aik', sg, sg)
       sg += self.jitter
       return sg   

    def forward(self, batch_idx=None):
        
        self.q_sigma = self.sigma()
        
        if batch_idx is None:
            batch_idx = np.arange(self.n) 
        
        q_mu_batch = self.q_mu[batch_idx, ...]
        q_sigma_batch = self.q_sigma[batch_idx, ...]

        q_x = torch.distributions.MultivariateNormal(q_mu_batch, q_sigma_batch)

        self.prior_x.loc = self.prior_x.loc[:len(batch_idx), ...]
        self.prior_x.scale = self.prior_x.covariance_matrix[:len(batch_idx), ...]
        x_kl = kl_gaussian_loss_term(q_x, self.prior_x, len(batch_idx), self.data_dim)        
        self.update_added_loss_term('x_kl', x_kl)
        return q_x.rsample()
    
    def reset(self, X_init_test, prior_x_test, data_dim):
        self.__init__(X_init_test, prior_x_test, data_dim)
        
class kl_gaussian_loss_term(AddedLossTerm):
    
    def __init__(self, q_x, p_x, n, data_dim):
        self.q_x = q_x
        self.p_x = p_x
        self.n = n
        self.data_dim = data_dim
        
    def loss(self): 
        
        # G 
        kl_per_latent_dim = kl_divergence(self.q_x, self.p_x).sum(axis=0) # vector of size latent_dim
        kl_per_point = kl_per_latent_dim.sum()/self.n # scalar
        # inside the forward method of variational ELBO, 
        # the added loss terms are expanded (using add_) to take the same 
        # shape as the log_lik term (has shape data_dim)
        # so they can be added together. Hence, we divide by data_dim to avoid 
        # overcounting the kl term
        return (kl_per_point/self.data_dim)


class PointNetEncoder(LatentVariable):
    def __init__(self, n, data_dim, latent_dim, prior_x, inter_dim=5, h_dims=(5, 5), rho_dims=(5, 5)):
        super().__init__(n, latent_dim)
        
        self.data_dim = data_dim
        self.prior_x = prior_x
        self.pointnet = PointNet(latent_dim, inter_dim, h_dims=h_dims, rho_dims=rho_dims,
                 min_sigma=1e-6, init_sigma=None, nonlinearity=torch.tanh)
        self.register_added_loss_term("x_kl")

    def forward(self, Y):
        q_x = self.pointnet(Y)
        x_kl = kl_gaussian_loss_term(q_x, self.prior_x, self.n, self.data_dim)
        self.update_added_loss_term('x_kl', x_kl)  # Update the KL term
        return q_x.rsample()
    
