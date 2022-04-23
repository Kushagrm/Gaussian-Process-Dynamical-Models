#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKI Example 

"""

import torch, gpytorch
import numpy as np
import matplotlib.pylab as plt
from models.bayesianGPLVM import BayesianGPLVM
from gpytorch.models import ApproximateGP
from tqdm import trange
from models.latent_variable import PointLatentVariable, MAPLatentVariable
from models.latent_variable import VariationalLatentVariable, VariationalDenseLatentVariable
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel, LinearKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from torch.utils.data import TensorDataset, DataLoader

class PointLatentVariable(gpytorch.Module):
    def __init__(self, n, q):
        super().__init__()
        self.register_parameter('X', torch.nn.Parameter(torch.ones(n, q)))

    def forward(self):
        return self.X

class LatentDynamicModel(BayesianGPLVM):
     def __init__(self, n, data_dim, latent_dim, n_inducing, X, nn_layers=None):
         
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        
        # Locations Z corresponding to u_{d}, they can be randomly initialized or 
        # regularly placed with shape (n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(n_inducing, latent_dim)
    
        # Sparse Variational Formulation
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
        super(LatentDynamicModel, self).__init__(X, q_f)
        
        # Kernel 
        #self.mean_module = ConstantMean(ard_num_dims=latent_dim)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

     def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist
    
     def _get_batch_idx(self, batch_size):
            
         valid_indices = np.arange(self.n)
         batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
         return np.sort(batch_indices)
           
if __name__ == '__main__':

    np.random.seed(42)
    t = torch.linspace(0,10,200) + torch.randn(200)*1e-4
    
    rbf = gpytorch.kernels.RBFKernel(ard_num_dims=1)
    rbf.lengthscale = 1.0
    norm_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(200), covariance_matrix=rbf(t,t).evaluate() + torch.eye(200)*1e-5)
    X = norm_dist.sample_n(2).T
    
    plt.plot(t, X)
    
    n, q = 200, 2

    Y = torch.vstack([
        0.1 * (X[:,0] + X[:,1])**2 - 3.5,
        0.01 * (X[:, 0] + X[:, 1])**3,
        2 * np.sin(0.5*(X[:, 0] + X[:,1])),
        2 * np.cos(0.5*(X[:, 0] + X[:,1])),
        4 - 0.1*(X[:, 0] + X[:, 1])**2]).T
    
    d = Y.shape[1]

    ## Declare model and latent-variable class
    X_prior_mean = torch.zeros(n, q)  # shape: N x Q
    X_init = torch.nn.Parameter(torch.zeros(n, 2))

    #prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
    prior_x = MultivariateNormalPrior(loc=torch.zeros(200), covariance_matrix=rbf(t,t).evaluate() + torch.eye(200)*1e-5)
    latent_var = VariationalDenseLatentVariable(X_init, prior_x, d)
    #latent_var = PointLatentVariable(n, q)
    
    #latent_var = MAPLatentVariable(X_init, prior_x)
    model = LatentDynamicModel(n=200, data_dim=d, latent_dim=q, n_inducing=100, X=latent_var)
    
    likelihood = GaussianLikelihood()
    elbo = VariationalELBO(likelihood, model, num_data=len(Y))

    optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()}
    ], lr=0.001)

    # Model params
    model.get_trainable_param_names()

    loss_list = []
    #iterator = trange(steps_per_model[model_name], leave=True)
    iterator = trange(10000)
    batch_size = 100
    for i in iterator: 
        batch_index = model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        #sample = latent_var.X
        sample_batch = model.sample_latent_variable(batch_index)  # a full sample returns latent x across all N
        #sample_batch = sample[batch_index]
        output_batch = model(sample_batch)
        loss = -elbo(output_batch, Y[batch_index].T).sum()
        loss_list.append(loss.item())
        iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
        loss.backward()
        optimizer.step()
        
    ## Check latents
    plt.scatter(X[:,0], X[:,1])
    plt.scatter(latent_var.X.detach()[:,0], latent_var.X.detach()[:,1])