#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import torch, gpytorch
from tqdm import trange

from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.models import ApproximateGP

import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')

class PoissonLikelihood(_OneDimensionalLikelihood):
    def __init__(self):
        super().__init__()

    def forward(self, function_samples, **kwargs):
        return base_distributions.Poisson(rate=function_samples.exp())

class PointLatentVariable(gpytorch.Module):
    def __init__(self, n, latent_dim):
        super().__init__()
        self.register_parameter('X', torch.nn.Parameter(torch.ones(n, latent_dim)))

    def forward(self):
        return self.X

class GPLVM(ApproximateGP):
    def __init__(self, n, latent_dim, data_dim, n_inducing):
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=(data_dim,)) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        super(GPLVM, self).__init__(q_f)

        self.intercept = ConstantMean(batch_shape=(data_dim,))
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

    def forward(self, X):
        return MultivariateNormal(self.intercept(X), self.covar_module(X))

def train(gplvm, likelihood, X_latent, Y, steps=1000):

    elbo_func = VariationalELBO(likelihood, gplvm, num_data=n)
    optimizer = torch.optim.Adam([
        dict(params=gplvm.parameters(), lr=0.01),
        dict(params=likelihood.parameters(), lr=0.01),
        dict(params=X_latent.parameters(), lr=0.01)
    ])

    losses = []; iterator = trange(steps, leave=False)
    for i in iterator:
        optimizer.zero_grad()
        loss = -elbo_func(gplvm(X_latent()), Y.T).sum()
        losses.append(loss.item())
        iterator.set_description('-elbo: ' + str(np.round(loss.item(), 2)) + '. Step: ' + str(i))
        loss.backward()
        optimizer.step()

    return losses

n, q, d, m = 2000, 2, 6, 50

X = np.random.normal(size=(n, q))

Y = np.vstack([
    0.1 * (X[:, 0] + X[:, 1])**2 - 3.5,
    0.01 * (X[:, 0] + X[:, 1])**3,
    2 * np.sin(0.5*(X[:, 0] + X[:, 1])),
    2 * np.cos(0.5*(X[:, 0] + X[:, 1])),
    4 - 0.1*(X[:, 0] + X[:, 1])**2,
    1 - 0.01*(X[:, 0] + X[:, 1])**3,
]).T

Y -= Y.min(axis=0)
Y = torch.tensor(np.random.poisson(lam=Y))

gplvm = GPLVM(n, q, d, m)
likelihood = PoissonLikelihood()
X_latent = PointLatentVariable(n, q)

losses = train(gplvm=gplvm, X_latent=X_latent, likelihood=likelihood, Y=Y, steps=10000)

plt.plot(losses)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y[:, 0])
ax.scatter(X[:, 0], X[:, 1], gplvm(torch.tensor(X_latent()).float()).loc.detach().T[:, 0].exp())
ax.set_zlim(0, 7)