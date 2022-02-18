
""" Based on the DGP implementation in GPyTorch 1.5.1 docs. """

import torch, gpytorch
import numpy as np
from tqdm import trange
from gpytorch.mlls import VariationalELBO
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel, LinearKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import DeepApproximateMLL
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')

class PointLatentVariable(gpytorch.Module):
    def __init__(self, n, q):
        super().__init__()
        self.register_parameter('X', torch.nn.Parameter(torch.ones(n, q)))

    def forward(self):
        return self.X

class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, in_dim, out_dim, mean_type='constant', covar_type='linear', n_inducing=24):
        inducing_points = torch.randn(out_dim, n_inducing, in_dim)
        out_dim = torch.Size([out_dim])

        q_u = CholeskyVariationalDistribution(n_inducing, out_dim)
        q_f = VariationalStrategy(self, inducing_points, q_u, learn_inducing_locations=True)

        super(ToyDeepGPHiddenLayer, self).__init__(q_f, in_dim, int(out_dim[0]))

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=out_dim)
        else:
            self.mean_module = LinearMean(in_dim)

        if covar_type == 'linear':
            self.covar_module = LinearKernel(batch_shape=out_dim, ard_num_dims=in_dim)
        else:
            kernel = RBFKernel(batch_shape=out_dim, ard_num_dims=in_dim)
            kernel = ScaleKernel(kernel, batch_shape=out_dim, ard_num_dims=None)
            self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        if len(other_inputs):
            if isinstance(x, MultitaskMultivariateNormal):
                x = x.rsample()
            n_smp = gpytorch.settings.num_likelihood_samples.value()
            processed_inputs = [inp.unsqueeze(0).expand(n_smp, *inp.shape) for inp in other_inputs]
            x = torch.cat([x] + processed_inputs, dim=-1)
        return super().__call__(x, are_samples=bool(len(other_inputs)))

class DeepGP(DeepGP):
    def __init__(self, q, d):
        hidden_layer = ToyDeepGPHiddenLayer(q, q, 'constant', 'linear')
        last_layer = ToyDeepGPHiddenLayer(q, d, 'constant', 'rbf')

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=d)

    def forward(self, inputs):
        hidden = self.hidden_layer(inputs)
        output = self.last_layer(hidden)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)

if __name__ == '__main__':

    np.random.seed(42)

    n, q = 500, 2
    X = torch.tensor(np.random.normal(size=(n, q))).float()
    Y = np.vstack([
        0.1 * (X[:, 0] + X[:, 1])**2 - 3.5,
        0.01 * (X[:, 0] + X[:, 1])**3,
        2 * np.sin(0.5*(X[:, 0] + X[:, 1])),
        2 * np.cos(0.5*(X[:, 0] + X[:, 1])),
        4 - 0.1*(X[:, 0] + X[:, 1])**2,
        1 - 0.01*(X[:, 0] + X[:, 1])**3,
    ]).T

    Y = torch.tensor(np.random.normal(Y, 0.1)).float()
    d = len(Y.T)

    model = DeepGP(q, d)
    latent_var = PointLatentVariable(n, q)

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.01),
        dict(params=latent_var.parameters(), lr=0.01)
    ])

    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, n))

    steps = 2000; losses = []
    iterator = trange(steps, leave=False)
    for i in iterator:
        optimizer.zero_grad()
        batch_idx = np.random.choice(np.arange(n), 200)
        x_batch = latent_var()[batch_idx, :]
        # need to add kl div of prior vs variational here if not point latent variable
        loss = -mll(model(x_batch), Y[batch_idx, :])
        losses.append(loss.item())
        iterator.set_description(
            '-MLL:' + str(np.round(loss.item(), 2)) + '; ' + \
            'Step:' + str(i))
        loss.backward()
        optimizer.step()

    plt.plot(losses[800:])

    Y_recon = model(latent_var()).mean.mean(axis=0).detach()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y[:, 0])
    ax.scatter(X[:, 0], X[:, 1], Y_recon[:, 0])
