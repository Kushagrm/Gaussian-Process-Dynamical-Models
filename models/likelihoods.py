
from torch import masked_fill
from gpytorch.likelihoods import GaussianLikelihood
from torch.distributions import Normal

class GaussianLikelihoodWithMissingObs(GaussianLikelihood):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_masked_obs(x):
        missing_idx = x.isnan()
        x_masked = x.masked_fill(missing_idx, -999.)
        return missing_idx, x_masked

    def expected_log_prob(self, target, input, *params, **kwargs):
        missing_idx, target = self._get_masked_obs(target)
        res = super().expected_log_prob(target, input, *params, **kwargs)
        return res * ~missing_idx

    def log_marginal(self, observations, function_dist, *params, **kwargs):
        missing_idx, observations = self._get_masked_obs(observations)
        res = super().log_marginal(observations, function_dist, *params, **kwargs)
        return res * ~missing_idx

if __name__ == '__main__':

    import torch
    import numpy as np
    from tqdm import trange
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.constraints import Interval
    torch.manual_seed(42)

    mu = torch.zeros(2, 3)
    sigma = torch.tensor([[
            [ 1,  1-1e-7, -1+1e-7],
            [ 1-1e-7,  1, -1+1e-7],
            [-1+1e-7, -1+1e-7,  1] ]]*2).float()
    mvn = MultivariateNormal(mu, sigma)
    x = mvn.sample_n(10000)
    x[np.random.binomial(1, 0.5, size=x.shape).astype(bool)] = np.nan
    x += np.random.normal(0, 0.5, size=x.shape)

    LikelihoodOfChoice = GaussianLikelihoodWithMissingObs
    likelihood = LikelihoodOfChoice(noise_constraint=Interval(1e-6, 2))

    opt = torch.optim.Adam(likelihood.parameters(), lr=0.5)

    bar = trange(1000)
    for _ in bar:
        opt.zero_grad()
        loss = -likelihood.log_marginal(x, mvn).sum()
        loss.backward()
        opt.step()
        bar.set_description("nll: " + str(int(loss.data)))
    print(likelihood.noise.sqrt()) # Test 1

    likelihood.expected_log_prob(x[0], mvn) == likelihood.log_marginal(x[0], mvn) # Test 2