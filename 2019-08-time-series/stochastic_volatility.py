import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta, AutoMultivariateNormal
from pyro.infer import SVI, JitTraceEnum_ELBO, Trace_ELBO
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.util import predictive, initialize_model
from pyro.optim import Adam

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)


"""
Multivariate Stochastic Volatility Model
See Model 2 on page 365 in https://pdfs.semanticscholar.org/fccc/6f4ee933d4330eabf377c08f8b2650e1f244.pdf
"""
def model(data):
    """
    y = diag(exp(h_t / 2)) * eps_t
    eps ~ Q L_eps rho_t
    h_{i+1} = mu + Phi(h_t - mu) + eta

    We do this in log space to convert multiplicative noise to additive noise
    so we can leverage the GaussianHMM distribution.

    log y_kt = h_kt / 2 + log <L_eps, delta_t>
            ~= h_kt / 2 + gamma_kt where gamma ~ MVN(0. sigma)
    and we moment match to compute epsilon.

    :param data: Tensor of the shape ``(securities, timesteps, returns)``
    :type data: torch.Tensor
    """
    obs_dim = data.shape[-1]
    hidden_dim = obs_dim
    with pyro.plate(len(data)):
        mu = pyro.param('mu', torch.zeros(hidden_dim))
        L = pyro.param('L', torch.eye(hidden_dim), constraint=constraints.lower_cholesky)
        init_dist = dist.MultivariateNormal(mu, scale_tril=L)

        L_eta = pyro.param('L_eta', torch.eye(hidden_dim), constraint=constraints.lower_cholesky)
        mu_eta = torch.zeros(hidden_dim)
        trans_matrix = pyro.param('phi', torch.ones(hidden_dim))
        # this gives us a zero matrix with phi on the diagonal
        trans_matrix = trans_matrix.diag()
        trans_dist = dist.MultivariateNormal(mu_eta, scale_tril=L_eta)

        mu_gamma  = pyro.param('mu_gamma', torch.zeros(obs_dim))
        L_gamma = pyro.param('sigma_gamma', torch.eye(obs_dim), constraint=constraints.lower_cholesky)
        obs_matrix = torch.eye(hidden_dim, obs_dim)
        # latent state is h_t - mu
        obs_dist = dist.MultivariateNormal(-mu_gamma, scale_tril=L_gamma)

        pyro.sample('obs', dist.GaussianHMM(init_dist, trans_matrix, trans_dist, obs_matrix, obs_dist), obs=data)


def hmc_model(data):
    """
    y = diag(exp(h_t / 2)) * eps_t
    eps ~ Q L_eps rho_t
    h_{i+1} = mu + Phi(h_t - mu) + eta

    We do this in log space to convert multiplicative noise to additive noise
    so we can leverage the GaussianHMM distribution.

    log y_kt = h_kt / 2 + log <L_eps, delta_t>
            ~= h_kt / 2 + gamma_kt where gamma ~ MVN(0. sigma)
    and we moment match to compute epsilon.

    :param data: Tensor of the shape ``(securities, timesteps, returns)``
    :type data: torch.Tensor
    """
    one = torch.ones(1)
    hidden_dim = 4
    obs_dim = data.shape[-1]
    with pyro.plate(len(data)):
        mu = pyro.sample('mu', dist.Normal(torch.zeros(hidden_dim), torch.ones(hidden_dim)))
        theta = pyro.sample("theta", dist.HalfCauchy(torch.ones(hidden_dim)))
        L = pyro.sample('L', dist.LKJCorrCholesky(hidden_dim, one))
        L = torch.mm(torch.diag(theta.sqrt()), L)
        init_dist = dist.MultivariateNormal(mu, scale_tril=L)

        mu_eta = torch.zeros(hidden_dim)
        theta_eta = pyro.sample("theta_eta", dist.HalfCauchy(torch.ones(hidden_dim)))
        L_eta = pyro.sample('L_eta', dist.LKJCorrCholesky(hidden_dim, one))
        L_eta = torch.mm(torch.diag(theta_eta.sqrt()), L_eta)
        trans_matrix = pyro.sample('phi', dist.Normal(torch.zeros(hidden_dim), torch.ones(hidden_dim)))
        # this gives us a zero matrix with phi on the diagonal
        trans_matrix = trans_matrix.diag()
        trans_dist = dist.MultivariateNormal(mu_eta, scale_tril=L_eta)

        mu_gamma = pyro.sample('mu_gamma', dist.Normal(torch.zeros(obs_dim), torch.ones(obs_dim)))
        theta_gamma = pyro.sample("theta_gamma", dist.HalfCauchy(torch.ones(obs_dim)))
        L_gamma = pyro.sample('sigma_gamma', dist.LKJCorrCholesky(obs_dim, one))
        L_gamma = torch.mm(torch.diag(theta_gamma.sqrt()), L_gamma)
        obs_matrix = torch.eye(hidden_dim, obs_dim)
        # latent state is h_t - mu
        obs_dist = dist.MultivariateNormal(-mu_gamma, scale_tril=L_gamma)

        pyro.sample('obs', dist.GaussianHMM(init_dist, trans_matrix, trans_dist, obs_matrix, obs_dist), obs=data)


def sequential_model(num_samples=1, timesteps=100, hidden_dim=4, obs_dim=4):
    """
    Generate data of shape: (samples, timesteps, obs_dim)
    where the generative model is defined by:
        y = exp(h/2) * eps
        h_{t+1} = mu + Phi (h_t - mu) + eta_t
    where eps and eta are sampled iid from a MVN distribution
    """
    ys = []
    mu = torch.zeros(hidden_dim)
    cov = 0.2 * torch.eye(hidden_dim, hidden_dim)
    mu_obs = torch.zeros(obs_dim)
    cov_obs = 0.2 * torch.eye(obs_dim, obs_dim)
    transition = 0.4 * torch.randn(hidden_dim, hidden_dim)
    # this is to generate data as the way model 2 does
    # we would use the entire transition matrix for model 3
    transition = transition.diag().diag().expand(num_samples, -1, -1)
    trans_dist = dist.MultivariateNormal(mu, cov).expand((num_samples,))
    obs_dist = dist.MultivariateNormal(mu_obs, cov_obs).expand((num_samples,))
    z = torch.zeros(num_samples, hidden_dim)
    obs = torch.eye(hidden_dim, obs_dim)

    for i in range(timesteps):
        trans_noise = pyro.sample('trans_noise', trans_dist)
        z = z.unsqueeze(1).bmm(transition).squeeze(1) + trans_noise
        # add observation noise
        obs_noise = pyro.sample('obs_noise', obs_dist)
        y = z @ obs + obs_noise
        ys.append(y)
    data = torch.stack(ys, 1)
    assert data.shape == (num_samples, timesteps, obs_dim)
    return data


def main(args):
    pyro.enable_validation(True)
    pyro.set_rng_seed(123)
    # generate synthetic data
    data = sequential_model()
    logging.debug(data.shape)
    if args.mcmc:
        logging.info("Using MCMC")
        init_params, potential_fn, transforms, _ = initialize_model(hmc_model, model_args=(data,))
        nuts_kernel = NUTS(potential_fn=potential_fn, step_size=args.learning_rate)
        mcmc = MCMC(nuts_kernel,
                    num_samples=args.num_epochs,
                    warmup_steps=2000,
                    num_chains=1,
                    initial_params=init_params,
                    transforms=transforms)
        mcmc.run(data)
        logging.info(mcmc.diagnostics())
    else:
        logging.info("Using SVI")
        # MAP estimation
        guide = AutoDelta(model)
        svi = SVI(model, guide, Adam({'lr': args.learning_rate}), Trace_ELBO())
        for i in range(args.num_epochs):
            loss = svi.step(data)
            logging.info(loss)
    for k, v in pyro.get_param_store().items():
        print(k, v)


if __name__ == "__main__":
    assert pyro.__version__.startswith('0.4.0')
    parser = argparse.ArgumentParser(description="Stochastic volatility")
    parser.add_argument("-n", "--num-epochs", default=200, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-2, type=float)
    parser.add_argument("--mcmc", default=False, action="store_true", help="whether to use MCMC (default SVI)")
    args = parser.parse_args()
    main(args)
