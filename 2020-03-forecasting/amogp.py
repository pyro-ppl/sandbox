# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import uuid

import numpy as np
import math
import pandas as pd

import torch
import torch.distributions.constraints as constraints
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.contrib.forecast import ForecastingModel, backtest, HMCForecaster, Forecaster
from pyro.contrib.timeseries import IndependentMaternGP
from pyro.nn import PyroParam, PyroSample
from pyro.infer.reparam import SymmetricStableReparam, LatentStableReparam, StudentTReparam
from pyro.infer.reparam import LinearHMMReparam
from pyro.poutine import reparam
from pyro.distributions import MultivariateNormal, LogNormal, Uniform, StudentT, Stable, Normal, Gamma

from os.path import exists
from urllib.request import urlopen

import pickle
from logger import get_logger

root_two = math.sqrt(2.0)


class IndependentMaternStableProcess(IndependentMaternGP):
    """
    A IndependentMaternGP with symmetric stable observation noise
    or symmetric stable transition noise.
    """
    def __init__(self, nu=1.5, dt=1.0, obs_dim=1,
                 length_scale_init=None, kernel_scale_init=None,
                 obs_noise_scale_init=None, obs_noise="student"):
        self.obs_noise = obs_noise
        assert obs_noise in ['gaussian', 'student', 'stable']
        super().__init__(nu=nu, dt=dt, obs_dim=obs_dim,
                         length_scale_init=length_scale_init,
                         kernel_scale_init=kernel_scale_init,
                         obs_noise_scale_init=obs_noise_scale_init)
        if obs_noise == "student":
           self.nu = PyroParam(torch.tensor(5.0), constraint=constraints.interval(1.01, 30.0))
        elif obs_noise == "stable":
           self.stability = PyroParam(torch.tensor(1.95), constraint=constraints.interval(1.01, 1.99))

    def _get_init_dist(self):
        cov = self.kernel.stationary_covariance()
        return MultivariateNormal(self.obs_matrix.new_zeros(self.obs_dim, self.kernel.state_dim), cov)

    def _get_obs_dist(self):
        scale = self.obs_noise_scale.unsqueeze(-1).unsqueeze(-1)
        if self.obs_noise == "stable":
            stability = self.stability.unsqueeze(-1).unsqueeze(-1)
            return Stable(stability, self.skew, scale=scale / root_two).to_event(1)
        elif self.obs_noise == "student":
            nu = self.nu.unsqueeze(-1).unsqueeze(-1)
            return StudentT(nu, torch.zeros(scale.shape, dtype=scale.dtype), scale).to_event(1)
        else:
            return Normal(0.0, scale=scale).to_event(1)

    def get_dist(self, duration=None):
        trans_matrix, process_covar = self.kernel.transition_matrix_and_covariance(dt=self.dt)
        trans_dist = MultivariateNormal(self.obs_matrix.new_zeros(self.obs_dim, 1, self.kernel.state_dim),
                                        process_covar.unsqueeze(-3))
        trans_matrix = trans_matrix.unsqueeze(-3)
        return dist.LinearHMM(self._get_init_dist(), trans_matrix, trans_dist,
                              self.obs_matrix, self._get_obs_dist(), duration=duration)


def get_data(args):
    data = torch.tensor(np.loadtxt(args['data_dir'] + '/ercot.csv', skiprows=1, delimiter=',', usecols=(5)))
    to_keep = 1100 * 48
    data = data[:to_keep].log().unsqueeze(-1).double().cuda()

    covariates = torch.zeros(data.size(0), 0).cuda()

    return data, covariates


class Model(ForecastingModel):
    def __init__(self, obs_noise="gaussian"):
        super().__init__()
        self.obs_noise = obs_noise
        self.noise_gp = IndependentMaternStableProcess(obs_dim=1, obs_noise=obs_noise,
                                                       length_scale_init=torch.tensor([10.0]))
        if obs_noise == "gaussian":
            self.config = {"residual": LinearHMMReparam()}
        elif obs_noise == "stable":
            self.config = {"residual": LinearHMMReparam(obs=SymmetricStableReparam())}
        elif obs_noise == "student":
            self.config = {"residual": LinearHMMReparam(obs=StudentTReparam())}

    def model(self, zero_data, covariates):
        noise_dist = self.noise_gp.get_dist(duration=zero_data.size(-2))
        noise_dist = dist.IndependentHMM(noise_dist)

        with reparam(config=self.config):
            self.predict(noise_dist, zero_data)


class GuideConv(nn.Module):
    def __init__(self, obs_dim=1, num_channels=4, kernel_size=8, hidden_dim=16, num_layers=2, cat_obs=False, gamma_variates=4, distribution="gamma"):
        super().__init__()
        self.obs_dim = obs_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.cat_obs = cat_obs
        self.gamma_variates = gamma_variates
        self.distribution = distribution
        nn_dim = num_channels if not cat_obs else num_channels + obs_dim
        if num_layers == 1:
            self.fc = nn.Linear(nn_dim, 2 * gamma_variates)
        elif num_layers == 2:
            self.fc = nn.Linear(nn_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 2 * gamma_variates)
        self.conv = nn.Conv1d(obs_dim, num_channels, kernel_size, stride=1, padding=kernel_size - 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.conv(x.t().unsqueeze(0))[:, :, :-(self.kernel_size - 1)].squeeze(0).t()
        if self.cat_obs:
            h = torch.cat([x, h], dim=-1)
        if self.num_layers == 1:
            h = self.fc(self.relu(h))
        else:
            h = self.fc2(self.relu(self.fc(self.relu(h))))
        if self.distribution == 'gamma':
            h = nn.functional.softplus(h)
            return h[:, :self.gamma_variates].clamp(min=0.1), h[:, self.gamma_variates:].clamp(min=0.1)
        else:
            return h[:, :self.gamma_variates], nn.functional.softplus(h[:, self.gamma_variates:]).clamp(min=0.01)


def main(**args):
    log_file = 'amogp.{}.tt_{}_{}.{}.log'
    log_file = log_file.format(args['obs_noise'],
                               args['train_window'], args['test_window'],
                               str(uuid.uuid4())[0:4])

    log = get_logger(args['log_dir'], log_file, use_local_logger=False)

    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    log(args)
    log("")

    pyro.set_rng_seed(args['seed'])

    results = {}

    data, covariates = get_data(args)
    print("data, covariates", data.shape, covariates.shape)

    guide_conv = None if args['obs_noise'] == 'gaussian' else GuideConv(num_channels=args['num_channels'], kernel_size=args['kernel_size'],
                                                                        hidden_dim=args['hidden_dim'], num_layers=2,
                                                                        cat_obs=False,
                                                                        gamma_variates=1,
                                                                        distribution='gamma').cuda()

    def amortized_guide(data, covariates, obs_dim=4):
        pyro.module("guide_conv", guide_conv)
        alpha, beta = guide_conv(data)
        pyro.sample("residual_obs_gamma", Gamma(alpha.unsqueeze(0), beta.unsqueeze(0)).to_event(3))

    def svi_forecaster_options(t0, t1, t2):
        num_steps = args['num_steps']  if t1 == args['train_window'] else 0
        lr = args['learning_rate']  #if t1 == args.train_window else 0.1 * args.learning_rate
        lrd = args['learning_rate_decay']  #if t1 == args.train_window else 0.1
        return {"num_steps": num_steps, "learning_rate": lr,
                "learning_rate_decay": lrd, "log_every": 20,
                "dct_gradients": False, "warm_start": False,
                "clip_norm": args['clip_norm'],
                "vectorize_particles": False,
                "num_particles": 1,
                "guide": None if args['obs_noise'] == 'gaussian' else amortized_guide}

    metrics = backtest(data, covariates,
                       lambda: Model(obs_noise=args['obs_noise']),
                       train_window=None,
                       min_train_window=args['train_window'],
                       test_window=args['test_window'],
                       stride=args['stride'],
                       batch_size=100,
                       amortized=True,
                       num_samples=args['num_eval_samples'],
                       forecaster_options=svi_forecaster_options)

    log("### EVALUATION ###")
    for name in ["mae", "rmse", "crps"]:
        values = [m[name] for m in metrics]
        mean, std = np.mean(values), np.std(values)
        results[name] = mean
        results[name + '_std'] = std
        log("{} = {:0.4g} +- {:0.4g}".format(name, mean, std))

    #print("shape", metrics[0]['pred'].shape)
    #results['pred'] = metrics[0]['pred'].data.cpu().numpy()[:, 0, :, 0]

    with open(args['log_dir'] + '/' + log_file[:-4] + '.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multivariate timeseries models")
    parser.add_argument("--obs-noise", default='gaussian', type=str,
                        choices=['gaussian', 'stable', 'student'])
    parser.add_argument("--train-window", default=1100 * 48 - 1600, type=int)
    parser.add_argument("--test-window", default=5, type=int)
    parser.add_argument("--stride", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('-ld', '--log-dir', type=str, default="./logs/")
    parser.add_argument('--data-dir', type=str, default="./data/")
    parser.add_argument("--num-channels", default=8, type=int)
    parser.add_argument("--kernel_size", default=8, type=int)
    parser.add_argument("--hidden-dim", default=32, type=int)
    parser.add_argument("--num-eval-samples", default=200, type=int)
    parser.add_argument("--clip-norm", default=10.0, type=float)
    parser.add_argument("-n", "--num-steps", default=101, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.01, type=float)
    args = parser.parse_args()

    main(**vars(args))