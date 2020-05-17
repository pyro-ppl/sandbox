# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import uuid

import numpy as np
import math
import pandas as pd

import torch
import torch.distributions.constraints as constraints

import pyro
import pyro.distributions as dist
from pyro.contrib.forecast import ForecastingModel, backtest, HMCForecaster, Forecaster
from pyro.contrib.timeseries import IndependentMaternGP
from pyro.nn import PyroParam, PyroSample
from pyro.infer.reparam import SymmetricStableReparam, LatentStableReparam, StudentTReparam
from pyro.infer.reparam import LinearHMMReparam
from pyro.poutine import reparam
from pyro.distributions import MultivariateNormal, LogNormal, Uniform, StudentT, Stable, Normal

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
    def __init__(self, nu=0.5, dt=1.0, obs_dim=1,
                 length_scale_init=None, kernel_scale_init=None,
                 obs_noise_scale_init=None, stable_noise="obs",
                 latent_globals=False):
        self.stable_noise = stable_noise
        self.latent_globals = latent_globals
        self.skew = 0.0
        assert stable_noise in ['obs', 'trans', 'none', 'stud']
        super().__init__(nu=nu, dt=dt, obs_dim=obs_dim,
                         length_scale_init=length_scale_init,
                         kernel_scale_init=kernel_scale_init,
                         obs_noise_scale_init=obs_noise_scale_init)
        if self.latent_globals:
            self.kernel.length_scale = PyroSample(LogNormal(1.0, 2.0))
            self.kernel.kernel_scale = PyroSample(LogNormal(0.0, 2.0))
            self.obs_noise_scale = PyroSample(LogNormal(-2.0, 2.0))
            if stable_noise in ['obs', 'trans']:
                self.stability = PyroSample(Uniform(1.01, 1.99))
            elif stable_noise == 'stud':
                self.nu = PyroSample(Uniform(1.01, 30.0))
        elif stable_noise != 'none':
           self.stability = PyroParam(torch.tensor(1.95), constraint=constraints.interval(1.01, 1.99))

    def _get_init_dist(self):
        cov = self.kernel.stationary_covariance()
        return MultivariateNormal(self.obs_matrix.new_zeros(self.obs_dim, self.kernel.state_dim), cov)

    def _get_obs_dist(self):
        scale = self.obs_noise_scale.unsqueeze(-1).unsqueeze(-1)
        if self.stable_noise == "obs":
            stability = self.stability.unsqueeze(-1).unsqueeze(-1)
            return Stable(stability, self.skew, scale=scale / root_two).to_event(1)
        elif self.stable_noise == "stud":
            nu = self.nu.unsqueeze(-1).unsqueeze(-1)
            return StudentT(nu, torch.zeros(scale.shape, dtype=scale.dtype), scale).to_event(1)
        else:
            return Normal(0.0, scale=scale).to_event(1)

    def get_dist(self, duration=None):
        trans_matrix, process_covar = self.kernel.transition_matrix_and_covariance(dt=self.dt)
        if self.stable_noise == "trans":
            assert self.kernel.nu == 0.5
            scale = process_covar.sqrt() / root_two
            #scale = process_covar.cholesky().unsqueeze(-1).unsqueeze(-1) / root_two
            stability = self.stability.unsqueeze(-1).unsqueeze(-1)
            #trans_dist = dist.TransformedDistribution(dist.Stable(stability, self.skew, scale=torch.ones(process_covar.shape[:-1])),
            #        transforms=dist.transforms.LowerCholeskyAffine(torch.zeros(scale.shape[:-1]), scale))
            trans_dist = dist.Stable(stability, self.skew, scale=scale).to_event(1)
        else:
            trans_dist = MultivariateNormal(self.obs_matrix.new_zeros(self.obs_dim, 1, self.kernel.state_dim),
                                            process_covar.unsqueeze(-3))
        trans_matrix = trans_matrix.unsqueeze(-3)
        return dist.LinearHMM(self._get_init_dist(), trans_matrix, trans_dist,
                              self.obs_matrix, self._get_obs_dist(), duration=duration)


def get_data(shock_times = [23, 61], args=None):
    data = 0.1 * torch.randn(args['train_window'] + args['test_window']).unsqueeze(-1)

    shock = 1.0 # args['shock']

    if args['mode'] == "jump":
        print("MODE JUMP")
        k = 0
        for t in shock_times:
            if k % 2 == 0:
                data[t:, 0] += shock
            else:
                data[t:, 0] -= shock
            k += 1

    covariates = torch.zeros(data.size(0), 0)

    return data, covariates


class Model(ForecastingModel):
    def __init__(self, noise_model="gaussian", latent_globals=False):
        super().__init__()
        self.noise_model = noise_model
        self.latent_globals = latent_globals
        if noise_model == "gaussian":
            self.noise_gp = IndependentMaternStableProcess(obs_dim=1, stable_noise='none',
                                                           length_scale_init=torch.tensor([10.0]),
                                                           latent_globals=latent_globals)
            self.config = {"residual": LinearHMMReparam()}
        elif noise_model == "stable-obs":
            self.noise_gp = IndependentMaternStableProcess(obs_dim=1, stable_noise='obs',
                                                           length_scale_init=torch.tensor([10.0]),
                                                           latent_globals=latent_globals)
            self.config = {"residual": LinearHMMReparam(obs=SymmetricStableReparam())}
        elif noise_model == "stud-obs":
            self.noise_gp = IndependentMaternStableProcess(obs_dim=1, stable_noise='stud',
                                                          length_scale_init=torch.tensor([10.0]),
                                                          latent_globals=latent_globals)
            self.config = {"residual": LinearHMMReparam(obs=StudentTReparam())}
        elif noise_model == "stable-trans":
            self.noise_gp = IndependentMaternStableProcess(obs_dim=1, stable_noise='trans',
                                                           length_scale_init=torch.tensor([10.0]),
                                                           latent_globals=latent_globals)
            self.config = {"residual": LinearHMMReparam(trans=SymmetricStableReparam())}

    def model(self, zero_data, covariates):
        if self.latent_globals:
            with pyro.plate("indep_gps", zero_data.size(-1), dim=-2):
                noise_dist = self.noise_gp.get_dist(duration=zero_data.size(-2))
        else:
            noise_dist = self.noise_gp.get_dist(duration=zero_data.size(-2))
        if self.noise_model in ["gaussian", "stable-obs", "stable-trans", "stud-obs"]:
            noise_dist = dist.IndependentHMM(noise_dist)

        with reparam(config=self.config):
            self.predict(noise_dist, zero_data)


def main(**args):
    log_file = '{}.shock.{:.1f}.tt_{}_{}.ns_{}_{}.mtd_{}.{}.{}.log'
    log_file = log_file.format(args['noise_model'], args['shock'],
                               args['train_window'], args['test_window'],
                               args['num_warmup'], args['num_samples'],
                               args['max_tree_depth'], args['mode'],
                               str(uuid.uuid4())[0:4])

    log = get_logger(args['log_dir'], log_file, use_local_logger=False)

    torch.set_default_dtype(torch.float64)
    log(args)
    log("")

    pyro.set_rng_seed(args['seed'])

    def hmc_forecaster_options(t0, t1, t2):
        return {"max_tree_depth": args['max_tree_depth'],
                "num_warmup": args['num_warmup'],
                "num_samples": args['num_samples']}

    results = {}

    data, covariates = get_data(args=args)
    print("data, covariates", data.shape, covariates.shape)

    metrics = backtest(data, covariates,
                       lambda: Model(noise_model=args['noise_model'], latent_globals=True),
                       train_window=None,
                       min_train_window=args['train_window'],
                       test_window=args['test_window'],
                       stride=args['stride'],
                       num_samples=args['num_eval_samples'],
                       forecaster_options=hmc_forecaster_options,
                       forecaster_fn=HMCForecaster)

    log("### EVALUATION ###")
    for name in ["mae", "rmse", "crps"]:
        values = [m[name] for m in metrics]
        mean, std = np.mean(values), np.std(values)
        results[name] = mean
        results[name + '_std'] = std
        log("{} = {:0.4g} +- {:0.4g}".format(name, mean, std))

    results['pred'] = metrics[0]['pred'].data.numpy()[:, 0, :, 0]
    results['samples'] = metrics[0]['samples']

    if args['plot']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        f, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
        T = data.size(0)

        pred = metrics[0]['pred'].data.numpy()[:, 0, :, 0]
        lower, upper = np.percentile(pred, 10, axis=0), np.percentile(pred, 90, axis=0)

        ax1.plot(np.arange(T), data[:, 0], 'ko', markersize=2)
        ax1.plot(np.arange(T)[-args['test_window']:], np.mean(pred, axis=0), ls='solid', color='b')
        ax1.plot(np.arange(T)[-args['test_window']:], np.median(pred, axis=0), ls='dashed', color='k')

        ax1.fill_between(np.arange(T)[-args['test_window']:], lower, upper, color='lightblue')
        ax1.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout(pad=0.7)
        plt.savefig('plot.{}.{:.2f}.pdf'.format(args['noise_model'], args['shock']))

    #with open(args['log_dir'] + '/' + log_file[:-4] + '.pkl', 'wb') as f:
    #    pickle.dump(results, f, protocol=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multivariate timeseries models")
    parser.add_argument("--noise-model", default='stable-trans', type=str,
                        choices=['gaussian', 'stable-obs', 'stable-trans', 'stud-obs'])
    parser.add_argument("--train-window", default=80, type=int)
    parser.add_argument("--test-window", default=20, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--num-warmup", default=200, type=int)
    parser.add_argument("--num-samples", default=400, type=int)
    parser.add_argument("--num-eval-samples", default=400, type=int)
    parser.add_argument("--max-tree-depth", default=4, type=int)
    parser.add_argument("--shock", default=4.0, type=float)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--mode", default="jump", type=str, choices=['jump', 'shift'])
    parser.add_argument('-ld', '--log-dir', type=str, default="./logs/")
    args = parser.parse_args()

    main(**vars(args))
