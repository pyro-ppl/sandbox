# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

import numpy as np

import torch
import torch.distributions.constraints as constraints

import pyro
import pyro.distributions as dist
from pyro.contrib.forecast import ForecastingModel, backtest, HMCForecaster
from pyro.contrib.timeseries import IndependentMaternGP
from pyro.contrib.timeseries import LinearlyCoupledMaternGP
from pyro.nn import PyroParam
from pyro.infer.reparam import SymmetricStableReparam, LatentStableReparam
from pyro.infer.reparam import LinearHMMReparam
from pyro.poutine import reparam
from pyro.distributions import MultivariateNormal

from os.path import exists
from urllib.request import urlopen

import pickle

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)


class IndependentMaternStableProcess(IndependentMaternGP):
    """
    A IndependentMaternGP with symmetric stable observation noise
    or symmetric stable transition noise.
    """
    def __init__(self, nu=0.5, dt=1.0, obs_dim=1,
                 length_scale_init=None, kernel_scale_init=None,
                 obs_noise_scale_init=None, stable_noise="obs"):
        self.stable_noise = stable_noise
        assert stable_noise in ['obs']
        super().__init__(nu=nu, dt=dt, obs_dim=obs_dim,
                         length_scale_init=length_scale_init,
                         kernel_scale_init=kernel_scale_init,
                         obs_noise_scale_init=obs_noise_scale_init)
        #self.stability = PyroParam(torch.tensor(1.95),
        #                           constraint=constraints.interval(1.50, 1.99))
        self.stability = 1.9
        self.skew = 0.0

    def _get_init_dist(self):
        cov = self.kernel.stationary_covariance().squeeze(-3)
        return MultivariateNormal(self.obs_matrix.new_zeros(self.obs_dim, self.kernel.state_dim), cov)

    def _get_obs_dist(self):
        scale = self.obs_noise_scale.unsqueeze(-1).unsqueeze(-1)
        if self.stable_noise == "obs":
            return dist.Stable(self.stability, self.skew, scale=scale).to_event(1)
        else:
            return dist.Normal(0.0, scale=scale).to_event(1)

    def get_dist(self, duration=None):
        trans_matrix, process_covar = self.kernel.transition_matrix_and_covariance(dt=self.dt)
        if self.stable_noise == "trans":
            trans_dist = dist.Stable(self.stability, self.skew, scale=process_covar.sqrt()).to_event(1)
        else:
            trans_dist = MultivariateNormal(self.obs_matrix.new_zeros(self.obs_dim, 1, self.kernel.state_dim),
                                            process_covar.unsqueeze(-3))
        trans_matrix = trans_matrix.unsqueeze(-3)
        return dist.LinearHMM(self._get_init_dist(), trans_matrix, trans_dist,
                              self.obs_matrix, self._get_obs_dist(), duration=duration)


def get_data(shock=1.0, shock_times = [37, 99, 125], args=None):
    torch.manual_seed(0)
    data = torch.cos(0.1 * torch.arange(201).float()).unsqueeze(-1)
    data += 0.05 * torch.randn(data.shape)
    if args.mode == "jump":
        print("MODE JUMP")
        for t in shock_times:
            data[t, 0] += shock
    elif args.mode == "shift":
        print("MODE SHIFT")
        for t in shock_times:
            data[t:, 0] += shock
    print("DATA\n", data[0:6,0].data.numpy(), data[-6:,0].data.numpy())
    covariates = torch.zeros(data.size(0), 0)
    return data, covariates


class Model(ForecastingModel):
    def __init__(self, noise_model="ind"):
        super().__init__()
        self.noise_model = noise_model
        if noise_model == "ind":
            self.noise_gp = IndependentMaternGP(obs_dim=1, length_scale_init=torch.tensor([10.0]))
            self.config = {}
        elif noise_model in ["stable-obs"]:
            self.noise_gp = IndependentMaternStableProcess(obs_dim=1, stable_noise=noise_model[7:],
                                                           length_scale_init=torch.tensor([10.0]))
            self.config = {"residual": LinearHMMReparam(obs=SymmetricStableReparam())}
        else:
            self.noise_gp = LinearlyCoupledMaternGP(obs_dim=1, num_gps=1)
            self.config = {}

    def model(self, zero_data, covariates):
        duration, dim = zero_data.shape
        assert dim == 1

        noise_dist = self.noise_gp.get_dist(duration=zero_data.size(-2))
        if self.noise_model in ["ind", "stable-obs"]:
            noise_dist = dist.IndependentHMM(noise_dist)

        with reparam(config=self.config):
            self.predict(noise_dist, zero_data)


def main(args):
    print(args)
    print("")

    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.seed)

    def forecaster_options(t0, t1, t2):
        _forecaster_options = {"max_tree_depth": 3,
                               "num_warmup": 100,
                               "num_samples": 200}

        return _forecaster_options

    results = {}

    for shock in [1.0, 4.0]:
    #for shock in [0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.4, 3.2, 4.0, 5.0, 6.4]:
        print("*** shock = %.2f ***" % shock)
        results[shock] = {}

        data, covariates = get_data(shock=shock, args=args)

        metrics = backtest(data, covariates, lambda: Model(noise_model=args.noise_model),
                           forecaster_fn=HMCForecaster,
                           train_window=None,
                           min_train_window=args.train_window,
                           test_window=args.test_window,
                           stride=args.stride,
                           num_samples=args.num_samples,
                           forecaster_options=forecaster_options)

        length_scale = pyro.param("noise_gp.kernel.length_scale").item()
        kernel_scale = pyro.param("noise_gp.kernel.kernel_scale").item()
        obs_noise_scale = pyro.param("noise_gp.obs_noise_scale").item()

        print("length_scale", length_scale)
        print("kernel_scale", kernel_scale)
        print("obs_noise_scale", obs_noise_scale)

        results[shock]['length_scale'] = length_scale
        results[shock]['kernel_scale'] = kernel_scale
        results[shock]['obs_noise_scale'] = obs_noise_scale

        #if args.noise_model == "stable-obs":
        #    stability = pyro.param("noise_gp.stability").item()
        #    results[shock]['stability'] = stability
        #    print("stability", stability)

        print("### EVALUATION ###")
        for name in ["mae", "rmse", "crps"]:
            values = [m[name] for m in metrics]
            mean, std = np.mean(values), np.std(values)
            results[shock][name] = mean
            print("{} = {:0.3g} +- {:0.3g}".format(name, mean, std))

    with open('results.' + args.noise_model + '.' + args.mode + '.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multivariate timeseries models")
    parser.add_argument("--noise-model", default='stable-obs', type=str,
                        choices=['ind', 'lc', 'stable-obs'])
    parser.add_argument("--train-window", default=200, type=int)
    parser.add_argument("--test-window", default=1, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--clip-norm", default=10.0, type=float)
    parser.add_argument("-n", "--num-steps", default=1200, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.03, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.003, type=float)
    parser.add_argument("--dct", action="store_true")
    parser.add_argument("--num-samples", default=500, type=int)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--mode", default="jump", type=str, choices=['jump', 'shift'])
    args = parser.parse_args()
    main(args)
