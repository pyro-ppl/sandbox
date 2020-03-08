# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

import numpy as np

import torch
import torch.distributions.constraints as constraints

import pyro
import pyro.distributions as dist
from pyro.contrib.forecast import ForecastingModel, backtest, HMCForecaster, Forecaster
from pyro.contrib.timeseries import IndependentMaternGP
from pyro.nn import PyroParam, PyroSample
from pyro.infer.reparam import SymmetricStableReparam, LatentStableReparam
from pyro.infer.reparam import LinearHMMReparam
from pyro.poutine import reparam
from pyro.distributions import MultivariateNormal, LogNormal, Uniform

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
                 obs_noise_scale_init=None, stable_noise="obs",
                 latent_globals=False):
        self.stable_noise = stable_noise
        self.latent_globals = latent_globals
        self.skew = 0.0
        assert stable_noise in ['obs']
        super().__init__(nu=nu, dt=dt, obs_dim=obs_dim,
                         length_scale_init=length_scale_init,
                         kernel_scale_init=kernel_scale_init,
                         obs_noise_scale_init=obs_noise_scale_init)
        if self.latent_globals:
            self.kernel.length_scale = PyroSample(LogNormal(2.0, 2 * torch.ones(self.obs_dim)).to_event(1))
            self.kernel.kernel_scale = PyroSample(LogNormal(0.0, 2 * torch.ones(self.obs_dim)).to_event(1))
            self.obs_noise_scale = PyroSample(LogNormal(-2.0, 2 * torch.ones(self.obs_dim)).to_event(1))
            self.stability = PyroSample(Uniform(1.01, 1.99 * torch.ones(1)).to_event(1))
        else:
           self.stability = PyroParam(torch.tensor(1.95), constraint=constraints.interval(1.01, 1.99))

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


def get_data(shock=1.0, shock_times = [43, 91, 137], args=None):
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
    covariates = torch.zeros(data.size(0), 0)
    return data, covariates


class Model(ForecastingModel):
    def __init__(self, noise_model="gaussian", latent_globals=False):
        super().__init__()
        self.noise_model = noise_model
        if noise_model == "gaussian":
            self.noise_gp = IndependentMaternGP(obs_dim=1, length_scale_init=torch.tensor([10.0]))
            self.config = {}
        elif noise_model in ["stable-obs"]:
            self.noise_gp = IndependentMaternStableProcess(obs_dim=1, stable_noise=noise_model[7:],
                                                           length_scale_init=torch.tensor([10.0]),
                                                           latent_globals=latent_globals)
            self.config = {"residual": LinearHMMReparam(obs=SymmetricStableReparam())}

    def model(self, zero_data, covariates):
        noise_dist = self.noise_gp.get_dist(duration=zero_data.size(-2))
        if self.noise_model in ["gaussian", "stable-obs"]:
            noise_dist = dist.IndependentHMM(noise_dist)

        prediction = torch.zeros(noise_dist.event_shape)
        with reparam(config=self.config):
            self.predict(noise_dist, prediction)


def main(args):
    print(args)
    print("")

    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.seed)

    def svi_forecaster_options(t0, t1, t2):
        num_steps = args.num_steps if t1 == args.train_window else 200
        lr = args.learning_rate if t1 == args.train_window else 0.1 * args.learning_rate
        lrd = args.learning_rate_decay if t1 == args.train_window else 0.1
        return {"num_steps": num_steps, "learning_rate": lr,
                "learning_rate_decay": lrd, "log_every": args.log_every,
                "dct_gradients": args.dct, "warm_start": False,
                "clip_norm": args.clip_norm}

    def hmc_forecaster_options(t0, t1, t2):
        return {"max_tree_depth": 1,
                "num_warmup": 10,
                "num_samples": 20}

    results = {}

    for shock in [1.0]:
    #for shock in [0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.4, 3.2, 4.0, 5.0, 6.4]:
        print("*** shock = %.2f ***" % shock)
        results[shock] = {}

        data, covariates = get_data(shock=shock, args=args)

        metrics = backtest(data, covariates,
                           lambda: Model(noise_model=args.noise_model,
                                         latent_globals=args.latent_globals),
                           train_window=None,
                           min_train_window=args.train_window,
                           test_window=args.test_window,
                           stride=args.stride,
                           num_samples=args.num_samples,
                           forecaster_options=svi_forecaster_options if args.inference=='svi' else hmc_forecaster_options,
                           forecaster_fn=Forecaster if args.inference=='svi' else HMCForecaster)

        if not args.latent_globals:
            length_scale = pyro.param("noise_gp.kernel.length_scale").item()
            kernel_scale = pyro.param("noise_gp.kernel.kernel_scale").item()
            obs_noise_scale = pyro.param("noise_gp.obs_noise_scale").item()

            print("length_scale", length_scale)
            print("kernel_scale", kernel_scale)
            print("obs_noise_scale", obs_noise_scale)

            results[shock]['length_scale'] = length_scale
            results[shock]['kernel_scale'] = kernel_scale
            results[shock]['obs_noise_scale'] = obs_noise_scale

        if args.noise_model == "stable-obs" and not args.latent_globals:
            stability = pyro.param("noise_gp.stability").item()
            results[shock]['stability'] = stability
            print("stability", stability)

        print("### EVALUATION ###")
        for name in ["mae", "rmse", "crps"]:
            values = [m[name] for m in metrics]
            mean, std = np.mean(values), np.std(values)
            results[shock][name] = mean
            print("{} = {:0.3g} +- {:0.3g}".format(name, mean, std))

    f = 'results.{}.{}.{}.{}.pkl'.format(args.noise_model, args.mode, args.inference, args.latent_globals)
    with open(f, 'wb') as f:
        pickle.dump(results, f, protocol=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multivariate timeseries models")
    parser.add_argument("--noise-model", default='stable-obs', type=str,
                        choices=['gaussian', 'stable-obs'])
    parser.add_argument("--train-window", default=200, type=int)
    parser.add_argument("--test-window", default=1, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--clip-norm", default=10.0, type=float)
    parser.add_argument("-n", "--num-steps", default=1200, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.03, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.003, type=float)
    parser.add_argument("--dct", action="store_true")
    parser.add_argument("--latent-globals", action="store_true")
    parser.add_argument("--num-samples", default=99, type=int)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--mode", default="jump", type=str, choices=['jump', 'shift'])
    parser.add_argument("--inference", default="svi", type=str, choices=['svi', 'hmc'])
    args = parser.parse_args()
    main(args)
