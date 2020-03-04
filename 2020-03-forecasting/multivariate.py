# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

import numpy as np

import torch
import torch.distributions.constraints as constraints

import pyro
import pyro.distributions as dist
from pyro.contrib.forecast import ForecastingModel, backtest
from pyro.contrib.timeseries import IndependentMaternGP, LinearlyCoupledMaternGP
from pyro.nn import PyroParam
from pyro.infer.reparam import SymmetricStableReparam
from pyro.infer.reparam import LinearHMMReparam

from os.path import exists
from urllib.request import urlopen

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)


class IndependentMaternStableProcess(IndependentMaternGP):
    """
    A IndependentMaternGP with (symmetric) stable observation noise.
    """
    def __init__(self, nu=1.5, dt=1.0, obs_dim=1,
                 length_scale_init=None, kernel_scale_init=None,
                 obs_noise_scale_init=None):
        super().__init__(nu=nu, dt=dt, obs_dim=obs_dim, length_scale_init=length_scale_init,
                         kernel_scale_init=kernel_scale_init, obs_noise_scale_init=obs_noise_scale_init)
        self.min_stability = 1.5
        self.stability = PyroParam(torch.tensor(1.99 - self.min_stability), constraint=constraints.positive)

    def _get_init_dist(self):
        return dist.MultivariateNormal(self.obs_matrix.new_zeros(self.obs_dim, self.kernel.state_dim),
                                       self.kernel.stationary_covariance().squeeze(-3))

    def _get_obs_dist(self):
        return dist.Stable(self.min_stability + self.stability, 0.0,
                           scale=self.obs_noise_scale.unsqueeze(-1).unsqueeze(-1)).to_event(1)

    def _get_trans_dist(self, trans_matrix, stationary_covariance):
        covar = stationary_covariance - torch.matmul(trans_matrix.transpose(-1, -2),
                                                     torch.matmul(stationary_covariance, trans_matrix))
        return dist.MultivariateNormal(covar.new_zeros(self.full_state_dim), covar)

    def get_dist(self, duration=None):
        trans_matrix, process_covar = self.kernel.transition_matrix_and_covariance(dt=self.dt)
        trans_dist = dist.MultivariateNormal(self.obs_matrix.new_zeros(self.obs_dim, 1, self.kernel.state_dim),
                                        process_covar.unsqueeze(-3))
        trans_matrix = trans_matrix.unsqueeze(-3)
        return dist.LinearHMM(self._get_init_dist(), trans_matrix, trans_dist,
                              self.obs_matrix, self._get_obs_dist(), duration=duration)

def preprocess(args):
    """
    """
    print("Loading data")
    if not exists("eeg.dat"):
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        with open("eeg.dat", "wb") as f:
            f.write(urlopen(url).read())

    data = np.loadtxt('eeg.dat', delimiter=',', skiprows=19)
    print("[raw data shape] {}".format(data.shape))
    data = data[::20, :]
    print("[data shape after thinning] {}".format(data.shape))
    data = torch.tensor(data[:, :-1]).float()

    data_mean = data.mean(0)
    data -= data_mean
    data_std = data.std(0)
    data /= data_std

    covariates = torch.zeros(data.size(0), 0)

    return data, covariates


class Model(ForecastingModel):
    def __init__(self, noise_model="ind"):
        super().__init__()
        self.noise_model = noise_model
        if noise_model == "ind":
            self.noise_gp = IndependentMaternGP(obs_dim=14)
        elif noise_model == "stable":
            self.noise_gp = IndependentMaternStableProcess(obs_dim=14)
        else:
            self.noise_gp = LinearlyCoupledMaternGP(obs_dim=14, num_gps=14)

    def model(self, zero_data, covariates):
        duration, dim = zero_data.shape
        assert dim == 14

        noise_dist = self.noise_gp.get_dist(duration=zero_data.size(-2))
        if self.noise_model in ["ind", "stable"]:
            noise_dist = dist.IndependentHMM(noise_dist)

        rep = LinearHMMReparam(obs=SymmetricStableReparam())
        with pyro.poutine.reparam(config={"residual": rep}):
            self.predict(noise_dist, zero_data)

        #self.predict(noise_dist, zero_data)

def main(args):
    print(args)
    print("")

    pyro.enable_validation(__debug__)
    data, covariates = preprocess(args)

    print("data",data.shape)
    print("covariates",covariates.shape)

    # The backtest() function automatically trains and evaluates our model on
    # different windows of data.
    def forecaster_options(t0, t1, t2):
        num_steps = args.num_steps if t1==args.train_window else 0
        _forecaster_options = {
            "num_steps": num_steps,
            "learning_rate": args.learning_rate,
            "learning_rate_decay": args.learning_rate_decay,
            "log_every": args.log_every,
            "dct_gradients": args.dct,
            "warm_start": True}
        return _forecaster_options

    metrics = backtest(data, covariates, lambda: Model(noise_model=args.noise_model),
                       train_window=None,
                       min_train_window=args.train_window,
                       test_window=args.test_window,
                       stride=args.stride,
                       num_samples=args.num_samples,
                       forecaster_options=forecaster_options)

    for name in ["mae", "rmse", "crps"]:
        values = [m[name] for m in metrics]
        mean = np.mean(values)
        std = np.std(values)
        print("{} = {:0.3g} +- {:0.3g}".format(name, mean, std))
    return metrics


if __name__ == "__main__":
    #assert pyro.__version__.startswith('1.2.1')
    parser = argparse.ArgumentParser(description="Multivariate timeseries models")
    parser.add_argument("--noise-model", default='stable', type=str, choices=['ind', 'lc', 'stable'])
    parser.add_argument("--train-window", default=500, type=int)
    parser.add_argument("--test-window", default=1, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("-n", "--num-steps", default=800, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.01, type=float)
    parser.add_argument("--dct", action="store_true")
    parser.add_argument("--num-samples", default=100, type=int)
    parser.add_argument("--log-every", default=5, type=int)
    parser.add_argument("--seed", default=1234567890, type=int)
    args = parser.parse_args()
    main(args)
