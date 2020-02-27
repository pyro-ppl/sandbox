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

from os.path import exists
from urllib.request import urlopen

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)


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
    def __init__(self):
        super().__init__()
        self.noise_gp = LinearlyCoupledMaternGP(obs_dim=14)
        #self.noise_gp = IndependentMaternGP(obs_dim=14)

    def model(self, zero_data, covariates):
        duration, dim = zero_data.shape[-2:]
        assert dim == 14

        #obs_noise_scale = pyro.param("obs_noise_scale", 0.1 * torch.ones(dim), constraint=constraints.positive)
        #jump_scale = pyro.param("jump_scale", 0.1 * torch.ones(dim), constraint=constraints.positive)

        #with self.time_plate:
        #    jumps = pyro.sample("jumps", dist.Normal(0, jump_scale).to_event(1))
        #prediction = jumps.cumsum(-2)

        noise_dist = self.noise_gp.get_dist(duration=zero_data.size(-2))
        self.predict(noise_dist, zero_data)

def main(args):
    pyro.enable_validation(__debug__)
    data, covariates = preprocess(args)

    noise_gp = IndependentMaternGP(obs_dim=14)
    noise_dist = noise_gp.get_dist(duration=7)
    print("noise_dist event_shape", noise_dist.event_shape, "batch_shape", noise_dist.batch_shape)

    print("data",data.shape)
    print("covariates",covariates.shape)

    # The backtest() function automatically trains and evaluates our model on
    # different windows of data.
    forecaster_options = {
        "num_steps": args.num_steps,
        "learning_rate": args.learning_rate,
        "learning_rate_decay": args.learning_rate_decay,
        "log_every": args.log_every,
        "dct_gradients": args.dct,
    }
    metrics = backtest(data, covariates, Model,
                       train_window=args.train_window,
                       test_window=args.test_window,
                       stride=args.stride,
                       num_samples=args.num_samples,
                       forecaster_options=forecaster_options)

    for name in ["mae", "rmse", "crps"]:
        values = [m[name] for m in metrics]
        mean = np.mean(values)
        std = np.std(values)
        print("{} = {:0.3g} +- {:0.3g}".format(name, mean, std))
        print("values", values)
    return metrics


if __name__ == "__main__":
    #assert pyro.__version__.startswith('1.2.1')
    parser = argparse.ArgumentParser(description="Multivariate timeseries models")
    parser.add_argument("--train-window", default=740, type=int)
    parser.add_argument("--test-window", default=1, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("-n", "--num-steps", default=301, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.05, type=float)
    parser.add_argument("--dct", action="store_true")
    parser.add_argument("--num-samples", default=100, type=int)
    parser.add_argument("--log-every", default=5, type=int)
    parser.add_argument("--seed", default=1234567890, type=int)
    args = parser.parse_args()
    main(args)
