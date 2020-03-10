# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

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
from pyro.infer.reparam import SymmetricStableReparam, LatentStableReparam
from pyro.infer.reparam import LinearHMMReparam
from pyro.poutine import reparam
from pyro.distributions import MultivariateNormal, LogNormal, Uniform

from os.path import exists
from urllib.request import urlopen

import pickle

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)
root_two = math.sqrt(2.0)


class IndependentMaternStableProcess(IndependentMaternGP):
    """
    A IndependentMaternGP with symmetric stable observation noise
    or symmetric stable transition noise.
    """
    def __init__(self, nu=1.5, dt=1.0, obs_dim=1,
                 length_scale_init=None, kernel_scale_init=None,
                 obs_noise_scale_init=None, stable_noise="obs",
                 latent_globals=False):
        self.stable_noise = stable_noise
        self.latent_globals = latent_globals
        self.skew = 0.0
        assert stable_noise in ['obs', 'trans', 'none']
        super().__init__(nu=nu, dt=dt, obs_dim=obs_dim,
                         length_scale_init=length_scale_init,
                         kernel_scale_init=kernel_scale_init,
                         obs_noise_scale_init=obs_noise_scale_init)
        if self.latent_globals:
            self.kernel.length_scale = PyroSample(LogNormal(1.0, 2.0))
            self.kernel.kernel_scale = PyroSample(LogNormal(0.0, 2.0))
            self.obs_noise_scale = PyroSample(LogNormal(-2.0, 2.0))
            if stable_noise != 'none':
                self.stability = PyroSample(Uniform(1.01, 1.99))
        elif stable_noise != 'none':
           self.stability = PyroParam(torch.tensor(1.95), constraint=constraints.interval(1.01, 1.99))

    def _get_init_dist(self):
        cov = self.kernel.stationary_covariance()
        return MultivariateNormal(self.obs_matrix.new_zeros(self.obs_dim, self.kernel.state_dim), cov)

    def _get_obs_dist(self):
        scale = self.obs_noise_scale.unsqueeze(-1).unsqueeze(-1)
        if self.stable_noise == "obs":
            stability = self.stability.unsqueeze(-1).unsqueeze(-1)
            return dist.Stable(stability, self.skew, scale=scale / root_two).to_event(1)
        else:
            return dist.Normal(0.0, scale=scale).to_event(1)

    def get_dist(self, duration=None):
        trans_matrix, process_covar = self.kernel.transition_matrix_and_covariance(dt=self.dt)
        if self.stable_noise == "trans":
            scale = process_covar.sqrt() / root_two
            stability = self.stability.unsqueeze(-1).unsqueeze(-1)
            trans_dist = dist.Stable(stability, self.skew, scale=scale).to_event(1)
        else:
            trans_dist = MultivariateNormal(self.obs_matrix.new_zeros(self.obs_dim, 1, self.kernel.state_dim),
                                            process_covar.unsqueeze(-3))
        trans_matrix = trans_matrix.unsqueeze(-3)
        return dist.LinearHMM(self._get_init_dist(), trans_matrix, trans_dist,
                              self.obs_matrix, self._get_obs_dist(), duration=duration)


def get_data2(shock=1.0, shock_times = [11, 15, 7], args=None):
    torch.manual_seed(0)
    data = torch.cos(0.1 * torch.arange(args.train_window + args.test_window).float()).unsqueeze(-1)
    data += 0.15 * torch.randn(data.shape)
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

def get_data(shock=1.0, shock_times = [7, 23, 37, 45, 71], args=None):
    data = torch.tensor(np.load('ushum.npy'))
    data = data[:args.train_window + args.test_window, 0:1]

    data_mean = data.mean(0)
    data -= data_mean
    data_std = data.std(0)
    data /= data_std

    if args.mode == "jump":
        print("MODE JUMP")
        k = 0
        for t in shock_times:
            if k % 2 == 0:
                data[t, 0] += shock
            else:
                data[t, 0] -= shock
            k += 1

    covariates = torch.zeros(data.size(0), 0)

    return data, covariates

def get_data4(shock=1.0, shock_times = [7, 23, 37, 45, 71, 88, 93], args=None):
    assert args.train_window + args.test_window < 101
    data = torch.tensor(pd.read_csv('nile.csv').volume.values).float().unsqueeze(-1)
    data = data[:args.train_window + args.test_window]

    data_mean = data.mean(0)
    data -= data_mean
    data_std = data.std(0)
    data /= data_std

    covariates = torch.zeros(data.size(0), 0)

    return data, covariates

def get_data3(shock=1.0, shock_times = [7, 23, 37, 45, 71, 88, 93], args=None):
    if not exists("eeg.dat"):
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        with open("eeg.dat", "wb") as f:
            f.write(urlopen(url).read())

    data = np.loadtxt('eeg.dat', delimiter=',', skiprows=19)
    data = data[::5, :]
    data = torch.tensor(data[:args.train_window + args.test_window, 4:5]).float()

    data_mean = data.mean(0)
    data -= data_mean
    data_std = data.std(0)
    data /= data_std

    if args.mode == "jump":
        print("MODE JUMP")
        k = 0
        for t in shock_times:
            if k % 2 == 0:
                data[t, 0] += shock
            else:
                data[t, 0] -= shock
            k += 1
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
        self.latent_globals = latent_globals
        if noise_model == "gaussian":
            #self.noise_gp = IndependentMaternGP(obs_dim=1, length_scale_init=torch.tensor([10.0]))
            self.noise_gp = IndependentMaternStableProcess(obs_dim=1, stable_noise='none',
                                                           length_scale_init=torch.tensor([10.0]),
                                                           latent_globals=latent_globals)
            #self.config = {}
            self.config = {"residual": LinearHMMReparam()}
        elif noise_model == "stable-obs":
            self.noise_gp = IndependentMaternStableProcess(obs_dim=1, stable_noise='obs',
                                                           length_scale_init=torch.tensor([10.0]),
                                                           latent_globals=latent_globals)
            self.config = {"residual": LinearHMMReparam(obs=SymmetricStableReparam())}
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
        if self.noise_model in ["gaussian", "stable-obs", "stable-trans"]:
            noise_dist = dist.IndependentHMM(noise_dist)

        with reparam(config=self.config):
            self.predict(noise_dist, zero_data)


def main(args):
    torch.set_default_dtype(torch.float64)
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
                "clip_norm": args.clip_norm,
                "vectorize_particles": False,
                "num_particles": 1}

    def hmc_forecaster_options(t0, t1, t2):
        return {"max_tree_depth": args.max_depth,
                "num_warmup": args.num_warmup,
                "num_samples": args.num_samples}

    results = {}

    for shock in [0.0, 1.0, 2.0, 3.0, 4.0]:
    #for shock in [0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.4, 3.2, 4.0, 5.0, 6.4]:
        print("*** shock = %.2f ***" % shock)
        results[shock] = {}

        data, covariates = get_data(shock=shock, args=args)
        print("data, covariates", data.shape, covariates.shape)

        metrics = backtest(data, covariates,
                           lambda: Model(noise_model=args.noise_model,
                                         latent_globals=args.latent_globals),
                           train_window=None,
                           min_train_window=args.train_window,
                           test_window=args.test_window,
                           stride=args.stride,
                           num_samples=args.num_eval_samples,
                           forecaster_options=svi_forecaster_options if args.inference=='svi' else hmc_forecaster_options,
                           forecaster_fn=Forecaster if args.inference=='svi' else HMCForecaster)

        #for name, value in pyro.get_param_store().named_parameters():
        #    if value.numel() == 1:
        #        print("[{}]".format(name), value)

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
            print("{} = {:0.4g} +- {:0.4g}".format(name, mean, std))

        if args.plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            f, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
            T = data.size(0)

            pred = metrics[0]['pred'].data.numpy()[:, 0, :, 0]
            lower, upper = np.percentile(pred, 10, axis=0), np.percentile(pred, 90, axis=0)

            ax1.plot(np.arange(T), data[:, 0], 'ko', markersize=2)
            ax1.plot(np.arange(T)[-args.test_window:], np.mean(pred, axis=0), ls='solid', color='b')
            ax1.plot(np.arange(T)[-args.test_window:], np.median(pred, axis=0), ls='dashed', color='k')

            ax1.fill_between(np.arange(T)[-args.test_window:], lower, upper, color='lightblue')
           # ax.set_ylabel("$y_{%d}$" % (which + 1), fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=14)

            plt.tight_layout(pad=0.7)
            plt.savefig('plot.{}.{:.2f}.pdf'.format(args.noise_model, shock))

    f = 'results.{}.{}.{}.{}.pkl'.format(args.noise_model, args.mode, args.inference, args.latent_globals)
    with open(f, 'wb') as f:
        pickle.dump(results, f, protocol=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multivariate timeseries models")
    parser.add_argument("--noise-model", default='stable-obs', type=str,
                        choices=['gaussian', 'stable-obs', 'stable-trans'])
    parser.add_argument("--train-window", default=80, type=int)
    parser.add_argument("--test-window", default=20, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--num-warmup", default=200, type=int)
    parser.add_argument("--num-samples", default=500, type=int)
    parser.add_argument("--num-eval-samples", default=5000, type=int)
    parser.add_argument("--max-depth", default=4, type=int)
    parser.add_argument("--clip-norm", default=10.0, type=float)
    parser.add_argument("-n", "--num-steps", default=1200, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.03, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.003, type=float)
    parser.add_argument("--dct", action="store_true")
    parser.add_argument("--latent-globals", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--mode", default="jump", type=str, choices=['jump', 'shift'])
    parser.add_argument("--inference", default="svi", type=str, choices=['svi', 'hmc'])
    args = parser.parse_args()
    main(args)
