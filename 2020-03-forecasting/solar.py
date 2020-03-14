# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import uuid
import time

import numpy as np
import math
import torch
import torch.distributions.constraints as constraints

import pyro
from pyro.contrib.forecast import ForecastingModel, backtest, Forecaster
from pyro.nn import PyroParam
from pyro.infer.reparam import SymmetricStableReparam, StudentTReparam, StableReparam, LocScaleReparam
from pyro.distributions import StudentT, Stable, Normal
from pyro.ops.tensor_utils import periodic_cumsum, periodic_features, periodic_repeat
import pyro.poutine as poutine

import pickle
from dataloader import get_data as get_raw_data
from logger import get_logger


root_two = math.sqrt(2.0)


class Model(ForecastingModel):
    def __init__(self, obs_noise="gaussian", trans_noise="gaussian"):
        super().__init__()
        self.obs_noise = obs_noise
        self.trans_noise = trans_noise

        if obs_noise == "gaussian":
            self.obs_config = {}
        elif obs_noise == "stable":
            self.obs_stability = PyroParam(torch.tensor(1.95), constraint=constraints.interval(1.01, 1.99))
            self.obs_config = {"residual": SymmetricStableReparam()}
        elif obs_noise == "student":
            self.obs_nu = PyroParam(torch.tensor(5.0), constraint=constraints.interval(1.01, 30.0))
            self.obs_config = {"residual": StudentTReparam()}

        if trans_noise in ["gaussian", "none"]:
            self.trans_config = {}
        elif trans_noise == "stable":
            self.trans_stability = PyroParam(torch.tensor(1.95), constraint=constraints.interval(1.01, 1.99))
            self.trans_config = {"drift": SymmetricStableReparam()}
        elif trans_noise == "student":
            self.trans_nu = PyroParam(torch.tensor(5.0), constraint=constraints.interval(1.01, 30.0))
            self.trans_config = {"drift": StudentTReparam()}

    def model(self, zero_data, covariates):
        feature_dim = covariates.size(-1)
        duration, num_stations = zero_data.shape

        seasonality_weights = pyro.sample("seasonality_weights",
                                          Normal(0, 0.1).expand([feature_dim, num_stations]).to_event(2))

        periodic = torch.einsum('...fs,tf->...ts', seasonality_weights, covariates)
        if periodic.dim() > 3:
            periodic = periodic.squeeze(-3)

        obs_scale = pyro.param("obs_scale", 0.2 * torch.ones(num_stations), constraint=constraints.positive)
        trans_scale = pyro.param("trans_scale", 1.0e-4 * torch.ones(num_stations), constraint=constraints.positive)
        station_bias = pyro.param("station_bias", torch.zeros(num_stations))

        with self.time_plate, poutine.reparam(config={"drift": LocScaleReparam()}):
            with poutine.reparam(config=self.trans_config):
                if self.trans_noise == "gaussian":
                    drift = pyro.sample("drift", Normal(torch.zeros(zero_data.shape), trans_scale).to_event(1))
                elif self.trans_noise == "stable":
                    drift = pyro.sample("drift", Stable(self.trans_stability,
                                                        0.0, scale=trans_scale / root_two).to_event(1))
                elif self.trans_noise == "student":
                    drift = pyro.sample("drift", StudentT(self.trans_nu, 0.0, scale=trans_scale).to_event(1))

        if self.trans_noise != "none":
            prediction = periodic + station_bias + drift.cumsum(-2)
        else:
            prediction = periodic + station_bias

        if self.obs_noise == "gaussian":
            noise_dist = Normal(0, obs_scale)
        elif self.obs_noise == "stable":
            noise_dist = Stable(self.obs_stability, 0.0, scale=obs_scale / root_two)
        elif self.obs_noise == "student":
            noise_dist = StudentT(self.obs_nu, 0.0, scale=obs_scale)

        with pyro.poutine.reparam(config=self.obs_config):
            self.predict(noise_dist, prediction)


def get_data(args=None):
    data, _, _, _ = get_raw_data(args['dataset'], args['data_dir'])
    print("raw data", data.shape)

    to_keep = args['train_window'] + args['test_window']
    assert to_keep <= data.size(0)

    data = data[:to_keep].float()

    data_mean = data.mean(0)
    data -= data_mean
    data_std = data.std(0)
    data /= data_std

    covariates = periodic_features(data.size(0), 24 * 60, 10)

    return data.cuda(), covariates.cuda()


def main(**args):
    log_file = '{}.{}.{}.tt_{}_{}.sd_{}.nst_{}.cn_{:.1f}.lr_{:.2f}.lrd_{:.2f}.seed_{}.{}.log'
    log_file = log_file.format(args['dataset'], args['trans_noise'], args['obs_noise'],
                               args['train_window'], args['test_window'],
                               args['state_dim'], args['num_steps'],
                               args['clip_norm'], args['learning_rate'], args['learning_rate_decay'],
                               args['seed'],
                               str(uuid.uuid4())[0:4])

    log = get_logger(args['log_dir'], log_file, use_local_logger=False)

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')

    log(args)
    log("")

    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args['seed'])

    t0 = time.time()

    def svi_forecaster_options(t0, t1, t2):
        num_steps = args['num_steps']  # if t1 == args.train_window else 200
        lr = args['learning_rate']  # if t1 == args.train_window else 0.1 * args.learning_rate
        lrd = args['learning_rate_decay']  # if t1 == args.train_window else 0.1
        return {"num_steps": num_steps, "learning_rate": lr,
                "learning_rate_decay": lrd, "log_every": args['log_every'],
                "dct_gradients": False, "warm_start": False,
                "clip_norm": args['clip_norm'],
                "vectorize_particles": False,
                "num_particles": 1}

    data, covariates = get_data(args=args)
    print("covariates", covariates.shape)

    results = {}

    metrics = backtest(data, covariates,
                       lambda: Model(obs_noise=args['obs_noise'], trans_noise=args['trans_noise']).cuda(),
                       train_window=None,
                       seed=args['seed'],
                       min_train_window=args['train_window'],
                       test_window=args['test_window'],
                       stride=args['stride'],
                       num_samples=args['num_eval_samples'],
                       batch_size=5,
                       forecaster_options=svi_forecaster_options,
                       forecaster_fn=Forecaster)

    log("### EVALUATION ###")
    for name in ["mae", "crps"]:
        values = [m[name] for m in metrics]
        mean, std = np.mean(values), np.std(values)
        results[name] = mean
        results[name + '_std'] = std
        log("{} = {:0.4g} +- {:0.4g}".format(name, mean, std))

    # pred = np.stack([m['pred'].data.cpu().numpy() for m in metrics])
    # results['pred'] = pred

    for name, value in pyro.get_param_store().items():
        if value.numel() == 1:
            results[name] = value.item()
            print("[{}]".format(name), value.item())
        elif value.numel() < 10:
            results[name] = value.data.cpu().numpy()
            print("[{}]".format(name), value.data.cpu().numpy())

    with open(args['log_dir'] + '/' + log_file[:-4] + '.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=2)

    log("[ELAPSED TIME]: {:.3f}".format(time.time() - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multivariate timeseries models")
    parser.add_argument("--trans-noise", default='stable', type=str,
                        choices=['gaussian', 'stable', 'student', 'skew', 'none'])
    parser.add_argument("--obs-noise", default='stable', type=str, choices=['gaussian', 'stable', 'student', 'skew'])
    parser.add_argument("--dataset", default='solar', type=str)
    parser.add_argument("--data-dir", default='./data/', type=str)
    parser.add_argument("--log-dir", default='./logs/', type=str)
    parser.add_argument("--train-window", default=250000, type=int)
    parser.add_argument("--test-window", default=50000, type=int)
    parser.add_argument("--stride", default=50000, type=int)
    parser.add_argument("--num-eval-samples", default=200, type=int)
    parser.add_argument("--clip-norm", default=10.0, type=float)
    parser.add_argument("-n", "--num-steps", default=300, type=int)
    parser.add_argument("-d", "--state-dim", default=5, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.03, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.001, type=float)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--log-every", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    main(**vars(args))
