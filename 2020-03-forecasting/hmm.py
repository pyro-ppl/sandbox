# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import uuid

import numpy as np
import math
import torch
import torch.distributions.constraints as constraints

import pyro
from pyro.contrib.forecast import ForecastingModel, backtest, Forecaster
from pyro.nn import PyroParam, PyroModule
from pyro.infer.reparam import SymmetricStableReparam, StudentTReparam, LinearHMMReparam
from pyro.distributions import StudentT, Stable, Normal, LinearHMM

import pickle
from dataloader import get_data as get_raw_data
from logger import get_logger


root_two = math.sqrt(2.0)


class StableLinearHMM(PyroModule):
    def __init__(self, obs_dim=1, trans_noise="gaussian", obs_noise="gaussian", state_dim=3):
        self.trans_noise = trans_noise
        self.obs_noise = obs_noise
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        assert trans_noise in ['gaussian', 'stable', 'student']
        assert obs_noise in ['gaussian', 'stable', 'student']
        super().__init__()
        self.obs_noise_scale = PyroParam(0.2 * torch.tensor(obs_dim), constraint=constraints.positive)
        self.trans_noise_scale = PyroParam(0.2 * torch.tensor(state_dim), constraint=constraints.positive)
        self.trans_matrix = PyroParam(0.3 * torch.randn(state_dim, state_dim))
        self.obs_matrix = PyroParam(0.3 * torch.randn(state_dim, obs_dim))
        if trans_noise == "stable":
            self.trans_stability = PyroParam(torch.tensor(1.95), constraint=constraints.interval(1.01, 1.99))
        elif trans_noise == "student":
            self.trans_nu = PyroParam(torch.tensor(5.0), constraint=constraints.interval(1.01, 30.0))
        if obs_noise == "stable":
            self.obs_stability = PyroParam(torch.tensor(1.95), constraint=constraints.interval(1.01, 1.99))
        elif obs_noise == "student":
            self.obs_nu = PyroParam(torch.tensor(5.0), constraint=constraints.interval(1.01, 30.0))

    def _get_init_dist(self):
        return Normal(torch.zeros(self.state_dim), torch.ones(self.state_dim)).to_event(1)

    def _get_obs_dist(self):
        if self.obs_noise == "stable":
            return Stable(self.obs_stability, torch.zeros(self.obs_dim),
                          scale=self.obs_noise_scale / root_two).to_event(1)
        elif self.obs_noise == "student":
            return StudentT(self.obs_nu, torch.zeros(self.obs_dim), self.obs_noise_scale).to_event(1)
        else:
            return Normal(torch.zeros(self.obs_dim), scale=self.obs_noise_scale).to_event(1)

    def _get_trans_dist(self):
        if self.trans_noise == "stable":
            return Stable(self.trans_stability, torch.zeros(self.state_dim),
                          scale=self.trans_noise_scale / root_two).to_event(1)
        elif self.trans_noise == "student":
            return StudentT(self.trans_nu, torch.zeros(self.state_dim), self.trans_noise_scale).to_event(1)
        else:
            return Normal(torch.zeros(self.state_dim), scale=self.trans_noise_scale).to_event(1)

    def get_dist(self, duration=None):
        return LinearHMM(self._get_init_dist(), self.trans_matrix, self._get_trans_dist(),
                         self.obs_matrix, self._get_obs_dist(), duration=duration)


def get_data(args=None):
    data, _, _, _ = get_raw_data(args['dataset'], args['data_dir'])

    to_keep = args['train_window'] + args['num_windows'] * args['test_window']
    assert to_keep <= data.size(0)

    data = data[:to_keep].float()

    data_mean = data.mean(0)
    data -= data_mean
    data_std = data.std(0)
    data /= data_std

    covariates = torch.zeros(data.size(0), 0)

    return data.cuda(), covariates.cuda()


class Model(ForecastingModel):
    def __init__(self, trans_noise="gaussian", obs_noise="gaussian", state_dim=3, obs_dim=14):
        super().__init__()
        self.trans_noise = trans_noise
        self.obs_noise = obs_noise
        self.hmm = StableLinearHMM(obs_dim=obs_dim, trans_noise=trans_noise, obs_noise=obs_noise, state_dim=state_dim)
        if trans_noise == "gaussian" and obs_noise == "gaussian":
            self.config = {"residual": LinearHMMReparam()}
        elif trans_noise == "stable" and obs_noise == "gaussian":
            self.config = {"residual": LinearHMMReparam(trans=SymmetricStableReparam())}
        elif trans_noise == "gaussian" and obs_noise == "stable":
            self.config = {"residual": LinearHMMReparam(obs=SymmetricStableReparam())}
        elif trans_noise == "stable" and obs_noise == "stable":
            self.config = {"residual": LinearHMMReparam(obs=SymmetricStableReparam(), trans=SymmetricStableReparam())}
        elif trans_noise == "gaussian" and obs_noise == "student":
            self.config = {"residual": LinearHMMReparam(obs=StudentTReparam())}
        elif trans_noise == "student" and obs_noise == "gaussian":
            self.config = {"residual": LinearHMMReparam(trans=StudentTReparam())}
        elif trans_noise == "student" and obs_noise == "student":
            self.config = {"residual": LinearHMMReparam(obs=StudentTReparam(), trans=StudentTReparam())}

    def model(self, zero_data, covariates):
        hmm = self.hmm.get_dist(duration=zero_data.size(-2))

        with pyro.poutine.reparam(config=self.config):
            self.predict(hmm, zero_data)


def main(**args):
    log_file = '{}.{}.{}.tt_{}_{}.nw_{}.sd_{}.nst_{}.cn_{:.1f}.lr_{:.2f}.lrd_{:.2f}.{}.log'
    log_file = log_file.format(args['dataset'], args['trans_noise'], args['obs_noise'],
                               args['train_window'], args['test_window'], args['num_windows'],
                               args['state_dim'], args['num_steps'],
                               args['clip_norm'], args['learning_rate'], args['learning_rate_decay'],
                               str(uuid.uuid4())[0:4])

    log = get_logger(args['log_dir'], log_file, use_local_logger=False)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    log(args)
    log("")

    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args['seed'])

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

    results = {}

    data, covariates = get_data(args=args)
    print("data, covariates", data.shape, covariates.shape)

    results = {}

    metrics = backtest(data, covariates,
                       lambda: Model(trans_noise=args['trans_noise'], state_dim=args['state_dim'],
                                     obs_noise=args['obs_noise'], obs_dim=data.size(-1)).cuda(),
                       train_window=None,
                       min_train_window=args['train_window'],
                       test_window=args['test_window'],
                       stride=args['stride'],
                       num_samples=args['num_eval_samples'],
                       forecaster_options=svi_forecaster_options,
                       forecaster_fn=Forecaster)

    log("### EVALUATION ###")
    for name in ["mae", "crps"]:
        values = [m[name] for m in metrics]
        mean, std = np.mean(values), np.std(values)
        results[name] = mean
        results[name + '_std'] = std
        log("{} = {:0.4g} +- {:0.4g}".format(name, mean, std))
    for name in ["mae_fine", "crps_fine"]:
        values = np.stack([m[name] for m in metrics])
        results[name] = values
        for t in range(values.shape[1]):
            mean = np.mean(values[:, t, :])
            std = np.std(values[:, t, :])
            metric_t = name[:-5] + '_{}'.format(t + 1)
            results[metric_t] = mean
            results[metric_t + '_std'] = std
            log("{} = {:0.4g} +- {:0.4g}".format(metric_t, mean, std))

    pred = np.stack([m['pred'].data.cpu().numpy() for m in metrics])
    results['pred'] = pred

    with open(args['log_dir'] + '/' + log_file[:-4] + '.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multivariate timeseries models")
    parser.add_argument("--trans-noise", default='stable', type=str, choices=['gaussian', 'stable', 'student'])
    parser.add_argument("--obs-noise", default='stable', type=str, choices=['gaussian', 'stable', 'student'])
    parser.add_argument("--dataset", default='metals', type=str)
    parser.add_argument("--data-dir", default='./data/', type=str)
    parser.add_argument("--log-dir", default='./logs/', type=str)
    parser.add_argument("--train-window", default=100, type=int)
    parser.add_argument("--test-window", default=5, type=int)
    parser.add_argument("--num-windows", default=2, type=int)
    parser.add_argument("--stride", default=5, type=int)
    parser.add_argument("--num-eval-samples", default=200, type=int)
    parser.add_argument("--clip-norm", default=20.0, type=float)
    parser.add_argument("-n", "--num-steps", default=3, type=int)
    parser.add_argument("-d", "--state-dim", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.005, type=float)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--log-every", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    main(**vars(args))
