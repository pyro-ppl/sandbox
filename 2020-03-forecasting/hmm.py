# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import uuid
import time

import numpy as np
import math
import torch
import torch.distributions.constraints as constraints
import torch.nn as nn

import pyro
from pyro.contrib.forecast import ForecastingModel, backtest, Forecaster
from pyro.nn import PyroParam, PyroModule
from pyro.infer.reparam import SymmetricStableReparam, StudentTReparam, LinearHMMReparam, StableReparam
from pyro.distributions import StudentT, Stable, Normal, LinearHMM, Gamma, LogNormal

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
        print("Initilized StableLinearHMM with state_dim = {}, obs_dim = {}".format(state_dim, obs_dim))
        assert trans_noise in ['gaussian', 'stable', 'student', 'skew']
        assert obs_noise in ['gaussian', 'stable', 'student', 'skew']
        super().__init__()
        self.obs_noise_scale = PyroParam(0.2 * torch.ones(obs_dim), constraint=constraints.positive)
        self.trans_noise_scale = PyroParam(0.2 * torch.ones(state_dim), constraint=constraints.positive)
        self.trans_matrix = PyroParam(0.3 * torch.randn(state_dim, state_dim))
        self.obs_matrix = PyroParam(0.3 * torch.randn(state_dim, obs_dim))
        if trans_noise in ["stable", "skew"]:
            self.trans_stability = PyroParam(torch.tensor(1.95), constraint=constraints.interval(1.01, 1.99))
            if trans_noise == "skew":
                self.trans_skew = PyroParam(torch.tensor(0.0), constraint=constraints.interval(-0.99, 0.99))
        elif trans_noise == "student":
            self.trans_nu = PyroParam(torch.tensor(5.0), constraint=constraints.interval(1.01, 30.0))
        if obs_noise in ["stable", "skew"]:
            self.obs_stability = PyroParam(torch.tensor(1.95), constraint=constraints.interval(1.01, 1.99))
            if obs_noise == "skew":
                self.obs_skew = PyroParam(torch.tensor(0.0), constraint=constraints.interval(-0.99, 0.99))
        elif obs_noise == "student":
            self.obs_nu = PyroParam(torch.tensor(5.0), constraint=constraints.interval(1.01, 30.0))

    def _get_init_dist(self):
        return Normal(torch.zeros(self.state_dim), torch.ones(self.state_dim)).to_event(1)

    def _get_obs_dist(self):
        if self.obs_noise == "stable":
            return Stable(self.obs_stability, torch.zeros(self.obs_dim),
                          scale=self.obs_noise_scale / root_two).to_event(1)
        elif self.obs_noise == "skew":
            return Stable(self.obs_stability, self.obs_skew,
                          scale=self.obs_noise_scale / root_two).to_event(1)
        elif self.obs_noise == "student":
            return StudentT(self.obs_nu, torch.zeros(self.obs_dim), self.obs_noise_scale).to_event(1)
        else:
            return Normal(torch.zeros(self.obs_dim), scale=self.obs_noise_scale).to_event(1)

    def _get_trans_dist(self):
        if self.trans_noise == "stable":
            return Stable(self.trans_stability, torch.zeros(self.state_dim),
                          scale=self.trans_noise_scale / root_two).to_event(1)
        elif self.trans_noise == "skew":
            return Stable(self.trans_stability, self.trans_skew,
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
    print("raw data", data.shape)

    to_keep = args['train_window'] + args['num_windows'] * args['test_window']
    assert to_keep <= data.size(0)

    data = data[:to_keep].float()

    data_mean = data.mean(0)
    data -= data_mean
    data_std = data.std(0)
    data /= data_std

    covariates = torch.zeros(data.size(0), 0)

    return data.double().cuda(), covariates.cuda()


class Model(ForecastingModel):
    def __init__(self, trans_noise="gaussian", obs_noise="gaussian", state_dim=3, obs_dim=14):
        super().__init__()
        self.trans_noise = trans_noise
        self.obs_noise = obs_noise
        self.hmm = StableLinearHMM(obs_dim=obs_dim, trans_noise=trans_noise, obs_noise=obs_noise, state_dim=state_dim)
        trans, obs = None, None

        if trans_noise == "stable":
            trans = SymmetricStableReparam()
        elif trans_noise == "skew":
            trans = StableReparam()
        elif trans_noise == "student":
            trans = StudentTReparam()
        if obs_noise == "stable":
            obs = SymmetricStableReparam()
        elif obs_noise == "skew":
            obs = StableReparam()
        elif obs_noise == "student":
            obs = StudentTReparam()

        self.config = {"residual": LinearHMMReparam(obs=obs, trans=trans)}

    def model(self, zero_data, covariates):
        hmm = self.hmm.get_dist(duration=zero_data.size(-2))

        with pyro.poutine.reparam(config=self.config):
            self.predict(hmm, zero_data)


class GuideConv(nn.Module):
    def __init__(self, obs_dim=4, num_channels=5, kernel_size=10, hidden_dim=40, num_layers=1, cat_obs=False, gamma_variates=4, distribution="gamma"):
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
    # torch.cuda.set_device(0)
    log_file = '{}.{}.{}.{}.tt_{}_{}.nw_{}.sd_{}.nst_{}.cn_{:.1f}.lr_{:.2f}.lrd_{:.2f}.seed_{}.arch_{}_{}_{}_{}_{}.{}.log'
    log_file = log_file.format(args['dataset'], args['trans_noise'], args['obs_noise'], args['guide'],
                               args['train_window'], args['test_window'], args['num_windows'],
                               args['state_dim'], args['num_steps'],
                               args['clip_norm'], args['learning_rate'], args['learning_rate_decay'],
                               args['seed'],
                               args['num_channels'], args['num_layers'], args['hidden_dim'], args['kernel_size'], args['cat_obs'],
                               str(uuid.uuid4())[0:4])

    log = get_logger(args['log_dir'], log_file, use_local_logger=False)

    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')

    log(args)
    log("")

    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args['seed'])

    t0 = time.time()

    guide_conv = None if args['guide'] == 'auto' else GuideConv(num_channels=args['num_channels'], kernel_size=args['kernel_size'],
                                                                hidden_dim=args['hidden_dim'], num_layers=args['num_layers'],
                                                                cat_obs=args['cat_obs'],
                                                                gamma_variates=4 if args['obs_noise']=='student' else args['state_dim'],
                                                                distribution=args['guide'][6:]).cuda()
    if args['guide'][:6] == 'custom':
        assert (args['trans_noise'] == 'student' and args['obs_noise'] == 'gaussian') or \
               (args['trans_noise'] == 'gaussian' and args['obs_noise'] == 'student')

    def guide(data, covariates, obs_dim=4):
        T = covariates.size(0)
        pyro.module("guide_conv", guide_conv)
        alpha, beta = guide_conv(data)
        #alpha = pyro.param("alpha", torch.zeros(T, 4))
        #beta = pyro.param("beta", torch.ones(T, 4), constraint=constraints.positive)
        #if torch.rand(1).item()<0.02:
        #    print("alpha, beta", alpha[41:44:, 1].data.cpu().numpy(), beta[41:44:, 1].data.cpu().numpy())
        #    print("data", data[41:44, 1].data.cpu().numpy())
        if args['guide'] == 'customgamma':
            if args['trans_noise'] == 'student':
                pyro.sample("residual_trans_gamma", Gamma(alpha, beta).to_event(2))
            elif args['obs_noise'] == 'student':
                pyro.sample("residual_obs_gamma", Gamma(alpha, beta).to_event(2))
        else:
            if args['trans_noise'] == 'student':
                pyro.sample("residual_trans_gamma", LogNormal(alpha, beta).to_event(2))
            elif args['obs_noise'] == 'student':
                pyro.sample("residual_obs_gamma", LogNormal(alpha, beta).to_event(2))

    def svi_forecaster_options(t0, t1, t2):
        num_steps = args['num_steps']  if t1 == args['train_window'] else 0
        lr = args['learning_rate']  #if t1 == args.train_window else 0.1 * args.learning_rate
        lrd = args['learning_rate_decay']  #if t1 == args.train_window else 0.1
        return {"num_steps": num_steps, "learning_rate": lr,
                "learning_rate_decay": lrd, "log_every": args['log_every'],
                "dct_gradients": False, "warm_start": False,
                "clip_norm": args['clip_norm'],
                "vectorize_particles": False,
                "num_particles": 1,
                "guide": None if args['guide'] == 'auto' else guide}

    data, covariates = get_data(args=args)
    results = {}

    metrics = backtest(data, covariates,
                       lambda: Model(trans_noise=args['trans_noise'], state_dim=args['state_dim'],
                                     obs_noise=args['obs_noise'], obs_dim=data.size(-1)).cuda(),
                       train_window=None,
                       seed=args['seed'],
                       min_train_window=args['train_window'],
                       test_window=args['test_window'],
                       stride=args['stride'],
                       num_samples=args['num_eval_samples'],
                       batch_size=1000,
                       amortized=True if args['guide']!='auto' else None,
                       forecaster_options=svi_forecaster_options,
                       forecaster_fn=Forecaster)

    num_eval_windows = (args['num_windows'] - 1) * args['test_window'] + 1
    pyro.set_rng_seed(0)
    index = torch.randperm(num_eval_windows)
    index_test = index[:math.ceil(0.80 * num_eval_windows)].data.cpu().numpy()
    index_val = index[math.ceil(0.80 * num_eval_windows):].data.cpu().numpy()

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
            metric_t = name[:-5] + '_{}'.format(t + 1)

            mean = np.mean(values[:, t, :])
            std = np.std(values[:, t, :])
            results[metric_t] = mean
            results[metric_t + '_std'] = std
            log("{} = {:0.4g} +- {:0.4g}".format(metric_t, mean, std))

            mean = np.mean(values[index_val, t, :])
            std = np.std(values[index_val, t, :])
            results[metric_t + '_val'] = mean
            results[metric_t + '_val_std'] = std
            log("{} = {:0.4g} +- {:0.4g}".format(metric_t + '_val', mean, std))

            mean = np.mean(values[index_test, t, :])
            std = np.std(values[index_test, t, :])
            results[metric_t + '_test'] = mean
            results[metric_t + '_test_std'] = std
            log("{} = {:0.4g} +- {:0.4g}".format(metric_t + '_test', mean, std))

    pred = np.stack([m['pred'].data.cpu().numpy() for m in metrics])
    results['pred'] = pred

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
    parser.add_argument("--trans-noise", default='gaussian', type=str, choices=['gaussian', 'stable', 'student', 'skew'])
    parser.add_argument("--obs-noise", default='student', type=str, choices=['gaussian', 'stable', 'student', 'skew'])
    parser.add_argument("--dataset", default='metals', type=str)
    parser.add_argument("--guide", default='customnormal', type=str, choices=['customgamma', 'customnormal', 'auto'])
    parser.add_argument("--data-dir", default='./data/', type=str)
    parser.add_argument("--log-dir", default='./logs/', type=str)
    parser.add_argument("--train-window", default=1000, type=int)
    parser.add_argument("--test-window", default=5, type=int)
    parser.add_argument("--num-windows", default=1, type=int)
    parser.add_argument("--num-channels", default=8, type=int)
    parser.add_argument("--kernel_size", default=8, type=int)
    parser.add_argument("--hidden-dim", default=32, type=int)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--num-eval-samples", default=1000, type=int)
    parser.add_argument("--clip-norm", default=10.0, type=float)
    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    parser.add_argument("-d", "--state-dim", default=5, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.003, type=float)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--cat-obs", action="store_true")
    parser.add_argument("--log-every", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    main(**vars(args))
