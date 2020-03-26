# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import uuid

import numpy as np
import math

import torch
from torch.distributions import constraints, transform_to
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.contrib.forecast import ForecastingModel, backtest
from pyro.contrib.timeseries import IndependentMaternGP
from pyro.nn import PyroParam
from pyro.infer.reparam import SymmetricStableReparam, StableReparam, StudentTReparam
from pyro.infer.reparam import LinearHMMReparam
from pyro.poutine import reparam
from pyro.distributions import MultivariateNormal, LogNormal, StudentT, Stable, Normal, Gamma, TransformedDistribution

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
        assert obs_noise in ['gaussian', 'student', 'stable', 'skew']
        super().__init__(nu=nu, dt=dt, obs_dim=obs_dim,
                         length_scale_init=length_scale_init,
                         kernel_scale_init=kernel_scale_init,
                         obs_noise_scale_init=obs_noise_scale_init)
        if obs_noise == "student":
           self.nu = PyroParam(torch.tensor(5.0), constraint=constraints.interval(1.01, 30.0))
        elif obs_noise == "stable":
           self.stability = PyroParam(torch.tensor(1.95), constraint=constraints.interval(1.01, 1.99))
        elif obs_noise == "skew":
           self.stability = PyroParam(torch.tensor(1.95), constraint=constraints.interval(1.01, 1.99))
           self.skew = PyroParam(torch.tensor(0.0), constraint=constraints.interval(-0.99, 0.99))

    def _get_init_dist(self):
        cov = self.kernel.stationary_covariance()
        return MultivariateNormal(self.obs_matrix.new_zeros(self.obs_dim, self.kernel.state_dim), cov)

    def _get_obs_dist(self):
        scale = self.obs_noise_scale.unsqueeze(-1).unsqueeze(-1)
        if self.obs_noise == "stable" or self.obs_noise == "skew":
            stability = self.stability.unsqueeze(-1).unsqueeze(-1)
            skew = 0.0 if self.obs_noise == "stable" else self.skew.unsqueeze(-1).unsqueeze(-1)
            return Stable(stability, skew, scale=scale / root_two).to_event(1)
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
    def __init__(self, obs_noise="gaussian", nu=1.5):
        super().__init__()
        self.obs_noise = obs_noise
        self.noise_gp = IndependentMaternStableProcess(obs_dim=1, obs_noise=obs_noise, nu=nu,
                                                       length_scale_init=torch.tensor([10.0]))
        if obs_noise == "gaussian":
            self.config = {"residual": LinearHMMReparam()}
        elif obs_noise == "stable":
            self.config = {"residual": LinearHMMReparam(obs=SymmetricStableReparam())}
        elif obs_noise == "skew":
            self.config = {"residual": LinearHMMReparam(obs=StableReparam())}
        elif obs_noise == "student":
            self.config = {"residual": LinearHMMReparam(obs=StudentTReparam())}

    def model(self, zero_data, covariates):
        noise_dist = self.noise_gp.get_dist(duration=zero_data.size(-2))
        noise_dist = dist.IndependentHMM(noise_dist)

        with reparam(config=self.config):
            self.predict(noise_dist, zero_data)


class GuideConv(nn.Module):
    def __init__(self, obs_dim=1, num_channels=4, kernel_size=8, hidden_dim=16, num_layers=2, cat_obs=False, distribution="student"):
        super().__init__()
        self.obs_dim = obs_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.cat_obs = cat_obs
        self.distribution = distribution
        num_params = {'student': 2, 'stable': 4, 'skew': 8}[distribution]
        nn_dim = num_channels if not cat_obs else num_channels + obs_dim
        if num_layers == 1:
            self.fc = nn.Linear(nn_dim, num_params)
        elif num_layers == 2:
            self.fc = nn.Linear(nn_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_params)
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
        if self.distribution == 'student':
            h = nn.functional.softplus(h)
            return h[:, 0:1].clamp(min=0.1), h[:, 1:2].clamp(min=0.1)
        elif self.distribution == 'stable':
            return h[:, 0:1], nn.functional.softplus(h[:, 1:2]).clamp(min=0.01), \
                   h[:, 2:3], nn.functional.softplus(h[:, 3:4]).clamp(min=0.01)
        elif self.distribution == 'skew':
            return h[:, 0:1], nn.functional.softplus(h[:, 1:2]).clamp(min=0.01), \
                   h[:, 2:3], nn.functional.softplus(h[:, 3:4]).clamp(min=0.01), \
                   h[:, 4:5], nn.functional.softplus(h[:, 5:6]).clamp(min=0.01), \
                   h[:, 6:7], nn.functional.softplus(h[:, 7:8]).clamp(min=0.01)


def main(**args):
    log_file = 'amogp.{}.nu_{}.tt_{}_{}.arch_{}_{}_{}.co_{}.seed_{}.{}.log'
    log_file = log_file.format(args['obs_noise'], args['nu'],
                               args['train_window'], args['test_window'],
                               args['num_channels'], args['kernel_size'], args['hidden_dim'],
                               args['cat_obs'], args['seed'],
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
                                                                        cat_obs=args['cat_obs'],
                                                                        distribution=args['obs_noise']).cuda()

    def amortized_student_guide(data, covariates):
        pyro.module("guide_conv", guide_conv)
        alpha, beta = guide_conv(data)
        pyro.sample("residual_obs_gamma", Gamma(alpha.unsqueeze(0), beta.unsqueeze(0)).to_event(3))

    def amortized_stable_guide(data, covariates):
        pyro.module("guide_conv", guide_conv)
        uni_loc, uni_scale, exp_loc, exp_scale = guide_conv(data)

        uni_base = Normal(uni_loc.unsqueeze(0), uni_scale.unsqueeze(0))
        transform = transform_to(constraints.interval(-math.pi / 2.0, math.pi / 2.0))
        q_uni = TransformedDistribution(uni_base.to_event(3), transform)
        pyro.sample("residual_obs_uniform", q_uni)

        q_exp = LogNormal(exp_loc.unsqueeze(0), exp_scale.unsqueeze(0)).to_event(3)
        pyro.sample("residual_obs_exponential", q_exp)

    def amortized_skew_guide(data, covariates):
        pyro.module("guide_conv", guide_conv)
        uni_t_loc, uni_t_scale, exp_t_loc, exp_t_scale, uni_z_loc, uni_z_scale, exp_z_loc, exp_z_scale = guide_conv(data)
        transform = transform_to(constraints.interval(-math.pi / 2.0, math.pi / 2.0))

        uni_t_base = Normal(uni_t_loc.unsqueeze(0), uni_t_scale.unsqueeze(0))
        q_uni_t = TransformedDistribution(uni_t_base.to_event(3), transform)
        pyro.sample("residual_obs_t_uniform", q_uni_t)
        q_exp_t = LogNormal(exp_t_loc.unsqueeze(0), exp_t_scale.unsqueeze(0)).to_event(3)
        pyro.sample("residual_obs_t_exponential", q_exp_t)

        uni_z_base = Normal(uni_z_loc.unsqueeze(0), uni_z_scale.unsqueeze(0))
        q_uni_z = TransformedDistribution(uni_z_base.to_event(3), transform)
        pyro.sample("residual_obs_z_uniform", q_uni_z)
        q_exp_z = LogNormal(exp_z_loc.unsqueeze(0), exp_z_scale.unsqueeze(0)).to_event(3)
        pyro.sample("residual_obs_z_exponential", q_exp_z)

    guide = {'gaussian': None,
             'student': amortized_student_guide,
             'skew': amortized_skew_guide,
             'stable': amortized_stable_guide}[args['obs_noise']]

    def svi_forecaster_options(t0, t1, t2):
        num_steps = args['num_steps']  if t1 == args['train_window'] else 0
        return {"num_steps": num_steps, "learning_rate": args['learning_rate'],
                "learning_rate_decay": args['learning_rate_decay'], "log_every": 20,
                "dct_gradients": False, "warm_start": False,
                "clip_norm": args['clip_norm'],
                "vectorize_particles": False,
                "num_particles": 1, "guide": guide}

    metrics = backtest(data, covariates,
                       lambda: Model(obs_noise=args['obs_noise'], nu=args['nu']),
                       train_window=None,
                       min_train_window=args['train_window'],
                       test_window=args['test_window'],
                       stride=args['stride'],
                       batch_size=200,
                       amortized=True,
                       num_samples=args['num_eval_samples'],
                       forecaster_options=svi_forecaster_options)

    log("### EVALUATION ###")
    for name in ["mae", "crps"]:
        values = [m[name] for m in metrics]
        mean, std = np.mean(values), np.std(values)
        results[name] = mean
        results[name + '_std'] = std
        log("{} = {:0.6g} +- {:0.6g}".format(name, mean, std))
    for name in ["mae_fine", "crps_fine"]:
        values = np.stack([m[name] for m in metrics])
        results[name] = values

        pyro.set_rng_seed(0)
        index = torch.randperm(values.shape[0])
        index_test = index[:math.ceil(0.80 * values.shape[0])].data.cpu().numpy()
        index_val = index[math.ceil(0.80 * values.shape[0]):].data.cpu().numpy()

        for t in range(values.shape[1]):
            metric_t = name[:-5] + '_{}'.format(t + 1)

            mean = np.mean(values[:, t, :])
            std = np.std(values[:, t, :])
            results[metric_t] = mean
            results[metric_t + '_std'] = std
            log("{} = {:0.6g} +- {:0.6g}".format(metric_t, mean, std))

            mean = np.mean(values[index_val, t, :])
            std = np.std(values[index_val, t, :])
            results[metric_t + '_val'] = mean
            results[metric_t + '_val_std'] = std
            log("{} = {:0.6g} +- {:0.6g}".format(metric_t + '_val', mean, std))

            mean = np.mean(values[index_test, t, :])
            std = np.std(values[index_test, t, :])
            results[metric_t + '_test'] = mean
            results[metric_t + '_test_std'] = std
            log("{} = {:0.6g} +- {:0.6g}".format(metric_t + '_test', mean, std))

    pred = np.stack([m['pred'].data.cpu().numpy() for m in metrics])
    results['pred'] = pred[:, :, :, 0]

    for name, value in pyro.get_param_store().items():
        if value.numel() == 1:
            results[name] = value.item()
            log("[{}]".format(name), value.item())

    with open(args['log_dir'] + '/' + log_file[:-4] + '.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multivariate timeseries models")
    parser.add_argument("--obs-noise", default='gaussian', type=str,
                        choices=['gaussian', 'stable', 'student', 'skew'])
    parser.add_argument("--train-window", default=1100*48-40, type=int)
    parser.add_argument("--test-window", default=5, type=int)
    parser.add_argument("--stride", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('-ld', '--log-dir', type=str, default="./logs/")
    parser.add_argument('--data-dir', type=str, default="./data/")
    parser.add_argument("--num-channels", default=8, type=int)
    parser.add_argument("--nu", default=1.5, type=float, choices=[0.5, 1.5, 2.5])
    parser.add_argument("--kernel_size", default=8, type=int)
    parser.add_argument("--hidden-dim", default=32, type=int)
    parser.add_argument("--num-eval-samples", default=400, type=int)
    parser.add_argument("--clip-norm", default=10.0, type=float)
    parser.add_argument("--cat-obs", default=0, type=int)
    parser.add_argument("-n", "--num-steps", default=301, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.001, type=float)
    args = parser.parse_args()

    main(**vars(args))
