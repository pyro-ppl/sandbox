# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import uuid
import time

import numpy as np
import math
import torch
import torch.distributions.constraints as constraints
from torch.nn.functional import softplus

import pyro
from pyro.contrib.forecast import ForecastingModel, backtest, Forecaster
from pyro.nn import PyroParam, PyroModule
from pyro.infer.reparam import SymmetricStableReparam, StudentTReparam, LinearHMMReparam, StableReparam
from pyro.distributions import StudentT, Stable, Normal, LinearHMM

import pickle
from dataloader import get_data as get_raw_data
from logger import get_logger
from pyro.ops.tensor_utils import periodic_cumsum, periodic_features, periodic_repeat


root_two = math.sqrt(2.0)
num_stations = 1
day = 24 * 60


class StableLinearHMM(PyroModule):
    def __init__(self, obs_dim=1, trans_noise="gaussian", obs_noise="gaussian", state_dim=3):
        self.trans_noise = trans_noise
        self.obs_noise = obs_noise
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        print("Initialized StableLinearHMM with state_dim = {}, obs_dim = {}".format(state_dim, obs_dim))
        assert trans_noise in ['gaussian', 'stable', 'student', 'skew']
        assert obs_noise == 'gaussian'
        super().__init__()
        #self.obs_noise_scale = PyroParam(0.2 * torch.ones(obs_dim), constraint=constraints.positive)
        self.trans_noise_scale = PyroParam(0.2 * torch.ones(state_dim), constraint=constraints.positive)
        self.trans_matrix = PyroParam(0.3 * torch.randn(state_dim, state_dim))
        self.obs_matrix = PyroParam(0.3 * torch.randn(state_dim, num_stations))
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

    def _get_obs_dist(self, obs_scale=None):
        #if self.obs_noise == "stable":
        #    return Stable(self.obs_stability, torch.zeros(self.obs_dim),
        #                  scale=self.obs_noise_scale / root_two).to_event(1)
        #elif self.obs_noise == "skew":
        #    return Stable(self.obs_stability, self.obs_skew,
        #                  scale=self.obs_noise_scale / root_two).to_event(1)
        #elif self.obs_noise == "student":
        #    return StudentT(self.obs_nu, torch.zeros(self.obs_dim), self.obs_noise_scale).to_event(1)
        #else:
        return Normal(torch.zeros(self.obs_dim), scale=obs_scale).to_event(1)

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

    def get_dist(self, duration=None, obs_scale=None, obs_matrix_multiplier=None):
        obs_matrix = self.obs_matrix.repeat_interleave(day, dim=-1)
        #print("obs_matrix", obs_matrix.shape)
        #print("obs_matrix_multiplier",obs_matrix_multiplier.shape)
        #print("obs_scale", obs_scale.shape)
        return LinearHMM(self._get_init_dist(), self.trans_matrix, self._get_trans_dist(),
                         obs_matrix_multiplier * obs_matrix, self._get_obs_dist(obs_scale), duration=duration)


def leftequal(x, delta):
    b = (x[delta:] == x[:-delta])
    res = np.concatenate([np.array([0.0] * delta, dtype=np.uint8), b])
    return res[:x.shape[0]]

def rightequal(x, delta):
    b = (x[delta:] == x[:-delta])
    res = np.concatenate([b, np.array([0.0] * delta, dtype=np.uint8)])
    return res[:x.shape[0]]

def extract_night_features(data):
    data = data.cpu().numpy()
    nightmin = np.min(data)

    darklast = (data == nightmin)
    for minute in range(1, 120):
        darklast = darklast & leftequal(data, minute)

    darknext = (data == nightmin)
    for minute in range(1, 120):
        darknext = darknext & rightequal(data, minute)

    dark = darklast | darknext
    darkshift = np.concatenate([np.zeros(day, dtype=np.uint8), dark[:-day]])
    darkshiftshift = np.concatenate([np.zeros(2 * day, dtype=np.uint8), dark[:-2 * day]])
    dark = dark & darkshift & darkshiftshift
    dark = np.concatenate([np.zeros(day, dtype=np.float32), dark[:-day]])

    return torch.tensor(dark)


def get_data(args=None):
    data, _, _, _ = get_raw_data(args['dataset'], args['data_dir'])
    data = data[..., :num_stations]
    print("raw data", data.shape)

    night_features = torch.zeros(data.shape)
    for dim in range(data.size(-1)):
        night_features[:, dim] = extract_night_features(data[:, dim])

    to_drop = 194 * day - 1
    to_keep = day * (args['train_window'] + args['num_windows'] * args['test_window'])
    assert to_keep + to_drop <= data.size(0)

    data = data[to_drop:to_drop + to_keep].float()
    covariates = periodic_features(data.size(0), day, 10)
    data = data.reshape(data.size(0) // day, day * data.size(-1))

    night_features = night_features[to_drop:to_drop + to_keep].double()
    night_features = night_features.reshape(night_features.size(0) // day, day * night_features.size(-1))

    covariates = covariates.reshape(covariates.size(0) // day, day * covariates.size(-1))
    print("periodic, night", covariates.shape, night_features.shape)
    covariates = torch.cat([night_features, covariates], dim=-1)

    print("covariates, data", covariates.shape, data.shape)

    return data.cuda(), covariates.cuda()


class Model(ForecastingModel):
    def __init__(self, trans_noise="gaussian", obs_noise="gaussian", state_dim=3, obs_dim=14, nightval=0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.trans_noise = trans_noise
        self.obs_noise = obs_noise
        self.nightval = nightval
        self.hmm = StableLinearHMM(obs_dim=obs_dim, trans_noise=trans_noise, obs_noise=obs_noise, state_dim=state_dim)
        trans, obs = None, None

        self.mean_granularity = 24 * 6
        self.night_scale = PyroParam(0.1 * torch.ones(num_stations), constraint=constraints.positive)
        self.obs_scale = PyroParam(0.2 * torch.ones(self.mean_granularity, num_stations), constraint=constraints.positive)

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
        night_covariates = covariates[:, :self.obs_dim]
        short_covariates = covariates[:, self.obs_dim:]
        night_covariates = night_covariates.reshape(night_covariates.size(0) * day, num_stations)
        short_covariates = short_covariates.reshape(short_covariates.size(0) * day, short_covariates.size(-1) // day)
        #print("short_covariates, night_covariates", short_covariates.shape, night_covariates.shape)
        day_covariates = 1.0 - night_covariates
        T_minutes = short_covariates.size(0)

        hour = torch.arange(self.mean_granularity).repeat(1 + T_minutes // self.mean_granularity)[:T_minutes]
        daily_weights = pyro.sample("daily_weights", Normal(0, 0.1).expand([short_covariates.size(-1), num_stations]).to_event(2))
        #print("daily_weights", daily_weights.shape)

        obs_scale = self.obs_scale.index_select(0, hour)
        #print("obs_scale", obs_scale.shape)
        obs_scale = day_covariates * obs_scale + night_covariates * self.night_scale
        obs_scale = obs_scale.reshape(obs_scale.size(0) // day, self.obs_dim)
        #print("obs_scale2", obs_scale.shape)

        periodic = torch.einsum('...fs,tf->...ts', daily_weights, short_covariates)
        if periodic.dim() > 3:
            periodic = periodic.squeeze(-3)
        #print("periodic", periodic.shape)

        periodic = day_covariates * periodic + night_covariates * self.nightval
        #print("periodic", periodic.shape)
        periodic = periodic.reshape(periodic.shape[:-2] + (periodic.size(-2) // day, self.obs_dim))
        if periodic.dim() == 3:
            periodic = periodic.squeeze(0)
        #print("periodic2", periodic.shape)
        obs_matrix_multiplier = day_covariates.reshape(day_covariates.size(0) // day, self.obs_dim).unsqueeze(-2)
        #print("obs_matrix_multiplier", obs_matrix_multiplier.shape)

        hmm = self.hmm.get_dist(duration=zero_data.size(-2), obs_scale=obs_scale,
                                obs_matrix_multiplier=obs_matrix_multiplier)

        with pyro.poutine.reparam(config=self.config):
            self.predict(hmm, periodic)


def main(**args):
    #torch.cuda.set_device(0)
    log_file = '{}.{}.{}.tt_{}_{}.nw_{}.sd_{}.nst_{}.cn_{:.1f}.lr_{:.2f}.lrd_{:.2f}.seed_{}.{}.log'
    log_file = log_file.format(args['dataset'], args['trans_noise'], args['obs_noise'],
                               args['train_window'], args['test_window'], args['num_windows'],
                               args['state_dim'], args['num_steps'],
                               args['clip_norm'], args['learning_rate'], args['learning_rate_decay'],
                               args['seed'],
                               str(uuid.uuid4())[0:4])

    log = get_logger(args['log_dir'], log_file, use_local_logger=False)

    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #torch.set_default_tensor_type('torch.DoubleTensor')
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
    #data = data.log1p()
    #data = torch.nn.functional.softplus(data)
    data = data.expm1().log()
    results = {}

    def transform(pred, truth):
         return softplus(pred), softplus(truth)
         #return pred.expm1().clamp(min=1.0e-8).log(), truth.expm1().clamp(min=1.0e-8).log()
#        return pred.expm1().clamp(min=0.0), truth.expm1()

    metrics = backtest(data, covariates,
                       lambda: Model(trans_noise=args['trans_noise'], state_dim=args['state_dim'],
                                     obs_noise=args['obs_noise'], obs_dim=data.size(-1), nightval=data.min()).cuda(),
                       train_window=None,
                       seed=args['seed'],
                       min_train_window=args['train_window'],
                       test_window=args['test_window'],
                       stride=args['stride'],
                       num_samples=args['num_eval_samples'],
                       batch_size=1,
                       transform=transform,
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
    for name in []:
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

    #with open(args['log_dir'] + '/' + log_file[:-4] + '.pkl', 'wb') as f:
    #    pickle.dump(results, f, protocol=2)

    log("[ELAPSED TIME]: {:.3f}".format(time.time() - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multivariate timeseries models")
    parser.add_argument("--trans-noise", default='student', type=str, choices=['gaussian', 'stable', 'student', 'skew'])
    parser.add_argument("--obs-noise", default='gaussian', type=str, choices=['gaussian', 'stable', 'student', 'skew'])
    parser.add_argument("--dataset", default='solar', type=str)
    parser.add_argument("--data-dir", default='./data/', type=str)
    parser.add_argument("--log-dir", default='./logs/', type=str)
    parser.add_argument("--train-window", default=92, type=int)
    parser.add_argument("--test-window", default=5, type=int)
    parser.add_argument("--num-windows", default=1, type=int)
    parser.add_argument("--stride", default=day, type=int)
    parser.add_argument("--num-eval-samples", default=300, type=int)
    parser.add_argument("--clip-norm", default=10.0, type=float)
    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    parser.add_argument("-d", "--state-dim", default=num_stations + 1, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.03, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.003, type=float)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--log-every", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    main(**vars(args))
