import argparse
import logging
import math
import operator
from collections import OrderedDict
from functools import reduce

import torch
import torch.nn as nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from preprocess import load_hourly_od
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam


# Work around slow lower_cholesky trnasform (.diag()....diag()).
# to_lower_cholesky = transform_to(constraints.lower_cholesky)
def to_lower_cholesky(x):
    return x.tril(-1) + x.diagonal(dim1=-2, dim2=-1).exp().diag_embed()


def inverse_sigmoid(y):
    return y.log() - (-y).log1p()


def bounded_exp(x, bound=1e5):
    return (x - math.log(bound)).sigmoid() * bound


def bounded_log(y, bound=1e5):
    return inverse_sigmoid(y / bound) + math.log(bound)


def _():
    test = torch.randn(10)
    assert (bounded_log(bounded_exp(test, 10), 10) - test).abs().max() < 1e-6


_()


def precision_to_scale_tril(P):
    L_inv = P.flip((-2, -1)).cholesky().flip((-2, -1)).transpose(-2, -1)
    L = torch.eye(P.size(-1)).triangular_solve(L_inv, upper=False).solution
    return L


def make_time_features(args, begin_time, end_time):
    time = torch.arange(begin_time, end_time, dtype=torch.float)
    time_mod_day = time / 24 % 1 * (2 * math.pi)
    time_mod_week = time / (24 * 7) % 1. * (2 * math.pi)
    features = torch.cat([
        make_seasonal_features(time_mod_day, order=12),
        make_seasonal_features(time_mod_week, order=8),
        make_trend_features(time, begin_time, end_time,
                            bandwidth=24 * 7)
    ], dim=-1)
    return features


def make_seasonal_features(signal, order):
    angles = signal.unsqueeze(-1) * torch.arange(1., 1. + order)
    return torch.cat([torch.cos(angles),
                      torch.sin(angles)], dim=-1)


def make_trend_features(signal, begin_time, end_time, bandwidth):
    num_points = int(math.ceil((end_time - begin_time) / bandwidth))
    logging.debug("Making {} global trend features".format(num_points))
    points = torch.linspace(begin_time, end_time, num_points)
    return ((signal.unsqueeze(-1) - points) / bandwidth).sigmoid()


def unpack_params(data, schema):
    assert isinstance(schema, OrderedDict)
    batch_shape = data.shape[:-1]
    offset = 0
    result = {}
    for name, shape in schema.items():
        numel = reduce(operator.mul, shape)
        chunk = data[..., offset: offset + numel]
        result[name] = chunk.reshape(batch_shape + shape)
        offset += numel
    return result


class Model:
    def __init__(self, args, begin_epoch, end_epoch, num_stations, feature_dim):
        self.begin_epoch = begin_epoch
        self.end_epoch = end_epoch
        self.num_stations = num_stations
        state_dim = args.state_dim
        gate_rate_dim = num_stations ** 2 * 2
        self.schema = OrderedDict([
            ("init_loc", (state_dim,)),
            ("init_scale_tril", (state_dim, state_dim)),
            ("trans_matrix", (state_dim, state_dim)),
            ("trans_loc", (state_dim,)),
            ("trans_scale_tril", (state_dim, state_dim)),
            ("obs_matrix", (state_dim, gate_rate_dim)),
            ("obs_loc", (gate_rate_dim,)),
            ("obs_scale", (gate_rate_dim,)),
        ])
        output_dim = sum(reduce(operator.mul, shape)
                         for shape in self.schema.values())
        self.nn = nn.Sequential(
            nn.Linear(feature_dim, args.model_nn_dim),
            nn.Sigmoid(),
            nn.Linear(args.model_nn_dim, output_dim))

    def __call__(self, time_features, trip_counts):
        num_hours, num_origins, num_destins = trip_counts.shape
        assert num_origins == self.num_stations
        assert num_destins == self.num_stations

        # Sample static parameters.
        init_loc = pyro.param("init_loc", lambda: torch.zeros(args.state_dim))
        init_scale_tril = pyro.param("init_scale_tril",
                                     lambda: torch.eye(args.state_dim, args.state_dim),
                                     constraint=constraints.lower_cholesky)
        init_dist = dist.MultivariateNormal(init_loc, scale_tril=init_scale_tril)

        # Sample time-varying parameters using a neural net.
        pyro.module("model_nn", self.nn)
        params = unpack_params(self.nn(time_features), self.schema)
        trans_matrix = params["trans_matrix"] + torch.eye(args.state_dim)
        trans_scale_tril = to_lower_cholesky(params["trans_scale_tril"])
        trans_dist = dist.MultivariateNormal(params["trans_loc"],
                                             scale_tril=trans_scale_tril)
        obs_matrix = params["obs_matrix"]
        obs_loc = params["obs_loc"]
        obs_scale = bounded_exp(params["obs_scale"])
        obs_dist = dist.Normal(obs_loc, obs_scale).to_event(1)

        # The model performs exact inference over a time-varying latent state.
        gate_rate = pyro.sample("gate_rate",
                                dist.GaussianHMM(init_dist,
                                                 trans_matrix, trans_dist,
                                                 obs_matrix, obs_dist))
        gate_rate = gate_rate.reshape(trip_counts.shape + (2,))
        gate = gate_rate[..., 0].sigmoid()
        rate = bounded_exp(gate_rate[..., 1])

        with pyro.plate("time", num_hours, dim=-3):
            with pyro.plate("origins", num_origins, dim=-2):
                with pyro.plate("destins", num_destins, dim=-1):
                    pyro.sample("trip_count",
                                dist.ZeroInflatedPoisson(gate, rate),
                                obs=trip_counts)


class Guide:
    def __init__(self, args, num_stations, feature_dim):
        self.nn = nn.Sequential(
            nn.Linear(feature_dim + num_stations ** 2, args.guide_nn_dim),
            nn.Sigmoid(),
            nn.Linear(args.guide_nn_dim, num_stations ** 2 * 2))

    def __call__(self, time_features, trip_counts):
        pyro.module("guide_mm", self.nn)
        batch_shape = trip_counts.shape[:-2]
        nn_input = torch.cat([time_features,
                              trip_counts.reshape(batch_shape + (-1,))], dim=-1)
        gate_rate = self.nn(nn_input)
        pyro.sample("gate_rate", dist.Delta(gate_rate, event_dim=2))


def make_minibatch(rows, begin_time, end_time, stations):
    time = rows[:, 0]
    rows = rows[(begin_time <= time) & (time < end_time)]
    time, origin, destin, count = rows.t()
    batch = torch.zeros(end_time - begin_time, len(stations), len(stations))
    batch[time - begin_time, origin, destin] = count.float()
    return batch


def train(args, dataset):
    rows = dataset["rows"]
    times = rows[:, 0]
    begin_epoch = times.min().item()
    end_epoch = 1 + times.max().item()
    num_stations = len(dataset["stations"])
    logging.debug("Training on {} stations in time range [{}, {})"
                  .format(num_stations, begin_epoch, end_epoch))
    time_features = make_time_features(args, begin_epoch, end_epoch)
    feature_dim = time_features.size(-1)

    model = Model(args, begin_epoch, end_epoch, num_stations, feature_dim)
    guide = Guide(args, num_stations, feature_dim)
    elbo = Trace_ELBO()
    optim = ClippedAdam({"lr": args.learning_rate})
    svi = SVI(model, guide, optim, elbo)
    losses = []
    for epoch in range(args.num_epochs):
        begin_time = begin_epoch
        epoch_loss = 0.
        while begin_time < end_epoch:
            end_time = min(begin_time + args.batch_size, end_epoch)
            feature_batch = time_features[begin_time - begin_epoch:end_time - begin_epoch]
            trip_counts = make_minibatch(rows, begin_time, end_time, dataset["stations"])
            loss = svi.step(feature_batch, trip_counts)
            losses.append(loss)
            epoch_loss += loss
        logging.info("epoch {} loss = {:0.4g}".format(epoch, loss))
    return losses


def main(args):
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.seed)

    datasets_by_year = load_hourly_od(args)
    dataset = datasets_by_year[0]
    train(args, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BART origin-destination forecast")
    parser.add_argument("--state-dim", default="16", type=int,
                        help="size of HMM state space in model")
    parser.add_argument("--model-nn-dim", default="64", type=int,
                        help="size of hidden layer in model net")
    parser.add_argument("--guide-nn-dim", default="64", type=int,
                        help="size of hidden layer in guide net")
    parser.add_argument("-n", "--num-epochs", default=1001, type=int)
    parser.add_argument("-b", "--batch-size", default=400, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("--seed", default=123456789, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format='%(relativeCreated) 9d %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    main(args)
