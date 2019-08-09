import argparse
import logging
import math
import operator
from collections import OrderedDict
from functools import reduce

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from preprocess import load_hourly_od


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


class ModelNet(nn.Module):
    def __init__(self, args, feature_dim, schema):
        super().__init__()
        self.schema = schema

    def forward(self, features):
        out = self.nn(features)
        return unpack_params(out, self.schema)

    @classmethod
    def make(cls, *args):
        if not hasattr(cls, "instance"):
            cls.instance = ModelNet(*args)
        return cls.instance


class Model:
    def __init__(self, args, begin_epoch, end_epoch, num_stations):
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
            ("obs_scale_tril", (gate_rate_dim, gate_rate_dim)),
        ])

        output_dim = sum(reduce(operator.mul, shape)
                         for shape in self.schema.values)
        feature_dim = self._make_features(torch.arange(1.)).size(-1)
        self.mpl = nn.Sequential(
            nn.Linear(feature_dim, args.model_nn_dim),
            nn.Sigmoid(),
            nn.Linear(args.model_nn_dim, output_dim))

    def _make_features(self, time):
        assert isinstance(time, torch.Tensor)
        assert time.dim() == 1
        time_mod_day = time / 24 % 1 * (2 * math.pi)
        time_mod_week = time / (24 * 7) % 1. * (2 * math.pi)
        features = torch.cat([
            make_seasonal_features(time_mod_day, order=12),
            make_seasonal_features(time_mod_week, order=8),
            make_trend_features(time, self.begin_epoch, self.end_epoch,
                                bandwidth=24 * 7)
        ])
        return features

    def __call__(self, batch):
        num_hours, num_origins, num_destins = batch.shape
        assert num_origins == self.num_stations
        assert num_destins == self.num_stations

        # Construct time features for the neural network.

        # We sample time-varying parameters using a neural net.
        pyro.module("model_nn", self.nn)
        time = batch[:, 0]
        features = self.make_features(time)
        params = self.nn(features)
        init_dist = dist.MultivariateNormal(params["init_loc"],
                                            params["init_scale_tril"].tril())
        trans_matrix = params["trans_matrix"] + torch.eye(args.state_dim)
        trans_dist = dist.MultivariateNormal(params["trans_loc"],
                                             params["trans_scale_tril"].tril())
        obs_matrix = params["obs_matrix"]
        obs_dist = dist.MultivariateNormal(params["obs_loc"],
                                           params["obs_scale_tril"].tril())

        # The model performs exact inference over a time-varying latent state.
        gate_rate = pyro.sample("gate_rate",
                                dist.GaussianHMM(init_dist,
                                                 trans_matrix, trans_dist,
                                                 obs_matrix, obs_dist))
        gate = gate_rate[..., 0]
        rate = gate_rate[..., 1].sigmoid() * 1e5

        with pyro.plate("time", num_hours, dim=-3):
            with pyro.plate("origins", num_origins, dim=-2):
                with pyro.plate("destins", num_destins, dim=-1):
                    pyro.sample("trip_count",
                                dist.ZeroInflatedPoisson(gate, rate),
                                obs=batch)


class Guide:
    def __init__(self, args, num_stations):
        self.nn = nn.Sequential(
            nn.Linear(num_stations ** 2, args.guide_nn_dim),
            nn.Sigmoid(),
            nn.Linear(args.guide_nn_dim, num_stations ** 2 * 2))

    def __call__(self, batch):
        pyro.module("guide_mm", self.nn)
        batch_shape = batch.shape[:-2]
        gate_rate = self.nn(batch.reshape(batch_shape + (-1,)))
        gate_rate = gate_rate.reshape(batch.shape + (2,))
        pyro.sample("gate_rate", dist.Delta(gate_rate, event_dim=3))


def make_minibatch(rows, begin_time, end_time, stations):
    time = rows[:, 0]
    rows = rows[(begin_time <= time) & (time < end_time)]
    time, origin, destin, count = rows.t()
    batch = torch.zeros(end_time - begin_time, len(stations), len(stations))
    batch[time - begin_time, origin, destin] = count.float()
    return batch


def make_seasonal_features(signal, order):
    angles = signal.unsqueeze(-1) * torch.arange(1., 1. + order)
    return torch.cat([torch.cos(angles),
                      torch.sin(angles)], dim=-1)


def make_trend_features(signal, bandwidth):
    raise NotImplementedError("TODO")


def train(args, dataset):
    data = dataset["rows"]
    times = data[:, 0]
    begin_epoch = times.min().item()
    end_epoch = 1 + times.max().item()
    num_stations = data.size(-1)
    logging.debug("Training on {} stations in time range [{}, {})"
                  .format(num_stations, begin_epoch, end_epoch))

    model = Model(args, begin_epoch, end_epoch)
    guide = Guide(args, num_stations)
    elbo = Trace_ELBO()
    optim = ClippedAdam({"lr": args.learning_Rate})
    svi = SVI(model, guide, optim, elbo)
    losses = []
    for epoch in range(args.num_epochs):
        begin_time = begin_epoch
        epoch_loss = 0.
        while begin_time < end_epoch:
            end_time = min(begin_time + args.batch_size, end_epoch)
            batch = make_minibatch(data, begin_time, end_time)
            loss = svi.step(batch)
            losses.append(loss)
            epoch_loss += loss
        logging.info("epoch {} loss = {:0.4g}".format(epoch, loss))
    return losses


def main(args):
    yearly_datasets = load_hourly_od(args)
    dataset = yearly_datasets[0]
    train(args, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BART origin-destination forecast")
    parser.add_argument("--state-dim", default="32", type=int,
                        help="size of HMM state space in model")
    parser.add_argument("--hidden-dim", default="128", type=int,
                        help="size of hidden layer in model net")
    parser.add_argument("-n", "--num-epochs", default=1001, type=int)
    parser.add_argument("-b", "--batch-size", default=5600, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format='%(relativeCreated) 9d %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    main(args)
