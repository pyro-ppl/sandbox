import argparse
import logging
import math
import operator
from collections import OrderedDict
from functools import reduce

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from torch.distributions import constraints

from preprocess import load_hourly_od


def vm(vector, matrix):
    return vector.unsqueeze(-2).matmul(matrix).squeeze(-2)


def rms(tensor):
    return tensor.pow(2).mean().sqrt()


def inverse_sigmoid(y):
    return y.log() - (-y).log1p()


def bounded_exp(x, bound=1e5):
    return (x - math.log(bound)).sigmoid() * bound


def bounded_log(y, bound=1e5):
    return inverse_sigmoid(y / bound) + math.log(bound)


test = torch.randn(10)
assert (bounded_log(bounded_exp(test, 10), 10) - test).abs().max() < 1e-6
del test


def make_time_features(args, begin_time, end_time):
    time = torch.arange(begin_time, end_time, dtype=torch.float)
    time_mod_week = time / (24 * 7) % 1. * (2 * math.pi)
    features = torch.cat([
        make_seasonal_features(time_mod_week, order=24 * 7 / 2),
        make_global_trend_features(time, begin_time, end_time,
                                   bandwidth=24 * 7)
    ], dim=-1)
    return features


def make_seasonal_features(signal, order):
    angles = signal.unsqueeze(-1) * torch.arange(1., 1. + order)
    return torch.cat([torch.cos(angles),
                      torch.sin(angles)], dim=-1)


def make_global_trend_features(signal, begin_time, end_time, bandwidth):
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
    def __init__(self, args, num_stations, feature_dim):
        self.num_stations = num_stations
        state_dim = args.state_dim
        gate_rate_dim = num_stations ** 2 * 2
        self.schema = OrderedDict([
            ("init_loc", (state_dim,)),
            ("init_scale", (state_dim,)),
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
        self.nn[0].bias.data.fill_(0)
        self.nn[2].bias.data.fill_(0)

    def _dynamics(self, features):
        """
        Compute dynamics parameters from time features.
        """
        pyro.module("model_nn", self.nn)
        params = unpack_params(self.nn(features), self.schema)

        init_loc = params["init_loc"][0]
        init_scale_tril = bounded_exp(params["init_scale"][0]).diag_embed()
        init_dist = dist.MultivariateNormal(init_loc, scale_tril=init_scale_tril)
        assert init_dist.batch_shape == ()
        state_dim = init_loc.size(-1)

        trans_matrix = pyro.param("trans_matrix", 0.9 * torch.eye(state_dim))
        trans_loc = torch.zeros(state_dim)
        trans_scale_tril = pyro.param("trans_scale_tril", 0.1 * torch.eye(state_dim),
                                      constraint=constraints.lower_cholesky)
        trans_dist = dist.MultivariateNormal(trans_loc, scale_tril=trans_scale_tril)

        obs_matrix = params["obs_matrix"]
        obs_loc = params["obs_loc"]
        obs_scale = bounded_exp(params["obs_scale"])
        obs_dist = dist.Normal(obs_loc, obs_scale).to_event(1)

        if logging.Logger(None).isEnabledFor(logging.DEBUG):
            logging.debug("trans matrix rms (on, off diag) = {:0.5g}, {:0.5g}"
                          .format(rms(trans_scale_tril.diagonal(dim1=-2, dim2=-1)),
                                  rms(trans_scale_tril.tril(-1))))
            logging.debug("trans scale_tril rms (on, off diag) = {:0.5g}, {:0.5g}"
                          .format(rms(trans_scale_tril.diagonal(dim1=-2, dim2=-1)),
                                  rms(trans_scale_tril.tril(-1))))
            logging.debug("obs matrix, loc, scale rms = {:0.5g}, {:0.5g}, {:0.5g}"
                          .format(rms(obs_matrix), rms(obs_loc), rms(obs_scale)))

        return init_dist, trans_matrix, trans_dist, obs_matrix, obs_dist

    def _unpack_gate_rate(self, gate_rate):
        gate_rate = gate_rate.reshape(
            gate_rate.shape[:-1] + (self.num_stations, self.num_stations, 2))
        gate = gate_rate[..., 0].sigmoid()
        rate = bounded_exp(gate_rate[..., 1])
        return gate, rate

    def __call__(self, features, trip_counts):
        total_hours = len(features)
        observed_hours, num_origins, num_destins = trip_counts.shape
        assert observed_hours <= total_hours
        assert num_origins == self.num_stations
        assert num_destins == self.num_stations
        time_plate = pyro.plate("time", observed_hours, dim=-3)
        origins_plate = pyro.plate("origins", num_origins, dim=-2)
        destins_plate = pyro.plate("destins", num_destins, dim=-1)

        # The first half of the model performs exact inference over
        # the observed portion of the time series.
        hmm = dist.GaussianHMM(*self._dynamics(features[:observed_hours]))
        gate_rate = pyro.sample("gate_rate", hmm)
        gate, rate = self._unpack_gate_rate(gate_rate)
        logging.debug("gate, rate mean = {:0.5g}, {:0.5g}"
                      .format(gate.mean(), rate.mean()))
        with time_plate, origins_plate, destins_plate:
            pyro.sample("trip_count", dist.ZeroInflatedPoisson(gate, rate),
                        obs=trip_counts)

        # The second half of the model forecasts forward.
        forecast = []
        forecast_hours = total_hours - observed_hours
        if forecast_hours > 0:
            _, trans_matrix, trans_dist, obs_matrix, obs_dist = \
                self._dynamics(features[observed_hours:])
        state = None
        for t in range(forecast_hours):
            if state is None:  # on first step
                state_dist = hmm.filter(trip_counts)
            else:
                loc = vm(state, trans_matrix[..., t, :, :]) + trans_dist.loc[..., t, :]
                scale_tril = trans_dist.scale_tril[..., t, :, :]
                state_dist = dist.MultivariateNormal(loc, scale_tril=scale_tril)
            state = pyro.sample("state_{}".format(t), state_dist)

            loc = vm(state, obs_matrix[..., t, :, :]) + obs_dist.loc[..., t, :]
            scale = obs_dist.scale[..., t, :]
            gate_rate = pyro.sample("gate_rate_{}".format(t),
                                    dist.Normal(loc, scale).to_event(1))
            gate, rate = self._unpack_gate_rate(gate_rate)

            with origins_plate, destins_plate:
                forecast.append(pyro.sample("trip_count_{}".format(t),
                                            dist.ZeroInflatedPoisson(gate, rate)))

        return forecast


class Guide:
    def __init__(self, args, num_stations, feature_dim):
        self.nn = nn.Sequential(
            nn.Linear(feature_dim + num_stations ** 2, args.guide_nn_dim),
            nn.Sigmoid(),
            nn.Linear(args.guide_nn_dim, num_stations ** 2 * 2 * 2))
        self.nn[0].bias.data.fill_(0)
        self.nn[2].bias.data.fill_(0)

    def __call__(self, features, trip_counts):
        observed_hours = len(trip_counts)
        pyro.module("guide_nn", self.nn)
        batch_shape = trip_counts.shape[:-2]
        nn_input = torch.cat([features[:observed_hours],
                              trip_counts.reshape(batch_shape + (-1,)).log1p()], dim=-1)
        loc_scale = self.nn(nn_input)
        split = int(loc_scale.size(-1)) // 2
        loc = loc_scale[..., :split]
        scale = bounded_exp(loc_scale[..., split:])
        pyro.sample("gate_rate", dist.Normal(loc, scale).to_event(2))


@torch.no_grad()
def forecast(model, guide, *args, **kwargs):
    with poutine.trace() as tr:
        guide(*args, **kwargs)
    with poutine.replay(trace=tr.trace):
        return model(*args, **kwargs)


def make_minibatch(rows, begin_time, end_time, stations):
    time = rows[:, 0]
    rows = rows[(begin_time <= time) & (time < end_time)]
    time, origin, destin, count = rows.t()
    batch = torch.zeros(end_time - begin_time, len(stations), len(stations))
    batch[time - begin_time, origin, destin] = count.float()
    return batch


def train(args, dataset):
    counts = dataset["counts"]
    num_stations = len(dataset["stations"])
    logging.info("Training on {} stations over {} hours, {} batches/epoch"
                 .format(num_stations, len(counts),
                         int(math.ceil(len(counts) / args.batch_size))))
    time_features = make_time_features(args, 0, len(counts))
    control_features = counts.max(1)[0].clamp(max=1)
    logging.info("On average {:0.1f}/{} stations are open at any one time"
                 .format(control_features.sum(-1).mean(), num_stations))
    features = torch.cat([time_features, control_features], -1)
    feature_dim = features.size(-1)
    logging.info("feature_dim = {}".format(feature_dim))
    metadata = {"args": args, "losses": [], "control": control_features}
    torch.save(metadata, args.training_filename)

    model = Model(args, num_stations, feature_dim)
    guide = Guide(args, num_stations, feature_dim)
    elbo = Trace_ELBO()
    optim = ClippedAdam({"lr": args.learning_rate})
    svi = SVI(model, guide, optim, elbo)
    losses = []
    for step in range(args.num_steps):
        begin_time = torch.randint(max(1, len(counts) - args.batch_size), ()).item()
        end_time = min(len(counts), begin_time + args.batch_size)
        feature_batch = features[begin_time: end_time]
        counts_batch = counts[begin_time: end_time]
        loss = svi.step(feature_batch, counts_batch) / counts_batch.numel()
        losses.append(loss)
        logging.debug("step {} loss = {:0.4g}".format(step, loss))

        if step % 100 == 0:
            pyro.get_param_store().save(args.param_store_filename)
            metadata = {"args": args, "losses": losses, "control": control_features}
            torch.save(metadata, args.training_filename)
    return losses


def main(args):
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.seed)

    dataset = load_hourly_od(args)
    if args.truncate_hours:
        dataset["counts"] = dataset["counts"][:args.truncate_hours]
    train(args, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BART origin-destination forecast")
    parser.add_argument("--param-store-filename", default="pyro_param_store.pkl")
    parser.add_argument("--training-filename", default="training.pkl")
    parser.add_argument("--truncate-hours", default="0", type=int,
                        help="optionally truncate to a subset of hours")
    parser.add_argument("--state-dim", default="8", type=int,
                        help="size of HMM state space in model")
    parser.add_argument("--model-nn-dim", default="64", type=int,
                        help="size of hidden layer in model net")
    parser.add_argument("--guide-nn-dim", default="64", type=int,
                        help="size of hidden layer in guide net")
    parser.add_argument("-n", "--num-steps", default=10001, type=int)
    parser.add_argument("-b", "--batch-size", default=168, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.002, type=float)
    parser.add_argument("--seed", default=123456789, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format='%(relativeCreated) 9d %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    main(args)
