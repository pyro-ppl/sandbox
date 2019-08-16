import argparse
import logging
import math

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


def bounded_exp(x, bound):
    return (x - math.log(bound)).sigmoid() * bound


def make_time_features(args, begin_time, end_time):
    time = torch.arange(begin_time, end_time, dtype=torch.float)
    time_mod_week = time / (24 * 7) % 1. * (2 * math.pi)
    order = 24 * 7 / 2
    angles = time_mod_week.unsqueeze(-1) * torch.arange(1., 1. + order)
    return torch.cat([torch.cos(angles),
                      torch.sin(angles)], dim=-1)


class Model(nn.Module):
    def __init__(self, args, features, trip_counts):
        super().__init__()
        self.args = args
        self.num_stations = trip_counts.size(-1)
        feature_dim = features.size(-1)
        gate_rate_dim = 2 * self.num_stations ** 2
        self.nn = nn.Sequential(
            nn.Linear(feature_dim, args.model_nn_dim),
            nn.Sigmoid(),
            nn.Linear(args.model_nn_dim, 2 * gate_rate_dim))
        self.nn[0].bias.data.fill_(0)
        self.nn[2].bias.data.fill_(0)

    def _dynamics(self, features):
        """
        Compute dynamics parameters from time features.
        """
        state_dim = self.args.state_dim
        gate_rate_dim = 2 * self.num_stations ** 2

        init_loc = torch.zeros(state_dim)
        init_scale_tril = pyro.param("init_scale", torch.full((state_dim,), 10.),
                                     constraint=constraints.positive).diag_embed()
        init_dist = dist.MultivariateNormal(init_loc, scale_tril=init_scale_tril)

        trans_matrix = pyro.param("trans_matrix", 0.99 * torch.eye(state_dim))
        trans_loc = torch.zeros(state_dim)
        trans_scale_tril = pyro.param("trans_scale", 0.1 * torch.ones(state_dim),
                                      constraint=constraints.positive).diag_embed()
        trans_dist = dist.MultivariateNormal(trans_loc, scale_tril=trans_scale_tril)

        obs_matrix = pyro.param("obs_matrix", torch.randn(state_dim, gate_rate_dim))
        obs_matrix.data /= obs_matrix.data.norm(dim=-1, keepdim=True)
        loc_scale = self.nn(features)
        loc, scale = loc_scale.reshape(loc_scale.shape[:-1] + (2, gate_rate_dim)).unbind(-2)
        scale = bounded_exp(scale, bound=10.)
        obs_dist = dist.Normal(loc, scale).to_event(1)

        return init_dist, trans_matrix, trans_dist, obs_matrix, obs_dist

    def _unpack_gate_rate(self, gate_rate):
        n = self.num_stations
        gate, rate = gate_rate.reshape(gate_rate.shape[:-1] + (2, n, n)).unbind(-3)
        gate = gate.sigmoid()
        rate = bounded_exp(rate, bound=1e4)
        return gate, rate

    def forward(self, features, trip_counts):
        pyro.module("model", self)
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


class Guide(nn.Module):
    def __init__(self, args, features, trip_counts):
        super().__init__()
        feature_dim = features.size(-1)
        num_stations = trip_counts.size(-1)
        self.diag_part = nn.Parameter(torch.zeros(2 * 2, 1))
        self.lowrank = nn.Sequential(
            nn.Linear(feature_dim + num_stations ** 2, args.guide_rank),
            nn.Sigmoid(),
            nn.Linear(args.guide_rank, 2 * 2 * num_stations ** 2))
        self.lowrank[0].bias.data.fill_(0)
        self.lowrank[2].bias.data.fill_(0)

    def forward(self, features, trip_counts):
        pyro.module("guide", self)
        assert features.dim() == 2
        assert trip_counts.dim() == 3
        observed_hours = len(trip_counts)
        log_counts = trip_counts.reshape(observed_hours, -1).log1p()
        loc_scale = ((self.diag_part * log_counts.unsqueeze(-2)).reshape(observed_hours, -1) +
                     self.lowrank(torch.cat([features[:observed_hours], log_counts], dim=-1)))
        loc, scale = loc_scale.reshape(observed_hours, 2, -1).unbind(1)
        scale = bounded_exp(scale, bound=10.)
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
    control_features = torch.cat([counts.max(1)[0].clamp(max=1),
                                  counts.max(2)[0].clamp(max=1)], dim=-1)
    logging.info("On average {:0.1f}/{} stations are open at any one time"
                 .format(control_features.sum(-1).mean(), num_stations))
    features = torch.cat([time_features, control_features], -1)
    feature_dim = features.size(-1)
    logging.info("feature_dim = {}".format(feature_dim))
    metadata = {"args": args, "losses": [], "control": control_features}
    torch.save(metadata, args.training_filename)

    def optim_config(module_name, param_name):
        config = {
            "lr": args.learning_rate,
            "betas": (0.8, 0.99),
            "weight_decay": 0.5 ** 1e-2,
        }
        if param_name == "init_scale":
            config["lr"] *= 0.1  # init_dist sees much less data per minibatch
        return config

    model = Model(args, features, counts)
    guide = Guide(args, features, counts)
    elbo = Trace_ELBO()
    optim = ClippedAdam(optim_config)
    svi = SVI(model, guide, optim, elbo)
    losses = []
    for step in range(args.num_steps):
        begin_time = torch.randint(max(1, len(counts) - args.batch_size), ()).item()
        end_time = min(len(counts), begin_time + args.batch_size)
        feature_batch = features[begin_time: end_time]
        counts_batch = counts[begin_time: end_time]
        loss = svi.step(feature_batch, counts_batch) / counts_batch.numel()
        assert math.isfinite(loss), loss
        losses.append(loss)
        logging.debug("step {} loss = {:0.4g}".format(step, loss))

        if step % 20 == 0:
            pyro.get_param_store().save(args.param_store_filename)
            metadata = {"args": args, "losses": losses, "control": control_features}
            torch.save(metadata, args.training_filename)

            if logging.Logger(None).isEnabledFor(logging.DEBUG):
                init_scale = pyro.param("init_scale").data
                trans_scale = pyro.param("trans_scale").data
                trans_matrix = pyro.param("trans_matrix").data
                eigs = trans_matrix.eig()[0].norm(dim=-1).sort(descending=True).values
                logging.debug("guide.diag_part = {}".format(guide.diag_part.data.squeeze()))
                logging.debug("init scale min/mean/max: {:0.3g} {:0.3g} {:0.3g}"
                              .format(init_scale.min(), init_scale.mean(), init_scale.max()))
                logging.debug("trans scale min/mean/max: {:0.3g} {:0.3g} {:0.3g}"
                              .format(trans_scale.min(), trans_scale.mean(), trans_scale.max()))
                logging.debug("trans mat eig:\n{}".format(eigs))

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
    parser.add_argument("--guide-rank", default="8", type=int,
                        help="size of hidden layer in guide net")
    parser.add_argument("-n", "--num-steps", default=501, type=int)
    parser.add_argument("-b", "--batch-size", default=24 * 7 * 2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("--seed", default=123456789, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format='%(relativeCreated) 9d %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    try:
        main(args)
    except Exception as e:
        print(e)
        import pdb
        pdb.post_mortem(e.__traceback__)
