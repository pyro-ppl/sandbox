import logging
import math

import torch
import torch.nn as nn
from pyro.generic import distributions as dist
from pyro.generic import infer, optim, pyro, pyro_backend
from torch.distributions import constraints

import funsor
import funsor.ops as ops
from funsor.domains import reals
from funsor.interpeter import interpretation
from funsor.montecarlo import monte_carlo
from funsor.pyro import dist_to_funsor, matrix_and_mvn_to_funsor, tensor_to_funsor
from funsor.sum_product import sequential_sum_product


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
    """
    The main generative model.
    This is used for both training and forecasting.
    """
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

    def _unpack_gate_rate(self, gate_rate, event_dim):
        """
        Unpack the ``gate_rate`` pair output from the neural net.
        This can be seen as a final layer of the neural net.
        """
        n = self.num_stations
        sample_shape = gate_rate.shape[:-3 - event_dim]
        time_shape = gate_rate.shape[-event_dim:-1]
        if not time_shape:
            time_shape = (1,)
        gate, rate = gate_rate.reshape(sample_shape + time_shape + (2, n, n)).unbind(-3)
        rate = bounded_exp(rate, bound=1e4)
        gate = gate.sigmoid()
        return gate, rate

    def _dynamics(self, features):
        """
        Compute dynamics parameters from time features.
        """
        device = features.device
        state_dim = self.args.state_dim
        gate_rate_dim = 2 * self.num_stations ** 2

        init_loc = torch.zeros(state_dim, device=device)
        init_scale_tril = pyro.param("init_scale",
                                     torch.full((state_dim,), 10., device=device),
                                     constraint=constraints.positive).diag_embed()
        init_dist = dist.MultivariateNormal(init_loc, scale_tril=init_scale_tril)

        trans_matrix = pyro.param("trans_matrix",
                                  0.99 * torch.eye(state_dim, device=device))
        trans_loc = torch.zeros(state_dim, device=device)
        trans_scale_tril = pyro.param("trans_scale",
                                      0.1 * torch.ones(state_dim, device=device),
                                      constraint=constraints.positive).diag_embed()
        trans_dist = dist.MultivariateNormal(trans_loc, scale_tril=trans_scale_tril)

        obs_matrix = pyro.param("obs_matrix", torch.randn(state_dim, gate_rate_dim, device=device))
        obs_matrix.data /= obs_matrix.data.norm(dim=-1, keepdim=True)
        loc_scale = self.nn(features)
        loc, scale = loc_scale.reshape(loc_scale.shape[:-1] + (2, gate_rate_dim)).unbind(-2)
        scale = bounded_exp(scale, bound=10.)
        obs_dist = dist.Normal(loc, scale).to_event(1)

        return init_dist, trans_matrix, trans_dist, obs_matrix, obs_dist

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
        gate, rate = self._unpack_gate_rate(gate_rate, event_dim=2)
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
                state_dist = hmm.filter(gate_rate)
            else:
                loc = vm(state, trans_matrix) + trans_dist.loc
                scale_tril = trans_dist.scale_tril
                state_dist = dist.MultivariateNormal(loc, scale_tril=scale_tril)
            state = pyro.sample("state_{}".format(t), state_dist)

            loc = vm(state, obs_matrix) + obs_dist.base_dist.loc[..., t, :]
            scale = obs_dist.base_dist.scale[..., t, :]
            gate_rate = pyro.sample("gate_rate_{}".format(t),
                                    dist.Normal(loc, scale).to_event(1))
            gate, rate = self._unpack_gate_rate(gate_rate, event_dim=1)

            with origins_plate, destins_plate:
                forecast.append(pyro.sample("trip_count_{}".format(t),
                                            dist.ZeroInflatedPoisson(gate, rate)))
        return forecast

    def log_prob(self, features, trip_counts):
        """
        Compute a non-normalized funsor distribution over gate_rate.
        """
        pyro.module("model", self)
        total_hours = len(features)
        observed_hours, num_origins, num_destins = trip_counts.shape
        assert observed_hours <= total_hours
        assert num_origins == self.num_stations
        assert num_destins == self.num_stations

        @funsor.torch.function(reals(2 * num_origins * num_destins),
                               (reals(num_origins, num_destins),
                                reals(num_origins, num_destins)))
        def unpack_gate_rate(gate_rate):
            batch_shape = gate_rate.shape[:-1]
            event_shape = (2, num_origins, num_destins)
            gate, rate = gate_rate.reshape(batch_shape + event_shape).unbind(-3)
            rate = bounded_exp(rate, bound=1e4)
            gate = gate.sigmoid()
            gate = torch.stack((torch.zeros_like(gate), gate), dim=-1)
            return gate, rate

        # Create a Gaussian latent dynamical system.
        init_dist, trans_matrix, trans_dist, obs_matrix, obs_dist = \
            self._dynamics(features[:observed_hours])
        init = dist_to_funsor(init_dist)(value="state")
        trans = matrix_and_mvn_to_funsor(trans_matrix, trans_dist,
                                         ("time",), "state", "state(time=1)")
        obs = matrix_and_mvn_to_funsor(obs_matrix, obs_dist,
                                       ("time",), "state(time=1)", "value")

        # Compute dynamic prior over gate_rate.
        prior = trans + obs(value="gate_rate")
        prior = sequential_sum_product(ops.logaddexp, ops.add,
                                       prior, "time", {"state": "state(time=1)"})
        prior += init
        prior = prior.reduce(ops.logaddexp, frozenset(["state", "state(time=1)"]))

        # Compute zero-inflated Poisson likelihood.
        gate, rate = unpack_gate_rate("gate_rate")
        likelihood = dist.Categorical(gate["origin", "destin"], value="gated")
        likelihood += funsor.Stack("gated", (
            dist.Poisson(rate["origin", "destin"], value=trip_counts),
            dist.Delta(0, value=trip_counts)))
        likelihood = likelihood.reduce(ops.logaddexp, "gated")
        likelihood = likelihood.reduce(ops.add, frozenset(["time", "origin", "destin"]))

        return prior, likelihood


class Guide(nn.Module):
    """
    The guide, aka encoder part of a variational autoencoder.
    This operates independently over time.
    """
    def __init__(self, args, features, trip_counts):
        super().__init__()
        feature_dim = features.size(-1)
        num_stations = trip_counts.size(-1)

        # Gate and rate are each sampled from diagonal normal distributions
        # whose parameters are estimated from the current counts using
        # a diagonal + low-rank approximation.
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

    def log_prob(self, features, trip_counts):
        assert features.dim() == 2
        assert trip_counts.dim() == 3
        observed_hours = len(trip_counts)
        log_counts = trip_counts.reshape(observed_hours, -1).log1p()
        loc_scale = ((self.diag_part * log_counts.unsqueeze(-2)).reshape(observed_hours, -1) +
                     self.lowrank(torch.cat([features[:observed_hours], log_counts], dim=-1)))
        loc, scale = loc_scale.reshape(observed_hours, 2, -1).unbind(1)
        scale = bounded_exp(scale, bound=10.)
        loc = tensor_to_funsor(loc, ("time",), 1)
        scale = tensor_to_funsor(scale, ("time",), 1)
        log_prob = funsor.Independent(dist.Normal(loc["i"], scale["i"], value="gate_rate"),
                                      "gate_rate", "i")
        return log_prob


def loss_function(model, guide, features, trip_counts):
    p_prior, p_likelihood = model(features, trip_counts)
    q = guide(features, trip_counts)

    # We can compute the KL part analytically.
    exact_part = funsor.Integrate(q, p_prior - q, frozenset(["gate_rate"]))

    # But we need to Monte Carlo approximate to compute likelihood.
    with interpretation(monte_carlo):
        approx_part = funsor.Integrate(q, p_likelihood, frozenset(["gate_rate"]))

    elbo = exact_part + approx_part
    loss = -elbo
    return loss


def train(args, dataset):
    """
    Train a model and guide to fit a dataset.
    """
    counts = dataset["counts"]
    num_stations = len(dataset["stations"])
    logging.info("Training on {} stations over {} hours, {} batches/epoch"
                 .format(num_stations, len(counts),
                         int(math.ceil(len(counts) / args.batch_size))))
    time_features = make_time_features(args, 0, len(counts))
    control_features = (counts.max(1)[0] + counts.max(2)[0]).clamp(max=1)
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
            "weight_decay": 0.01 ** (1 / args.num_steps),
        }
        if param_name == "init_scale":
            config["lr"] *= 0.1  # init_dist sees much less data per minibatch
        return config

    with pyro_backend("funsor"):
        training_counts = counts[:args.truncate] if args.truncate else counts
        data_size = len(training_counts)
        model = Model(args, features, training_counts).to(device=args.device)
        guide = Guide(args, features, training_counts).to(device=args.device)
        elbo = infer.Trace_ELBO()
        optimizer = optim.ClippedAdam(optim_config)
        svi = infer.SVI(model, guide, optimizer, elbo)
        losses = []
        for step in range(args.num_steps):
            begin_time = torch.randint(max(1, data_size - args.batch_size), ()).item()
            end_time = min(data_size, begin_time + args.batch_size)
            feature_batch = features[begin_time: end_time].to(device=args.device)
            counts_batch = counts[begin_time: end_time].to(device=args.device)
            loss = svi.step(feature_batch, counts_batch) / counts_batch.numel()
            assert math.isfinite(loss), loss
            losses.append(loss)
            logging.debug("step {} loss = {:0.4g}".format(step, loss))

            if step % 20 == 0:
                # Save state every few steps.
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
