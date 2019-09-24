import argparse
import logging
import math

import torch
import torch.nn as nn
from pyro.generic import distributions as dist
from pyro.generic import infer, optim, pyro
from torch.distributions import constraints

import funsor
import funsor.distributions as fdist
import funsor.ops as ops
from funsor.domains import reals
from funsor.interpreter import dispatched_interpretation, interpretation
from funsor.montecarlo import monte_carlo
from funsor.pyro.convert import (
        AffineNormal, dist_to_funsor, matrix_and_mvn_to_funsor, tensor_to_funsor)
from funsor.sum_product import MarkovProduct
from funsor.terms import Subs, normalize, Binary, Funsor
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


@dispatched_interpretation
def monte_carlo_debug(cls, *args):
    monte_carlo_debug.dispatch(cls, *args)
    return monte_carlo(cls, *args)


@monte_carlo_debug.register(Subs, AffineNormal, tuple)
def _(term, subs):
    print(f"DEBUG\n{term.pretty()}\n{subs}")


@monte_carlo_debug.register(Binary, ops.GetitemOp, Funsor, Funsor)
def _(op, lhs, rhs):
    print(f"DEBUG\n{op}\n{lhs.pretty()}\n{rhs.pretty()}")


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
        """
        Compute a non-normalized funsor distribution over gate_rate.
        """
        pyro.module("model", self)
        total_hours = len(features)
        observed_hours, num_origins, num_destins = trip_counts.shape
        assert observed_hours <= total_hours
        assert num_origins == self.num_stations
        assert num_destins == self.num_stations
        gate_rate = funsor.Variable(
            "gate_rate_t", reals(observed_hours, 2 * num_origins * num_destins))["time"]

        @funsor.torch.function(reals(2 * num_origins * num_destins),
                               (reals(num_origins, num_destins, 2),
                                reals(num_origins, num_destins)))
        def unpack_gate_rate(gate_rate):
            batch_shape = gate_rate.shape[:-1]
            event_shape = (2, num_origins, num_destins)
            gate, rate = gate_rate.reshape(batch_shape + event_shape).unbind(-3)
            rate = bounded_exp(rate, bound=1e4)
            gate = torch.stack((torch.zeros_like(gate), gate), dim=-1)
            return gate, rate

        # Create a Gaussian latent dynamical system.
        init_dist, trans_matrix, trans_dist, obs_matrix, obs_dist = \
            self._dynamics(features[:observed_hours])
        init = dist_to_funsor(init_dist)(value="state")
        trans = matrix_and_mvn_to_funsor(trans_matrix, trans_dist,
                                         ("time",), "state", "state(time=1)")
        obs = matrix_and_mvn_to_funsor(obs_matrix, obs_dist,
                                       ("time",), "state(time=1)", "gate_rate")

        # Compute dynamic prior over gate_rate.
        prior = trans + obs(gate_rate=gate_rate)
        prior = MarkovProduct(ops.logaddexp, ops.add,
                              prior, "time", {"state": "state(time=1)"})
        prior += init
        prior = prior.reduce(ops.logaddexp, {"state", "state(time=1)"})

        # Compute zero-inflated Poisson likelihood.
        gate, rate = unpack_gate_rate(gate_rate)
        likelihood = fdist.Categorical(gate["origin", "destin"], value="gated")
        trip_counts = tensor_to_funsor(trip_counts, ("time", "origin", "destin"))
        likelihood += funsor.Stack("gated", (
            fdist.Poisson(rate["origin", "destin"], value=trip_counts),
            fdist.Delta(0, value=trip_counts)))
        likelihood = likelihood.reduce(ops.logaddexp, "gated")
        likelihood = likelihood.reduce(ops.add, {"time", "origin", "destin"})

        assert set(prior.inputs) == {"gate_rate_t"}, prior.inputs
        assert set(likelihood.inputs) == {"gate_rate_t"}, likelihood.inputs
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
        assert features.dim() == 2
        assert trip_counts.dim() == 3
        observed_hours = len(trip_counts)
        log_counts = trip_counts.reshape(observed_hours, -1).log1p()
        loc_scale = ((self.diag_part * log_counts.unsqueeze(-2)).reshape(observed_hours, -1) +
                     self.lowrank(torch.cat([features[:observed_hours], log_counts], dim=-1)))
        loc, scale = loc_scale.reshape(observed_hours, 2, -1).unbind(1)
        scale = bounded_exp(scale, bound=10.)
        diag_normal = dist.Normal(loc, scale).to_event(1)
        log_prob = dist_to_funsor(diag_normal, ("time",))(value="gate_rate")
        log_prob = funsor.Independent(log_prob, "gate_rate_t", "time", "gate_rate")

        assert set(log_prob.inputs) == {"gate_rate_t"}, log_prob.inputs
        return log_prob


def elbo_loss(model, guide, args, features, trip_counts):
    q = guide(features, trip_counts)
    if args.debug:
        print(f"q = {q.quote()}")
    with interpretation(normalize):
        p_prior, p_likelihood = model(features, trip_counts)
        if args.debug:
            print(f"p_prior = {p_prior.quote()}")
            print(f"p_likelihood = {p_likelihood.quote()}")

    if args.analytic_kl:
        # We can compute the KL part exactly.
        exact_part = funsor.Integrate(q, p_prior - q, frozenset(["gate_rate_t"]))

        # But we need to Monte Carlo approximate to compute likelihood.
        with interpretation(monte_carlo):
            approx_part = funsor.Integrate(q, p_likelihood, frozenset(["gate_rate_t"]))

        elbo = exact_part + approx_part
    else:
        with interpretation(normalize):
            p = p_prior + p_likelihood
            pq = p - q
        # Monte Carlo approximate everything.
        with interpretation(monte_carlo):
            elbo = funsor.Integrate(q, pq, frozenset(["gate_rate_t"]))

    loss = -elbo
    assert not loss.inputs, loss.inputs
    assert isinstance(loss, funsor.Tensor), loss.pretty()
    return loss.data


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

    training_counts = counts[:args.truncate] if args.truncate else counts
    data_size = len(training_counts)
    model = Model(args, features, training_counts).to(device=args.device)
    guide = Guide(args, features, training_counts).to(device=args.device)
    optimizer = optim.ClippedAdam(optim_config)
    svi = infer.SVI(model, guide, optimizer, elbo_loss)
    losses = []
    for step in range(args.num_steps):
        begin_time = torch.randint(max(1, data_size - args.batch_size), ()).item()
        end_time = min(data_size, begin_time + args.batch_size)
        feature_batch = features[begin_time: end_time].to(device=args.device)
        counts_batch = counts[begin_time: end_time].to(device=args.device)
        loss = svi.step(args, feature_batch, counts_batch) / counts_batch.numel()
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


def main(args):
    assert pyro.__version__ >= "0.4.1"
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.seed)
    dataset = load_hourly_od(args)
    if args.tiny:
        dataset["stations"] = dataset["stations"][:args.tiny]
        dataset["counts"] = dataset["counts"][:, :args.tiny, :args.tiny]
    train(args, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BART origin-destination forecast")
    parser.add_argument("--param-store-filename", default="pyro_param_store.pkl")
    parser.add_argument("--training-filename", default="training.pkl")
    parser.add_argument("--truncate", default=0, type=int,
                        help="optionally truncate to a subset of hours")
    parser.add_argument("--tiny", default=0, type=int,
                        help="optionally truncate to a subset of stations")
    parser.add_argument("--state-dim", default=8, type=int,
                        help="size of HMM state space in model")
    parser.add_argument("--model-nn-dim", default=64, type=int,
                        help="size of hidden layer in model net")
    parser.add_argument("--guide-rank", default=8, type=int,
                        help="size of hidden layer in guide net")
    parser.add_argument("--analytic-kl", action="store_true")
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-b", "--batch-size", default=24 * 7 * 2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("--seed", default=123456789, type=int)
    parser.add_argument("--device", default="")
    parser.add_argument("--cuda", dest="device", action="store_const", const="cuda")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.basicConfig(format='%(relativeCreated) 9d %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        main(args)
    except (Exception, KeyboardInterrupt) as e:
        print(e)
        import pdb
        pdb.post_mortem(e.__traceback__)
