import argparse
import itertools
import math
import multiprocessing
import os
import pickle
import random
from timeit import default_timer

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from pyro.contrib.easyguide import EasyGuide
from pyro.infer import MCMC, NUTS, SVI, EnergyDistance, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoNormal, init_to_sample
from pyro.infer.reparam import StableReparam, SymmetricStableReparam
from pyro.nn import PyroParam
from pyro.optim import Adam, ClippedAdam
from torch.distributions import constraints

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "univariate")
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)


def model(data=None, skew=None, num_samples=None):
    stability = pyro.sample("stability", dist.Uniform(0, 2))
    if skew is None:
        skew = pyro.sample("skew", dist.Uniform(-1, 1))
    scale = pyro.sample("scale", dist.LogNormal(0, 10))
    loc = pyro.sample("loc", dist.Normal(0, scale))
    with pyro.plate("plate", num_samples if data is None else len(data)):
        if data is not None:
            data = pyro.subsample(data, event_dim=0)
        return pyro.sample("x", dist.Stable(stability, skew, scale, loc), obs=data)


class AmortizedGuide(EasyGuide):
    def __init__(self, model):
        super().__init__(model)
        self.stability = PyroParam(torch.tensor(INITS["stability"]), constraints.interval(0., 2.))
        self.skew = PyroParam(torch.tensor(INITS["skew"]), constraints.interval(-1., 1.))
        self.scale = PyroParam(torch.tensor(INITS["scale"]), constraints.positive)
        self.loc = PyroParam(torch.tensor(INITS["loc"]))
        self.nn = None

    def guide(self, data, skew=None):
        pyro.sample("stability", dist.Delta(self.stability))
        if skew is None:
            pyro.sample("skew", dist.Delta(self.skew))
        pyro.sample("scale", dist.Delta(self.scale))
        pyro.sample("loc", dist.Delta(self.loc))

        group = self.group("x_.*")
        if self.nn is None:
            self.nn = nn.Sequential(
                nn.Linear(1, ARGS.hidden_dim),
                nn.Sigmoid(),
                nn.Linear(ARGS.hidden_dim, ARGS.hidden_dim),
                nn.Sigmoid(),
                nn.Linear(ARGS.hidden_dim, 2 * group.event_shape[0]),
            )

        subsample = None
        if ARGS.batch_size < len(data):
            subsample = torch.randint(len(data), (ARGS.batch_size,))
        with self.plate("plate", len(data), subsample=subsample):
            data = pyro.subsample(data, event_dim=0)
            nn_input = (data - self.loc).div(self.scale).unsqueeze(-1)
            loc, scale = self.nn(nn_input).reshape(len(data), 2, group.event_shape[0]).unbind(-2)
            scale = nn.functional.softplus(scale)
            group.sample("x_group", dist.Normal(loc, scale).to_event(1))

    def median(self):
        return {k: getattr(self, k) for k in INITS}


INITS = {"stability": 1.8, "skew": 0.0, "scale": 1.0, "loc": 0.0}


def init_loc_fn(site):
    if site["name"] in INITS:
        return torch.tensor(INITS[site["name"]])
    return init_to_sample(site)


def synthesize(stability, skew, num_samples):
    data = {
        "stability": torch.tensor(float(stability)),
        "skew": torch.tensor(float(skew)),
        "scale": torch.ones(()),
        "loc": torch.zeros(()),
    }
    with poutine.condition(data=data):
        return model(num_samples=num_samples)


METHODS = {}


def method(name):
    def decorator(fn):
        METHODS[name] = fn
        return fn
    return decorator


class RetryError(Exception):
    pass


@method("SVI")
def svi(data, min_stability, skew):
    rep = StableReparam() if skew is None else SymmetricStableReparam()
    reparam_model = poutine.reparam(model, {"x": rep})
    guide = AutoNormal(reparam_model)
    num_steps = 1001
    optim = ClippedAdam({"lr": 0.1, "lrd": 0.1 ** (1 / num_steps), "betas": (0.8, 0.95)})
    svi = SVI(reparam_model, guide, optim, Trace_ELBO())
    for step in range(num_steps):
        loss = svi.step(data, skew)
        if __debug__ and step % ARGS.log_every == 0:
            print("step {} loss = {:0.4g}".format(step, loss / data.numel()))
        if math.isnan(loss):
            raise RetryError
    median = guide.median()
    return {
        "stability": median["stability"].item(),
        "skew": median["skew"].item() if skew is None else 0.,
        "scale": median["scale"].item(),
        "loc": median["loc"].item(),
    }


@method("SSVI")
def ssvi(data, min_stability, skew):
    rep = StableReparam() if skew is None else SymmetricStableReparam()
    reparam_model = poutine.reparam(model, {"x": rep})

    create_plates = None
    if ARGS.batch_size < len(data):
        def create_plates(data, skew=None):
            subsample = torch.randint(len(data), (ARGS.batch_size,))
            return pyro.plate("plate", len(data), subsample=subsample)

    guide = AutoNormal(reparam_model, create_plates=create_plates)
    num_steps = 1 + max(1000, 10 * len(data) // ARGS.batch_size)
    optim = ClippedAdam({"lr": 0.1, "lrd": 0.1 ** (1 / num_steps), "betas": (0.8, 0.95)})
    svi = SVI(reparam_model, guide, optim, Trace_ELBO())
    for step in range(num_steps):
        loss = svi.step(data, skew)
        if __debug__ and step % ARGS.log_every == 0:
            print("step {} loss = {:0.4g}".format(step, loss / data.numel()))
        if math.isnan(loss):
            raise RetryError
    median = guide.median()
    return {
        "stability": median["stability"].item(),
        "skew": median["skew"].item() if skew is None else 0.,
        "scale": median["scale"].item(),
        "loc": median["loc"].item(),
    }


@method("ASVI")
def asvi(data, min_stability, skew):
    rep = StableReparam() if skew is None else SymmetricStableReparam()
    reparam_model = poutine.reparam(model, {"x": rep})
    guide = AmortizedGuide(reparam_model)
    num_steps = 2001
    optim = ClippedAdam({"lr": 0.02, "lrd": 0.1 ** (1 / num_steps), "betas": (0.95, 0.99)})
    svi = SVI(reparam_model, guide, optim, Trace_ELBO())
    for step in range(num_steps):
        loss = svi.step(data, skew)
        if __debug__ and step % ARGS.log_every == 0:
            print("step {} loss = {:0.4g}".format(step, loss / data.numel()))
        if math.isnan(loss):
            raise RetryError
    median = guide.median()
    return {
        "stability": median["stability"].item(),
        "skew": median["skew"].item() if skew is None else 0.,
        "scale": median["scale"].item(),
        "loc": median["loc"].item(),
    }


@method("HMC")
def hmc(data, min_stability, skew):
    rep = StableReparam() if skew is None else SymmetricStableReparam()
    reparam_model = poutine.reparam(model, {"x": rep})
    kernel = NUTS(reparam_model, max_tree_depth=3)
    mcmc = MCMC(kernel, warmup_steps=199, num_samples=1001,
                disable_progbar=(not __debug__))
    mcmc.run(data, skew)
    samples = mcmc.get_samples()
    median = {k: v.median() for k, v in samples.items()}
    return {
        "stability": median["stability"].item(),
        "skew": median["skew"].item() if skew is None else 0.,
        "scale": median["scale"].item(),
        "loc": median["loc"].item(),
    }


# This does not seem to converge.
# @method("Energy")
def energy(data, min_stability, skew):
    guide = AutoDelta(model, init_loc_fn=init_loc_fn)
    num_steps = 1001
    optim = Adam({"lr": 0.1})
    energy = EnergyDistance(beta=min_stability, num_particles=ARGS.num_particles)
    svi = SVI(model, guide, optim, energy)
    for step in range(num_steps):
        loss = svi.step(data, skew)
        if __debug__ and step % 50 == 0:
            print("step {} loss = {:0.4g}".format(step, loss / data.numel()))
    median = guide.median()
    return {
        "stability": median["stability"].item(),
        "skew": median["skew"].item() if skew is None else 0.,
        "scale": median["scale"].item(),
        "loc": median["loc"].item(),
    }


def evaluate(name, stability, skew, num_samples, seed):
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    data = synthesize(stability, skew, num_samples)
    min_stability = min(1.0, stability - 0.1)

    time = -default_timer()
    for retries in range(ARGS.max_retries):
        try:
            pred = METHODS[name](data, min_stability, None if skew else skew)
        except RetryError:
            print("error, retrying")
        else:
            break
    time += default_timer()

    truth = {"stability": stability, "skew": skew, "scale": 1., "loc": 0.}
    error = {k: abs(pred[k] - v) for k, v in truth.items()}

    if __debug__:
        print("time = {:0.3g}".format(time))
        print("True: {}".format(", ".join("{}={:0.3g}".format(k, v) for k, v in truth.items())))
        print("Pred: {}".format(", ".join("{}={:0.3g}".format(k, v) for k, v in pred.items())))
        print("Error: {}".format(", ".join("{}={:0.3g}".format(k, v) for k, v in error.items())))
        assert error["stability"] < 0.5
        assert error["skew"] < 0.5
        assert error["scale"] < 1
        assert error["loc"] < 1

    return {
        "name": name,
        "num_samples": num_samples,
        "seed": seed,
        "truth": truth,
        "pred": pred,
        "error": error,
        "time": time,
        "retries": retries,
    }


def _evaluate(args):
    print(*args)
    filename = os.path.join(RESULTS, "{}_{:0.1f}_{:0.1f}_{}_{}.pkl".format(*args))
    if os.path.exists(filename) and not ARGS.force:
        with open(filename, "rb") as f:
            return pickle.load(f)
    result = evaluate(*args)
    with open(filename, "wb") as f:
        pickle.dump(result, f)
    return result


def main(args):
    pyro.enable_validation(__debug__)
    grid = itertools.product(args.method.split(","),
                             map(float, args.stability.split(",")),
                             map(float, args.skew.split(",")),
                             map(int, args.num_samples.split(",")),
                             range(args.num_seeds))
    grid = list(grid)
    if args.shuffle:
        pyro.set_rng_seed(args.shuffle)
        random.shuffle(grid)
    map_ = map if __debug__ else multiprocessing.Pool().map
    summary = list(map_(_evaluate, grid))
    with open(os.path.join(RESULTS, "summary.pkl"), "wb") as f:
        pickle.dump(summary, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Univariate stable parameter fitting")
    parser.add_argument("--method", default=",".join(sorted(METHODS)))
    parser.add_argument("--hidden-dim", default=16, type=int)
    parser.add_argument("--num-particles", default=8, type=int)
    parser.add_argument("--num-samples", default="100,1000,10000")
    parser.add_argument("--stability", default="0.5,1.0,1.5,1.7,1.9")
    parser.add_argument("--skew", default="0.0,0.1,0.5,0.9")
    parser.add_argument("--batch-size", default=1000, type=int)
    parser.add_argument("--max-retries", default=5, type=int)
    parser.add_argument("--num-seeds", default=50, type=int)
    parser.add_argument("--shuffle", default=0, type=int)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--log-every", default=50, type=int)
    ARGS = parser.parse_args()
    try:
        main(ARGS)
    except Exception as e:
        print(e)
        import pdb
        pdb.post_mortem(e.__traceback__)
