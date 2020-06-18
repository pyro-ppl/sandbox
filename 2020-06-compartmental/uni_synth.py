# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import math
import resource
import pickle
from timeit import default_timer

import torch

import pyro
from pyro.contrib.epidemiology.models import (HeterogeneousSIRModel,
                                              OverdispersedSEIRModel,
                                              OverdispersedSIRModel,
                                              SimpleSEIRModel, SimpleSIRModel,
                                              SuperspreadingSEIRModel,
                                              SuperspreadingSIRModel)
from pyro.contrib.forecast.evaluate import eval_crps, eval_mae, eval_rmse
from pyro.infer.mcmc.util import summary

fmt = '%(process)d %(message)s'
logging.getLogger("pyro").handlers[0].setFormatter(logging.Formatter(fmt))
logging.basicConfig(format=fmt, level=logging.INFO)


def Model(args, data):
    """Dispatch between different model classes."""
    if args.heterogeneous:
        assert args.incubation_time == 0
        assert args.overdispersion == 0
        return HeterogeneousSIRModel(args.population, args.recovery_time, data)
    elif args.incubation_time > 0:
        assert args.incubation_time > 1
        if args.concentration < math.inf:
            return SuperspreadingSEIRModel(args.population, args.incubation_time,
                                           args.recovery_time, data)
        elif args.overdispersion > 0:
            return OverdispersedSEIRModel(args.population, args.incubation_time,
                                          args.recovery_time, data)
        else:
            return SimpleSEIRModel(args.population, args.incubation_time,
                                   args.recovery_time, data)
    else:
        if args.concentration < math.inf:
            return SuperspreadingSIRModel(args.population, args.recovery_time, data)
        elif args.overdispersion > 0:
            return OverdispersedSIRModel(args.population, args.recovery_time, data)
        else:
            return SimpleSIRModel(args.population, args.recovery_time, data)


def generate_data(args):
    extended_data = [None] * (args.duration + args.forecast)
    model = Model(args, extended_data)
    logging.info("Simulating from a {}".format(type(model).__name__))
    for attempt in range(100):
        truth = model.generate({"R0": args.R0,
                                "rho": args.response_rate,
                                "k": args.concentration,
                                "od": args.overdispersion})
        obs = truth["obs"][:args.duration]
        new_I = truth.get("S2I", truth.get("E2I"))

        obs_sum = int(obs.sum())
        new_I_sum = int(new_I[:args.duration].sum())
        assert 0 <= args.min_obs_portion < args.max_obs_portion <= 1
        min_obs = int(math.ceil(args.min_obs_portion * args.population))
        max_obs = int(math.floor(args.max_obs_portion * args.population))
        if min_obs <= obs_sum <= max_obs:
            logging.info("Observed {:d}/{:d} infections:\n{}".format(
                obs_sum, new_I_sum, " ".join(str(int(x)) for x in obs)))
            return truth

    if obs_sum < min_obs:
        raise ValueError("Failed to generate >={} observations. "
                         "Try decreasing --min-obs-portion (currently {})."
                         .format(min_obs, args.min_obs_portion))
    else:
        raise ValueError("Failed to generate <={} observations. "
                         "Try increasing --max-obs-portion (currently {})."
                         .format(max_obs, args.max_obs_portion))


def _item(x):
    if isinstance(x, torch.Tensor):
        x = x.reshape(-1).median().item()
    elif isinstance(x, dict):
        for key, value in x.items():
            x[key] = _item(value)
    return x


def infer_mcmc(args, model):
    parallel = args.num_chains > 1

    mcmc = model.fit_mcmc(heuristic_num_particles=args.smc_particles,
                          warmup_steps=args.warmup_steps,
                          num_samples=args.num_samples,
                          num_chains=args.num_chains,
                          mp_context="spawn" if parallel else None,
                          max_tree_depth=args.max_tree_depth,
                          arrowhead_mass=args.arrowhead_mass,
                          num_quant_bins=args.num_bins,
                          haar=args.haar,
                          haar_full_mass=args.haar_full_mass,
                          jit_compile=args.jit)

    result = summary(mcmc._samples)
    result = _item(result)
    return result


def infer_svi(args, model):
    losses = model.fit_svi(heuristic_num_particles=args.smc_particles,
                           num_samples=args.num_samples,
                           num_steps=args.svi_steps,
                           num_particles=args.svi_particles,
                           haar=args.haar,
                           jit=args.jit)

    return {"loss_initial": losses[0], "loss_final": losses[-1]}


def evaluate(args, truth, model, samples):
    metrics = [("mae", eval_mae), ("rmse", eval_rmse), ("crps", eval_crps)]
    result = {}
    for key, pred in samples.items():
        if key == "obs":
            pred = pred[..., args.duration:]

        result[key] = {}
        result[key]["mean"] = pred.mean().item()
        result[key]["std"] = pred.std(dim=0).mean().item()

        if key in truth:
            true = truth[key]
            if key == "obs":
                true = true[..., args.duration:]
            for metric, fn in metrics:
                result[key][metric] = fn(pred, true)

    return result


def main(args):
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.rng_seed + 20200617)

    result = {}

    truth = generate_data(args)

    t0 = default_timer()

    model = Model(args, data=truth["obs"][:args.duration])
    infer = {"mcmc": infer_mcmc, "svi": infer_svi}[args.infer]
    result["infer"] = infer(args, model)

    t1 = default_timer()

    samples = model.predict(forecast=args.forecast)

    t2 = default_timer()

    result["evaluate"] = evaluate(args, truth, model, samples)
    result["times"] = {"infer": t1 - t0, "predict": t2 - t1}
    result["rusage"] = resource.getrusage(resource.RUSAGE_SELF)

    if args.outfile:
        with open(args.outfile, "wb") as f:
            pickle.dump(result, f)
    return result


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.1')
    parser = argparse.ArgumentParser(description="CompartmentalModel experiments")
    parser.add_argument("--population", default=1000, type=float)
    parser.add_argument("--min-obs-portion", default=0.1, type=float)
    parser.add_argument("--max-obs-portion", default=0.3, type=float)
    parser.add_argument("--duration", default=20, type=int)
    parser.add_argument("--forecast", default=10, type=int)
    parser.add_argument("--R0", default=1.5, type=float)
    parser.add_argument("--recovery-time", default=7.0, type=float)
    parser.add_argument("--incubation-time", default=0.0, type=float)
    parser.add_argument("--concentration", default=math.inf, type=float)
    parser.add_argument("--response-rate", default=0.5, type=float)
    parser.add_argument("--overdispersion", default=0., type=float)
    parser.add_argument("--heterogeneous", action="store_true")
    parser.add_argument("--infer", default="mcmc")
    parser.add_argument("--mcmc", action="store_const", const="mcmc", dest="infer")
    parser.add_argument("--svi", action="store_const", const="svi", dest="infer")
    parser.add_argument("--haar", action="store_true")
    parser.add_argument("--nohaar", action="store_const", const=False, dest="haar")
    parser.add_argument("--haar-full-mass", default=10, type=int)
    parser.add_argument("--num-samples", default=200, type=int)
    parser.add_argument("--smc-particles", default=1024, type=int)
    parser.add_argument("--svi-steps", default=5000, type=int)
    parser.add_argument("--svi-particles", default=32, type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--num-chains", default=2, type=int)
    parser.add_argument("--max-tree-depth", default=5, type=int)
    parser.add_argument("--arrowhead-mass", action="store_true")
    parser.add_argument("--rng-seed", default=0, type=int)
    parser.add_argument("--num-bins", default=1, type=int)
    parser.add_argument("--double", action="store_true", default=True)
    parser.add_argument("--single", action="store_false", dest="double")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--jit", action="store_true", default=True)
    parser.add_argument("--nojit", action="store_false", dest="jit")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--outfile")
    args = parser.parse_args()
    args.population = int(args.population)  # to allow e.g. --population=1e6

    if args.warmup_steps is None:
        args.warmup_steps = args.num_samples
    if args.double:
        if args.cuda:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            torch.set_default_dtype(torch.float64)
    elif args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    main(args)
