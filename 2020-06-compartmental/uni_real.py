# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import logging
import os
import pickle
import resource
import sys
import urllib.request
from collections import OrderedDict
from timeit import default_timer

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import pad

import pyro
import pyro.distributions as dist
from pyro.contrib.epidemiology import CompartmentalModel, binomial_dist, infection_dist
from pyro.contrib.forecast.evaluate import eval_crps, eval_mae, eval_rmse
from pyro.infer.mcmc.util import summary
from pyro.ops.tensor_utils import convolve
from util import DATA, get_filename

fmt = '%(process)d %(message)s'
logging.getLogger("pyro").handlers[0].setFormatter(logging.Formatter(fmt))
logging.basicConfig(format=fmt, level=logging.INFO)


# Misc California county populations as of early 2020.
counties = OrderedDict([
    # Bay Area counties.
    ("Santa Clara", 1.763e6),
    ("Alameda", 1.495e6),
    ("Contra Costa", 1.038e6),
    ("San Francisco", 871e3),
    ("San Mateo", 712e3),
    ("Sonoma", 479e3),
    ("Solano", 412e3),
    ("Marin", 251e3),
    ("Napa", 135e3),
    # Misc non Bay Area counties.
    ("Los Angeles", 10.04e6),
    ("Riverside", 2.471e6),
    ("San Diego", 3.338e6),
    ("Orange", 3.176e6),
    ("San Bernardino", 2.18e6),
    ("Imperial", 181e3),
    ("Kern", 900e3),
    ("Fresno", 999e3),
    ("Tulare", 466e3),
    ("Santa Barbara", 446e3),
])
counties = OrderedDict((k, int(v)) for k, v in counties.items())


def load_df(basename):
    url = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
           "csse_covid_19_data/csse_covid_19_time_series/")
    local_path = os.path.join(DATA, basename)
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url + basename, local_path)
    return pd.read_csv(local_path)


def load_data(args):
    cum_cases_df = load_df("time_series_covid19_confirmed_US.csv")
    cum_deaths_df = load_df("time_series_covid19_deaths_US.csv")

    # Convert to torch.Tensor.
    county = list(counties)[args.county]
    population = counties[county]
    i = list(cum_cases_df["Admin2"]).index(county)
    cum_cases = cum_cases_df.iloc[i, 11:]
    i = list(cum_deaths_df["Admin2"]).index(county)
    cum_deaths = cum_deaths_df.iloc[i, 12:]
    cum_cases = torch.tensor(cum_cases, dtype=torch.get_default_dtype()).contiguous()
    cum_deaths = torch.tensor(cum_deaths, dtype=torch.get_default_dtype()).contiguous()
    assert cum_cases.shape == cum_deaths.shape
    start_date = datetime.datetime.strptime(cum_cases_df.columns[11], "%m/%d/%y")

    # Distribute reported cases and deaths among previous few days.
    if args.report_lag:
        kernel = torch.ones(args.report_lag) / args.report_lag
        cum_cases = convolve(cum_cases, kernel, mode="valid").round()
        cum_deaths = convolve(cum_deaths, kernel, mode="valid").round()

    # Convert from cumulative to difference data, and clamp to ensure positivity.
    new_cases = (cum_cases - pad(cum_cases[:-1], (1, 0), value=0)).clamp(min=0)
    new_deaths = (cum_deaths - pad(cum_deaths[:-1], (1, 0), value=0)).clamp(min=0)

    # Truncate.
    truncate = (args.start_date - start_date).days
    assert truncate > 0, "start date is too early"
    new_cases = new_cases[truncate:].contiguous()
    new_deaths = new_deaths[truncate:].contiguous()
    start_date += datetime.timedelta(days=truncate)
    logging.info(f"{county} data shape = {tuple(new_cases.shape)}")

    return {"population": population,
            "new_cases": new_cases,
            "new_deaths": new_deaths,
            "start_date": start_date}


class Model(CompartmentalModel):
    def __init__(self, args, population, new_cases, new_deaths):
        assert new_cases.dim() == 1
        assert new_cases.shape == new_deaths.shape
        duration = len(new_cases)
        compartments = ("S", "E", "I")  # R is implicit.
        super().__init__(compartments, duration, population)

        self.incubation_time = args.incubation_time
        self.recovery_time = args.recovery_time
        self.new_cases = new_cases
        self.new_deaths = new_deaths

        # Intervene via a step function.
        t1 = (args.intervene_date - args.start_date).days
        t2 = self.duration + args.forecast
        self.intervene = torch.cat([torch.zeros(t1), torch.ones(t2 - t1)])

    def global_model(self):
        tau_e = self.incubation_time
        tau_i = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(1., 0.5))  # Weak prior.
        external_rate = pyro.sample("external_rate", dist.LogNormal(-2, 2))
        rho = pyro.sample("rho", dist.Beta(10, 10))  # About 50% response rate.
        mu = pyro.sample("mu", dist.Beta(2, 100))  # About 2% mortality rate.
        drift = pyro.sample("drift", dist.LogNormal(-3, 1.))
        od = pyro.sample("od", dist.Beta(1, 3))
        return R0, external_rate, tau_e, tau_i, rho, mu, drift, od

    def initialize(self, params):
        # Start with no local infections and close to basic reproductive number.
        return {"S": self.population, "E": 0, "I": 0,
                "R_factor": torch.tensor(0.98)}

    def transition(self, params, state, t):
        R0, external_rate, tau_e, tau_i, rho, mu, drift, od = params

        # Assume drift is 4x larger after various interventions begin.
        drift = drift * (0.25 + 0.75 * self.intervene[t])

        # Assume effective reproductive number Rt varies in time.
        sigmoid = torch.distributions.transforms.SigmoidTransform()
        R_factor = pyro.sample("R_factor_{}".format(t),
                               dist.TransformedDistribution(
                                   dist.Normal(sigmoid.inv(state["R_factor"]), drift),
                                   sigmoid))
        Rt = pyro.deterministic("Rt_{}".format(t), R0 * R_factor, event_dim=0)
        I_external = external_rate * tau_i / Rt

        # Sample flows between compartments.
        S2E = pyro.sample("S2E_{}".format(t),
                          infection_dist(individual_rate=Rt / tau_i,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"] + I_external,
                                         population=self.population,
                                         overdispersion=od))
        E2I = pyro.sample("E2I_{}".format(t),
                          binomial_dist(state["E"], 1 / tau_e, overdispersion=od))
        I2R = pyro.sample("I2R_{}".format(t),
                          binomial_dist(state["I"], 1 / tau_i, overdispersion=od))

        # Update compartments and heterogeneous variables.
        state["S"] = state["S"] - S2E
        state["E"] = state["E"] + S2E - E2I
        state["I"] = state["I"] + E2I - I2R
        state["R_factor"] = R_factor

        # Condition on observations.
        t_is_observed = isinstance(t, slice) or t < self.duration
        pyro.sample("new_cases_{}".format(t),
                    binomial_dist(S2E, rho, overdispersion=od),
                    obs=self.new_cases[t] if t_is_observed else None)
        pyro.sample("new_deaths_{}".format(t),
                    binomial_dist(I2R, mu, overdispersion=od),
                    obs=self.new_deaths[t] if t_is_observed else None)

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
                          num_quant_bins=args.num_bins,
                          haar=True,
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
                           guide_rank=args.guide_rank,
                           init_scale=args.init_scale,
                           learning_rate=args.learning_rate,
                           learning_rate_decay=args.learning_rate_decay,
                           betas=args.betas,
                           jit=args.jit)

    return {"loss_initial": losses[0], "loss_final": losses[-1]}


def predict(args, model, truth):
    samples = model.predict(forecast=args.forecast)

    if args.plot:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        shelter_in_place = datetime.datetime.strptime("3/16/20", "%m/%d/%y")
        axes[-1].text(shelter_in_place + datetime.timedelta(days=1), 0.2,
                      "shelter in place")
        for ax in axes:
            ax.axvline(shelter_in_place, color="black", linestyle=":", lw=1, alpha=0.3)
            ax.axvline(truth["start_date"] + datetime.timedelta(days=model.duration),
                       color="black", lw=1, alpha=0.3)
        axes[0].set_title("{}, population {}".format(
            list(counties)[args.county], truth["population"]))
        time = np.array([truth["start_date"] + datetime.timedelta(days=t)
                         for t in range(model.duration + args.forecast)])

        # Plot forecasted series.
        num_samples = samples["R0"].size(0)
        for name, ax in zip(["new_cases", "new_deaths"], axes):
            pred = samples[name][..., model.duration:]
            median = pred.median(dim=0).values
            p05 = pred.kthvalue(int(round(0.5 + 0.05 * num_samples)), dim=0).values
            p95 = pred.kthvalue(int(round(0.5 + 0.95 * num_samples)), dim=0).values
            ax.fill_between(time[model.duration:], p05, p95, color="red", alpha=0.3,
                            label="90% CI")
            ax.plot(time[model.duration:], median, "r-", label="median")
            ax.plot(time, truth[name], "k--", label="truth")
            ax.set_yscale("log")
            ax.set_ylim(0.5, None)
            ax.set_ylabel(f"{name.replace('_', ' ')} / day")
            ax.legend(loc="upper left")

        # Plot the latent time series.
        ax = axes[2]
        for name, color in zip(["E", "I"], ["red", "blue"]):
            value = samples[name]
            median = value.median(dim=0).values
            p05 = value.kthvalue(int(round(0.5 + 0.05 * num_samples)), dim=0).values
            p95 = value.kthvalue(int(round(0.5 + 0.95 * num_samples)), dim=0).values
            ax.fill_between(time, p05, p95, color=color, alpha=0.3)
            ax.plot(time, median, color=color, label=name)
        ax.set_yscale("log")
        ax.set_ylim(0.5, None)
        ax.set_ylabel("# people")
        ax.legend(loc="best")

        # Plot Rt time series.
        Rt = samples["Rt"]
        median = Rt.median(dim=0).values
        p05 = Rt.kthvalue(int(round(0.5 + 0.05 * num_samples)), dim=0).values
        p95 = Rt.kthvalue(int(round(0.5 + 0.95 * num_samples)), dim=0).values
        ax = axes[3]
        ax.fill_between(time, p05, p95, color="red", alpha=0.3, label="90% CI")
        ax.plot(time, median, "r-", label="median")
        ax.axhline(1, color="black", linestyle=":", lw=1, alpha=0.3)
        ax.set_ylim(0, None)
        ax.set_ylabel("Rt")
        ax.legend(loc="best")

        ax.set_xlim(time[0], time[-1])
        locator = mdates.AutoDateLocator(minticks=5, maxticks=15)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)

    return samples


def evaluate(args, truth, model, samples):
    metrics = [("mae", eval_mae), ("rmse", eval_rmse), ("crps", eval_crps)]
    result = {}
    for key, pred in samples.items():
        if key in ("new_cases", "new_deaths"):
            pred = pred[..., model.duration:]

        result[key] = {}
        result[key]["mean"] = pred.mean().item()
        result[key]["std"] = pred.std(dim=0).mean().item()

        if key in truth:
            true = truth[key][..., model.duration:]
            for metric, fn in metrics:
                result[key][metric] = fn(pred, true)

    # Print estimated values.
    covariates = [(name, value.squeeze())
                  for name, value in sorted(samples.items())
                  if value[0].numel() == 1]
    for name, value in covariates:
        mean = value.mean().item()
        std = value.std().item()
        logging.info(f"{name} = {mean:0.3g} \u00B1 {std:0.3g}")

    if args.plot and args.infer == "mcmc":
        # Plot pairwise joint distributions for selected variables.
        import matplotlib.pyplot as plt
        N = len(covariates)
        fig, axes = plt.subplots(N, N, figsize=(8, 8), sharex="col", sharey="row")
        for i in range(N):
            axes[i][0].set_ylabel(covariates[i][0])
            axes[0][i].set_xlabel(covariates[i][0])
            axes[0][i].xaxis.set_label_position("top")
            for j in range(N):
                ax = axes[i][j]
                ax.set_xticks(())
                ax.set_yticks(())
                ax.scatter(covariates[j][1], -covariates[i][1],
                           lw=0, color="darkblue", alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

    return result


def main(args):
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.rng_seed + 20200619)

    result = {"file": __file__, "args": args, "argv": sys.argv}

    truth = load_data(args)
    result["data"] = {
        "population": truth["population"],
        "total_cases": truth["new_cases"].sum().item(),
        "total_deaths": truth["new_deaths"].sum().item(),
        "max_cases": truth["new_cases"].max().item(),
        "max_deaths": truth["new_deaths"].max().item(),
    }

    t0 = default_timer()

    model = Model(args, truth["population"],
                  truth["new_cases"][:-args.forecast],
                  truth["new_deaths"][:-args.forecast])
    infer = {"mcmc": infer_mcmc, "svi": infer_svi}[args.infer]
    result["infer"] = infer(args, model)

    t1 = default_timer()

    samples = predict(args, model, truth)

    t2 = default_timer()

    result["evaluate"] = evaluate(args, truth, model, samples)
    result["times"] = {"infer": t1 - t0, "predict": t2 - t1}
    result["rusage"] = resource.getrusage(resource.RUSAGE_SELF)
    logging.info("DONE")
    return result


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description="CompartmentalModel experiments")
        self.add_argument("--county", default=0, type=int,
                          help="which SF Bay Area county, 0-8")
        self.add_argument("--start-date", default="2/1/20")
        self.add_argument("--intervene-date", default="3/1/20")
        self.add_argument("--report-lag", type=int, default=5)
        self.add_argument("--forecast", default=14, type=int)
        self.add_argument("--recovery-time", default=14.0, type=float)
        self.add_argument("--incubation-time", default=5.5, type=float)
        self.add_argument("--infer", default="svi")
        self.add_argument("--mcmc", action="store_const", const="mcmc", dest="infer")
        self.add_argument("--svi", action="store_const", const="svi", dest="infer")
        self.add_argument("--haar-full-mass", default=10, type=int)
        self.add_argument("--guide-rank", type=int)
        self.add_argument("--init-scale", default=0.01, type=float)
        self.add_argument("--num-samples", default=200, type=int)
        self.add_argument("--smc-particles", default=1024, type=int)
        self.add_argument("--svi-steps", default=5000, type=int)
        self.add_argument("--svi-particles", default=32, type=int)
        self.add_argument("--learning-rate", default=0.1, type=float)
        self.add_argument("--learning-rate-decay", default=0.01, type=float)
        self.add_argument("--betas", default="0.8,0.99")
        self.add_argument("--warmup-steps", type=int)
        self.add_argument("--num-chains", default=2, type=int)
        self.add_argument("--max-tree-depth", default=5, type=int)
        self.add_argument("--rng-seed", default=0, type=int)
        self.add_argument("--num-bins", default=1, type=int)
        self.add_argument("--double", action="store_true", default=True)
        self.add_argument("--single", action="store_false", dest="double")
        self.add_argument("--cuda", action="store_true")
        self.add_argument("--jit", action="store_true", default=True)
        self.add_argument("--nojit", action="store_false", dest="jit")
        self.add_argument("--plot", action="store_true")

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)

        assert args.forecast > 0

        args.betas = tuple(map(float, args.betas.split(",")))

        # Parse dates.
        for name, value in args.__dict__.items():
            if name.endswith("_date"):
                value = datetime.datetime.strptime(value, "%m/%d/%y")
                setattr(args, name, value)

        if args.warmup_steps is None:
            args.warmup_steps = args.num_samples

        if args.double:
            if args.cuda:
                torch.set_default_tensor_type(torch.cuda.DoubleTensor)
            else:
                torch.set_default_dtype(torch.float64)
        elif args.cuda:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        return args


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.1')
    args = Parser().parse_args()

    args.plot = True  # DEBUG
    if args.plot:
        main(args)
        import matplotlib.pyplot as plt
        plt.show()
    else:
        # Cache output.
        outfile = get_filename(__file__, args)
        if not os.path.exists(outfile):
            result = main(args)
            with open(outfile, "wb") as f:
                pickle.dump(result, f)
            logging.info("Saved {}".format(outfile))
