import argparse
import os
import pickle

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from pyro.contrib.evaluate import backtest
from pyro.contrib.examples.bart import load_bart_od
from pyro.contrib.forecast import ForecastingModel
from pyro.infer.reparam import LocScaleReparam, StudentTReparam, SymmetricStableReparam
from pyro.ops.tensor_utils import periodic_repeat


class Model2(ForecastingModel):
    def __init__(self, args):
        super().__init__()
        self.dist = args.dist

    def model(self, zero_data, covariates):
        num_stations, num_stations, duration, one = zero_data.shape

        origin_plate = pyro.plate("origin", num_stations, dim=-3)
        destin_plate = pyro.plate("destin", num_stations, dim=-2)
        hour_of_week_plate = pyro.plate("hour_of_week", 24 * 7, dim=-1)

        if self.dist == "stable":
            drift_stability = pyro.sample("drift_stability", dist.Uniform(1, 2))
        elif self.dist == "studentt":
            drift_dof = pyro.sample("drift_dof", dist.Uniform(1, 10))
        drift_scale = pyro.sample("drift_scale", dist.LogNormal(-20, 5))

        with origin_plate:
            with hour_of_week_plate:
                origin_seasonal = pyro.sample("origin_seasonal", dist.Normal(0, 5))
        with destin_plate:
            with hour_of_week_plate:
                destin_seasonal = pyro.sample("destin_seasonal", dist.Normal(0, 5))
            with self.time_plate:
                with poutine.reparam(config={"drift": LocScaleReparam()}):
                    if self.dist == "stable":
                        with poutine.reparam(config={"drift": SymmetricStableReparam()}):
                            drift = pyro.sample("drift", dist.Stable(drift_stability, 0, drift_scale))
                    elif self.dist == "studentt":
                        with poutine.reparam(config={"drift": StudentTReparam()}):
                            drift = pyro.sample("drift", dist.StudentT(drift_dof, 0, drift_scale))
                    else:
                        assert self.dist == "gaussian"
                        drift = pyro.sample("drift", dist.Normal(0, drift_scale))
        with origin_plate, destin_plate:
            pairwise = pyro.sample("pairwise", dist.Normal(0, 1))

        seasonal = origin_seasonal + destin_seasonal  # Note this broadcasts.
        seasonal = periodic_repeat(seasonal, duration, dim=-1)
        motion = drift.cumsum(dim=-1)  # A Levy stable motion to model shocks.
        prediction = motion + seasonal + pairwise

        with origin_plate:
            origin_scale = pyro.sample("origin_scale", dist.LogNormal(-5, 5))
        with destin_plate:
            destin_scale = pyro.sample("destin_scale", dist.LogNormal(-5, 5))
        scale = origin_scale + destin_scale

        scale = scale.unsqueeze(-1)
        prediction = prediction.unsqueeze(-1)
        noise_dist = dist.Normal(0, scale)
        with origin_plate, destin_plate:
            self.predict(noise_dist, prediction)


def main(args):
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.seed)

    dataset = load_bart_od()
    print(dataset.keys())
    print(dataset["counts"].shape)
    print(" ".join(dataset["stations"]))

    data = dataset["counts"].permute(1, 2, 0).unsqueeze(-1).log1p().contiguous()
    print(dataset["counts"].shape, data.shape)
    covariates = torch.zeros(data.size(-2), 0)  # empty

    def create_plates(zero_data, covariates):
        num_origins, num_destins, duration, one = zero_data.shape
        return [pyro.plate("origin", num_origins, subsample_size=args.batch_size, dim=-3),
                pyro.plate("destin", num_destins, subsample_size=args.batch_size, dim=-2)]

    forecaster_options = {
        "create_plates": create_plates,
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "log_every": args.log_every,
    }

    filename = os.path.abspath(__file__)[:3] + ".{}.pkl".format(args.dist)
    if args.force or not os.path.exists(filename):
        windows = backtest(data, covariates, Model2,
                           train_window=args.train_widow,
                           test_window=args.test_window,
                           stride=args.stride,
                           forecaster_options=forecaster_options)
        with open(filename, "wb") as f:
            pickle.dump(f, windows)
    with open(filename, "rb") as f:
        windows = pickle.load(f)

    for name in ["crps", "mae"]:
        values = torch.tensor([w[name] for w in windows])
        print("{} = {:0.3g} +- {:0.2g}".format(name, values.mean().item(), values.std().item()))


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.3.0")
    parser = argparse.ArgumentParser(description="Multivariate BART forecasting")
    parser.add_argument("--dist", default="stable")
    parser.add_argument("--train-window", default=24 * 90, type=int)
    parser.add_argument("--test-window", default=24 * 7, type=int)
    parser.add_argument("--stride", default=30, type=int)
    parser.add_argument("-b", "--batch-size", default=10, type=int)
    parser.add_argument("-n", "--num-steps", default=2000, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--log-every", default=50, type=int)
    parser.add_argument("--seed", default=1234567890, type=int)
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()
    main(args)
