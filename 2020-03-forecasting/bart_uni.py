import argparse
import os
import pickle

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from pyro.contrib.examples.bart import load_bart_od
from pyro.contrib.forecast import ForecastingModel, backtest
from pyro.contrib.forecast.evaluate import backtest
from pyro.infer.reparam import LocScaleReparam, StableReparam
from pyro.ops.tensor_utils import periodic_features

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS = os.environ.get("BART_RESULTS", RESULTS)
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)


class Model(ForecastingModel):
    def __init__(self, dist_type):
        super().__init__()
        self.dist_type = dist_type

    def model(self, zero_data, covariates):
        data_dim = zero_data.size(-1)
        feature_dim = covariates.size(-1)

        # Globals.
        bias = pyro.sample("bias", dist.Normal(0, 10).expand([data_dim]).to_event(1))
        weight = pyro.sample("weight", dist.Normal(0, 0.1).expand([feature_dim]).to_event(1))
        trans_scale = pyro.sample("trans_scale", dist.LogNormal(-20, 5).expand([1]).to_event(1))
        obs_scale = pyro.sample("obs_scale", dist.LogNormal(-5, 5).expand([1]).to_event(1))
        if self.dist_type == "normal":
            pass
        elif self.dist_type == "stable":
            trans_stability = pyro.sample("trans_stability", dist.Uniform(1, 2).expand([1]).to_event(1))
            trans_skew = pyro.sample("trans_skew", dist.Uniform(-1, 1).expand([1]).to_event(1))
            obs_stability = pyro.sample("obs_stability", dist.Uniform(1, 2).expand([1]).to_event(1))
            obs_skew = pyro.sample("obs_skew", dist.Uniform(-1, 1).expand([1]).to_event(1))
        elif self.dist_type == "studentt":
            trans_dof = pyro.sample("trans_dof", dist.Uniform(1, 10).expand([1]).to_event(1))
            obs_dof = pyro.sample("obs_dof", dist.Uniform(1, 10).expand([1]).to_event(1))
        else:
            raise ValueError(self.dist_type)

        # Series locals.
        with self.time_plate:
            if self.dist_type == "normal":
                with poutine.reparam(config={"drift": LocScaleReparam()}):
                    drift = pyro.sample("drift", dist.Normal(zero_data, trans_scale).to_event(1))
            elif self.dist_type == "stable":
                with poutine.reparam(config={"drift": LocScaleReparam()}):
                    with poutine.reparam(config={"drift": StableReparam()}):
                        drift = pyro.sample("drift",
                                            dist.Stable(trans_stability, trans_skew, trans_scale).to_event(1))
            elif self.dist_type == "studentt":
                with poutine.reparam(config={"drift": LocScaleReparam(shape_params=["df"])}):
                    drift = pyro.sample("drift",
                                        dist.StudentT(trans_dof, zero_data, trans_scale).to_event(1))
            else:
                raise ValueError(self.dist_type)
        motion = drift.cumsum(-2)

        # Form prediction.
        prediction = motion + bias + (weight * covariates).sum(-1, keepdim=True)
        assert prediction.shape[-2:] == zero_data.shape

        # Form noise dist.
        if self.dist_type == "normal":
            obs_dist = dist.Normal(0, obs_scale)
            self.predict(obs_dist, prediction)
        elif self.dist_type == "stable":
            obs_dist = dist.Stable(obs_stability, obs_skew, obs_scale)
            with poutine.reparam(config={"residual": StableReparam()}):
                self.predict(obs_dist, prediction)
        elif self.dist_type == "studentt":
            obs_dist = dist.StudentT(obs_dof, 0, obs_scale)
            self.predict(obs_dist, prediction)
        else:
            raise ValueError(self.dist_type)


def main(args):
    pyro.enable_validation(__debug__)

    dataset = load_bart_od()
    print(dataset.keys())
    print(dataset["counts"].shape)
    print(" ".join(dataset["stations"]))

    T, O, D = dataset["counts"].shape
    data = dataset["counts"][:T // (24 * 7) * 24 * 7].reshape(T // (24 * 7), -1).sum(-1).log()
    data = data.unsqueeze(-1)
    print(dataset["counts"].shape, data.shape)
    covariates = periodic_features(len(data), 365.25 / 7)

    forecaster_options = {
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "log_every": args.log_every,
    }

    for dist_type in args.dist.split(","):
        assert dist_type in {"normal", "stable", "studentt"}, dist_type
        filename = os.path.join(
            RESULTS, os.path.basename(__file__)[:-3] + ".{}.pkl".format(dist_type))
        if args.force or not os.path.exists(filename):
            windows = backtest(data, covariates, lambda: Model(dist_type),
                               min_train_window=args.min_train_window,
                               test_window=args.test_window,
                               stride=args.stride,
                               forecaster_options=forecaster_options,
                               seed=args.seed)
            with open(filename, "wb") as f:
                pickle.dump(windows, f)
        with open(filename, "rb") as f:
            windows = pickle.load(f)
        for name in ["crps", "mae"]:
            values = torch.tensor([w[name] for w in windows])
            print("{} = {:0.3g} +- {:0.2g}".format(name, values.mean(), values.std()))


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.3.0")
    parser = argparse.ArgumentParser(description="Univariate BART forecasting")
    parser.add_argument("--dist", default="normal,stable,studentt")
    parser.add_argument("--min-train-window", default=52 * 2, type=int)
    parser.add_argument("--test-window", default=52, type=int)
    parser.add_argument("-s", "--stride", default=4, type=int)
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--seed", default=1234567890, type=int)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda"):
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("Running on device {}".format(args.device))

    try:
        main(args)
    except Exception as e:
        import pdb
        print(e)
        pdb.post_mortem(e.__traceback__)
