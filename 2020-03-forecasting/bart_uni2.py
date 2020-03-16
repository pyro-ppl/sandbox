import argparse
import logging
import os
import pickle

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from pyro.contrib.examples.bart import load_bart_od
from pyro.contrib.forecast import ForecastingModel, backtest
from pyro.infer.reparam import LocScaleReparam, StableReparam
from pyro.ops.tensor_utils import periodic_cumsum, periodic_features, periodic_repeat

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS = os.environ.get("BART_RESULTS", RESULTS)
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)


class Model(ForecastingModel):
    def __init__(self, dist_type):
        super().__init__()
        self.dist_type = dist_type

    def model(self, zero_data, covariates):
        duration = zero_data.size(-2)
        feature_dim = covariates.size(-1)

        # Globals.
        bias = pyro.sample("bias", dist.Normal(0, 10))
        weight = pyro.sample("weight", dist.Normal(0, 0.1).expand([feature_dim]).to_event(1))
        how_scale = pyro.sample("how_scale", dist.LogNormal(-20, 5))
        trans_scale = pyro.sample("trans_scale", dist.LogNormal(-20, 5))
        obs_scale = pyro.sample("obs_scale", dist.LogNormal(-5, 5))
        if self.dist_type == "normal":
            pass
        elif self.dist_type == "stable":
            stability = pyro.sample("stability", dist.Uniform(1, 2))
            how_skew = pyro.sample("how_skew", dist.Uniform(-1, 1))
            trans_skew = pyro.sample("trans_skew", dist.Uniform(-1, 1))
            obs_skew = pyro.sample("obs_skew", dist.Uniform(-1, 1))
        elif self.dist_type == "studentt":
            dof = pyro.sample("dof", dist.Uniform(1, 10))
        else:
            raise ValueError(self.dist_type)

        # Series locals.
        with pyro.plate("hour_of_week", 24 * 7, dim=-1):
            how_init = pyro.sample("how_init", dist.Normal(0, 10))
        with self.time_plate:
            if self.dist_type == "normal":
                with poutine.reparam(config={"drift": LocScaleReparam(), "how_drift": LocScaleReparam()}):
                    drift = pyro.sample("drift", dist.Normal(0, trans_scale))
                    how_drift = pyro.sample("how_drift", dist.Normal(0, how_scale))
            elif self.dist_type == "stable":
                with poutine.reparam(config={"drift": LocScaleReparam(), "how_drift": LocScaleReparam()}):
                    with poutine.reparam(config={"drift": StableReparam(), "how_drift": StableReparam()}):
                        drift = pyro.sample("drift", dist.Stable(stability, trans_skew, trans_scale))
                        how_drift = pyro.sample("how_drift", dist.Stable(stability, how_skew, how_scale))
            elif self.dist_type == "studentt":
                with poutine.reparam(config={"drift": LocScaleReparam(shape_params=["df"]),
                                             "how_drift": LocScaleReparam(shape_params=["df"])}):
                    drift = pyro.sample("drift", dist.StudentT(dof, 0, trans_scale))
                    how_drift = pyro.sample("how_drift", dist.StudentT(dof, 0, how_scale))
            else:
                raise ValueError(self.dist_type)

        # Form prediction.
        periodic_lowpass = (weight * covariates).sum(-1)
        periodic_highpass = (periodic_repeat(how_init, duration, dim=-1) +
                             periodic_cumsum(how_drift, 24 * 7, dim=-1))
        motion = drift.cumsum(-1)
        prediction = bias + periodic_highpass + periodic_lowpass + motion
        prediction = prediction.unsqueeze(-1)
        assert prediction.shape[-2:] == zero_data.shape

        # Form noise dist.
        if self.dist_type == "normal":
            obs_dist = dist.Normal(0, obs_scale.unsqueeze(-1))
            self.predict(obs_dist, prediction)
        elif self.dist_type == "stable":
            obs_dist = dist.Stable(stability.unsqueeze(-1), obs_skew.unsqueeze(-1), obs_scale.unsqueeze(-1))
            with poutine.reparam(config={"residual": StableReparam()}):
                self.predict(obs_dist, prediction)
        elif self.dist_type == "studentt":
            obs_dist = dist.StudentT(dof.unsqueeze(-1), 0, obs_scale.unsqueeze(-1))
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
    data = dataset["counts"].reshape(T, -1).sum(-1, keepdim=True).log1p()
    data = data.to(args.device)
    print(dataset["counts"].shape, data.shape)
    covariates = periodic_features(len(data), 365.25 * 24, 24)

    forecaster_options = {
        "learning_rate": args.learning_rate,
        "clip_norm": args.clip_norm,
        "num_steps": args.num_steps,
        "log_every": args.log_every,
    }

    def transform(pred, truth):
        pred = pred.clamp(min=0)
        return pred, truth

    for dist_type in args.dist.split(","):
        assert dist_type in {"normal", "stable", "studentt"}, dist_type
        print(dist_type)
        filename = os.path.join(
            RESULTS, os.path.basename(__file__)[:-3] + ".{}.pkl".format(dist_type))
        if args.force or not os.path.exists(filename):
            windows = backtest(data, covariates, lambda: Model(dist_type),
                               min_train_window=args.min_train_window,
                               test_window=args.test_window,
                               stride=args.stride,
                               forecaster_options=forecaster_options,
                               batch_size=args.batch_size,
                               transform=transform,
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
    parser = argparse.ArgumentParser(description="Univariate BART hourly forecasting")
    parser.add_argument("--dist", default="normal,stable,studentt")
    parser.add_argument("--min-train-window", default=4 * 365 * 24, type=int)
    parser.add_argument("--test-window", default=4 * 7 * 24, type=int)
    parser.add_argument("-s", "--stride", default=30 * 24, type=int)
    parser.add_argument("-b", "--batch-size", default=10, type=int)
    parser.add_argument("-n", "--num-steps", default=2001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--clip-norm", default=50, type=float)
    parser.add_argument("-l", "--log-every", default=100, type=int)
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
