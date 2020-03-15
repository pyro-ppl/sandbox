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
from pyro.infer.reparam import LocScaleReparam, StableReparam, SymmetricStableReparam
from pyro.ops.tensor_utils import periodic_repeat

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS = os.environ.get("BART_RESULTS", RESULTS)
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)


def _safe_transpose(tensor, dim1, dim2):
    """
    Transposes two tensor dims, expanding ``tensor.shape`` by 1's on the left
    as needed.
    """
    assert dim1 < 0 and dim2 < 0
    if tensor.dim() < -min(dim1, dim2):
        tensor = tensor.reshape((1,) * (-min(dim1, dim2) - tensor.dim()) + tensor.shape)
    return tensor.transpose(dim1, dim2)


class Model(ForecastingModel):
    def __init__(self, dist_type):
        super().__init__()
        self.dist_type = dist_type

    def model(self, zero_data, covariates):
        num_stations, num_stations, duration, one = zero_data.shape

        origin_plate = pyro.plate("origin", num_stations, dim=-3)
        destin_plate = pyro.plate("destin", num_stations, dim=-2)
        hour_of_week_plate = pyro.plate("hour_of_week", 24 * 7, dim=-1)

        # Globals.
        trans_scale = pyro.sample("trans_scale", dist.LogNormal(-20, 5))
        if self.dist_type == "normal":
            pass
        elif self.dist_type == "stable":
            trans_stability = pyro.sample("trans_stability", dist.Uniform(1, 2))
            obs_stability = pyro.sample("obs_stability", dist.Uniform(1, 2))
            obs_skew = pyro.sample("obs_skew", dist.Uniform(-1, 1))
        elif self.dist_type == "studentt":
            trans_dof = pyro.sample("trans_dof", dist.Uniform(1, 10))
            obs_dof = pyro.sample("obs_dof", dist.Uniform(1, 10))
        else:
            raise ValueError(self.dist_type)

        # Series locals.
        with origin_plate:
            origin_scale = pyro.sample("origin_scale", dist.LogNormal(-5, 5))
            with hour_of_week_plate:
                origin_seasonal = pyro.sample("origin_seasonal", dist.Normal(0, 5))
        with destin_plate:
            destin_scale = pyro.sample("destin_scale", dist.LogNormal(-5, 5))
            with hour_of_week_plate:
                destin_seasonal = pyro.sample("destin_seasonal", dist.Normal(0, 5))
            with self.time_plate:
                if self.dist_type == "normal":
                    with poutine.reparam(config={"drift": LocScaleReparam()}):
                        drift = pyro.sample("drift", dist.Normal(0, trans_scale))
                elif self.dist_type == "stable":
                    with poutine.reparam(config={"drift": LocScaleReparam()}):
                        with poutine.reparam(config={"drift": SymmetricStableReparam()}):
                            drift = pyro.sample(
                                "drift", dist.Stable(trans_stability, 0, trans_scale))
                elif self.dist_type == "studentt":
                    with poutine.reparam(config={"drift": LocScaleReparam(shape_params=["df"])}):
                        drift = pyro.sample(
                            "drift", dist.StudentT(trans_dof, 0, trans_scale))
                else:
                    raise ValueError(self.dist_type)
        with origin_plate, destin_plate:
            pairwise = pyro.sample("pairwise", dist.Normal(0, 1))

        # Form prediction.
        destin_motion = drift.cumsum(dim=-1)
        origin_motion = _safe_transpose(destin_motion, destin_plate.dim, origin_plate.dim)
        destin_prediction = destin_motion + periodic_repeat(destin_seasonal, duration, dim=-1)
        origin_prediction = origin_motion + periodic_repeat(origin_seasonal, duration, dim=-1)
        assert destin_prediction.shape[-2:] == (destin_plate.subsample_size, duration)
        assert origin_prediction.shape[-3:] == (origin_plate.subsample_size, 1, duration)
        prediction = destin_prediction + origin_prediction + pairwise  # Note the broadcast.
        assert prediction.shape[-3:] == (origin_plate.subsample_size,
                                         destin_plate.subsample_size,
                                         duration)

        # Form noise dist.
        scale = origin_scale + destin_scale
        scale = scale.unsqueeze(-1)
        prediction = prediction.unsqueeze(-1)
        with origin_plate, destin_plate:
            if self.dist_type == "normal":
                noise_dist = dist.Normal(0, scale)
                self.predict(noise_dist, prediction)
            elif self.dist_type == "stable":
                noise_dist = dist.Stable(obs_stability.unsqueeze(-1), obs_skew.unsqueeze(-1), scale)
                with poutine.reparam(config={"residual": StableReparam()}):
                    self.predict(noise_dist, prediction)
            elif self.dist_type == "studentt":
                noise_dist = dist.StudentT(obs_dof.unsqueeze(-1), 0, scale)
                self.predict(noise_dist, prediction)
            else:
                raise ValueError(self.dist_type)


def main(args):
    pyro.enable_validation(__debug__)

    dataset = load_bart_od()
    print(dataset.keys())
    print(dataset["counts"].shape)
    print(" ".join(dataset["stations"]))

    data = dataset["counts"].permute(1, 2, 0).unsqueeze(-1).log1p().contiguous()
    data = data.to(args.device)
    print(dataset["counts"].shape, data.shape)
    covariates = torch.zeros(data.size(-2), 0)  # empty

    def create_plates(zero_data, covariates):
        num_origins, num_destins, duration, one = zero_data.shape
        origin_plate = pyro.plate("origin", num_origins, subsample_size=10, dim=-3)
        # We reuse the subsample so that both plates are subsampled identically.
        with origin_plate as subsample:
            print("subsample.shape = {}".format(subsample.shape))
            pass
        destin_plate = pyro.plate("destin", num_destins, subsample=subsample, dim=-2)
        return origin_plate, destin_plate

    forecaster_options = {
        "create_plates": create_plates,
        "learning_rate": args.learning_rate,
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
                               train_window=args.train_window,
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
    parser = argparse.ArgumentParser(description="Multivariate BART forecasting")
    parser.add_argument("--dist", default="normal,stable,studentt")
    parser.add_argument("--train-window", default=24 * 90, type=int)
    parser.add_argument("--test-window", default=24 * 14, type=int)
    parser.add_argument("-s", "--stride", default=24 * 100, type=int)
    parser.add_argument("-b", "--batch-size", default=20, type=int)
    parser.add_argument("-n", "--num-steps", default=2001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("-l", "--log-every", default=50, type=int)
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
