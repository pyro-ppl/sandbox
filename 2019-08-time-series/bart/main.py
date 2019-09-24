import argparse
import logging

import pyro
import torch

from forecast import train
from preprocess import load_hourly_od


def main(args):
    assert pyro.__version__ >= "0.4.1"
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.seed)
    dataset = load_hourly_od(args)
    if args.tiny:
        dataset["stations"] = dataset["stations"][:args.tiny]
        dataset["counts"] = dataset["counts"][:, :args.tiny, :args.tiny]
    forecaster = train(args, dataset)
    if forecaster is None:
        return

    num_samples = 10
    forecast = forecaster(0, 24 * 7, 24)
    assert forecast.shape == (24,) + dataset["counts"].shape[-2:]
    forecast = forecaster(0, 24 * 7, 24, num_samples=num_samples)
    assert forecast.shape == (num_samples, 24) + dataset["counts"].shape[-2:]
    return forecast


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BART origin-destination forecast")
    parser.add_argument("--param-store-filename", default="pyro_param_store.pkl")
    parser.add_argument("--forecaster-filename", default="forecaster.pkl")
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
