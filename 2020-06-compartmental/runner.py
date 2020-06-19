# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import pickle
import subprocess
import sys
from importlib import import_module

from util import get_filename


class Experiment:
    """
    An experiment consists of a collection of tasks.
    Each task generates a datapoint by running a python script.
    Result datapoints are cached in pickle files named by fingerprint.
    """
    def __init__(self, generate_tasks):
        self.__name__ = generate_tasks.__name__
        self.tasks = [[sys.executable] + task for task in generate_tasks()]
        self.files = []
        for task in self.tasks:
            script = task[1]
            parser = import_module(script.replace(".py", "")).Parser()
            outfile = get_filename(script, parser.parse_args(task[2:]))
            self.files.append(outfile)

    @property
    def results(self):
        """
        Iterates over the subset of experiment results that have been generated.
        """
        for outfile in self.files:
            if os.path.exists(outfile):
                with open(outfile, "rb") as f:
                    result = pickle.load(f)
                yield result


@Experiment
def short_uni_synth():
    base = [
        "uni_synth.py",
        "--population=1000",
        "--duration=20", "--forecast=10",
        "--R0=3", "--incubation-time=2", "--recovery-time=4",
    ]
    for svi_steps in [1000, 2000, 5000, 10000]:
        for rng_seed in range(10):
            yield base + ["--svi",
                          "--num-samples=1000",
                          f"--svi-steps={svi_steps}",
                          f"--rng-seed={rng_seed}"]
    for num_bins in [1, 2, 4]:
        for num_samples in [200, 500, 1000]:
            num_warmup = int(round(0.4 * num_samples))
            if num_bins == 1:
                num_seeds = 10
            else:
                num_seeds = 2
            for rng_seed in range(num_seeds):
                yield base + ["--mcmc",
                              f"--warmup-steps={num_warmup}",
                              f"--num-samples={num_samples}",
                              f"--num-bins={num_bins}",
                              f"--rng-seed={rng_seed}"]


@Experiment
def long_uni_synth():
    base = [
        "uni_synth.py",
        "--population=100000",
        "--duration=100", "--forecast=30",
        "--R0=2.5", "--incubation-time=4", "--recovery-time=10",
    ]
    for svi_steps in [1000, 2000, 5000, 10000]:
        for rng_seed in range(10):
            yield base + ["--svi",
                          "--num-samples=1000",
                          f"--svi-steps={svi_steps}",
                          f"--rng-seed={rng_seed}"]
    for num_samples in [200, 500, 1000, 2000, 5000]:
        num_warmup = int(round(0.4 * num_samples))
        for rng_seed in range(10):
            yield base + ["--mcmc",
                          "--num-bins=1",
                          f"--warmup-steps={num_warmup}",
                          f"--num-samples={num_samples}",
                          f"--rng-seed={rng_seed}"]
    for num_bins in [2, 4]:
        for num_samples in [200, 500, 1000]:
            num_warmup = int(round(0.4 * num_samples))
            for rng_seed in range(2):
                yield base + ["--mcmc",
                              f"--warmup-steps={num_warmup}",
                              f"--num-samples={num_samples}",
                              f"--num-bins={num_bins}",
                              f"--rng-seed={rng_seed}"]


def main(args):
    experiment = globals()[args.experiment]
    for task, outfile in zip(experiment.tasks, experiment.files):
        print(" \\\n  ".join(task))
        if args.dry_run or os.path.exists(outfile):
            continue
        subprocess.check_call(task)

    print("-------------------------")
    print("COMPLETED {} TASKS".format(len(experiment.tasks)))
    print("-------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment runner")
    parser.add_argument("--experiment")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    main(args)
