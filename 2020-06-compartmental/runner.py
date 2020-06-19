# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import subprocess
import sys


def short_uni_synth():
    base = [
        sys.executable,
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
            if num_bins > 1:
                num_seeds = 2
            elif num_samples > 200:
                num_seeds = 5
            else:
                num_seeds = 10
            for rng_seed in range(num_seeds):
                yield base + ["--mcmc",
                              "--warmup-steps=200",
                              f"--num-samples={num_samples}",
                              f"--num-bins={num_bins}",
                              f"--rng-seed={rng_seed}"]


def long_uni_synth():
    base = [
        sys.executable,
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
    for num_bins in [1, 2, 4]:
        for num_samples in [200, 500, 1000]:
            if num_bins > 1:
                num_seeds = 2
            elif num_samples > 200:
                num_seeds = 5
            else:
                num_seeds = 10
            for rng_seed in range(num_seeds):
                yield base + ["--mcmc",
                              "--warmup-steps=200",
                              f"--num-samples={num_samples}",
                              f"--num-bins={num_bins}",
                              f"--rng-seed={rng_seed}"]


def main(args):
    tasks = list(globals()[args.experiment]())
    for task in tasks:
        print(" \\\n  ".join(task))
        if not args.dry_run:
            subprocess.check_call(task)

    print("-------------------------")
    print("COMPLETED {} TASKS".format(len(tasks)))
    print("-------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment runner")
    parser.add_argument("--experiment")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    main(args)
