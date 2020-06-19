# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import subprocess
import sys
from hashlib import sha1
from importlib import import_module

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")


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
            num_warmup = max(200, int(round(0.4 * num_samples)))
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
    for num_samples in [200, 500, 1000, 2000, 5000]:
        num_warmup = max(200, int(round(0.4 * num_samples)))
        for rng_seed in range(10):
            yield base + ["--mcmc",
                          "--num-bins=1",
                          f"--warmup-steps={num_warmup}",
                          f"--num-samples={num_samples}",
                          f"--rng-seed={rng_seed}"]
    for num_bins in [2, 4]:
        for num_samples in [200, 500, 1000]:
            num_warmup = max(200, int(round(0.4 * num_samples)))
            for rng_seed in range(2):
                yield base + ["--mcmc",
                              f"--warmup-steps={num_warmup}",
                              f"--num-samples={num_samples}",
                              f"--num-bins={num_bins}",
                              f"--rng-seed={rng_seed}"]


def main(args):
    tasks = list(globals()[args.experiment]())
    for task in tasks:
        print(" \\\n  ".join(task))
        if args.dry_run:
            continue

        # Optimization: Parse args to compute output filename and check for
        # previous completion. This is equivalent to but much cheaper than
        # creating a new process and checking in the process.
        script = task[1]
        parser = import_module(script.replace(".py", "")).Parser()
        args_dict = parser.parse_args(task[2:]).__dict__
        unique = script, sorted(args_dict.items())
        fingerprint = sha1(str(unique).encode()).hexdigest()
        outfile = os.path.join(RESULTS, fingerprint + ".pkl")
        if os.path.exists(outfile):
            continue

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
