# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import multiprocessing
import os
import random
import subprocess
import sys

CPUS = multiprocessing.cpu_count()
ENV = os.environ.copy()
ROOT = os.path.dirname(os.path.abspath(__file__))
TEMP = os.path.join(ROOT, "temp")
LOGS = os.path.join(ROOT, "logs")
ERRORS = os.path.join(ROOT, "errors")
RESULTS = os.path.join(ROOT, "results")

# Ensure directories exist.
for path in [TEMP, LOGS, ERRORS, RESULTS]:
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            assert os.path.exists(path)


def work(task):
    args, spec = task
    basename = (args.script_filename + "." +
                "_".join("{}={}".format(k, v) for k, v in spec.items()))
    result_file = os.path.join(RESULTS, basename + ".pkl")
    if os.path.exists(result_file) and not args.force:
        return result_file
    elif args.skip:
        return None

    temp_file = os.path.join(TEMP, basename + ".pkl")
    log_file = os.path.join(LOGS, basename + ".txt")
    spec["outfile"] = temp_file
    command = ([sys.executable, args.script_filename] +
               ["--{}={}".format(k, v) for k, v in spec.items()])
    print(" ".join(command))
    if args.dry_run:
        return result_file
    try:
        with open(log_file, "w") as f:
            subprocess.check_call(command, stderr=f, stdout=f, env=ENV)
        os.rename(temp_file, result_file)  # Use rename to make write atomic.
        return result_file
    except subprocess.CalledProcessError as e:
        pdb_command = [sys.executable, "-m", "pdb", "-cc"] + command[1:-1]
        msg = "{}\nTo reproduce, run:\n{}".format(e, " \\\n  ".join(pdb_command))
        print(msg)
        with open(os.path.join(ERRORS, basename + ".txt"), "w") as f:
            f.write(msg)
        return None


def main(args):
    tasks = []
    with open(args.args_filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            spec = {k: v for k, v in zip(header, row) if v}
            for seed in args.rng_seed.split(","):
                spec["rng-seed"] = seed
                tasks.append((args, spec.copy()))
    if args.shuffle:
        random.shuffle(tasks)

    if args.num_workers == 1:
        map_ = map
    else:
        print("Running {} tasks on {} workers".format(len(tasks), args.num_workers))
        map_ = multiprocessing.Pool(args.num_workers).map
    results = list(map_(work, tasks))
    if args.skip:
        results = [r for r in results if r is not None]
    else:
        assert all(results)

    results.sort()
    if args.outfile:
        with open(args.outfile, "w") as f:
            f.write("\n".join(results))

    print("-------------------------")
    print("COMPLETED {}/{} TASKS".format(len(results), len(tasks)))
    print("-------------------------")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment runner")
    parser.add_argument("--script-filename")
    parser.add_argument("--args-filename")
    parser.add_argument("--outfile")
    parser.add_argument("--rng-seed", default="0")
    parser.add_argument("--num-workers", type=int, default=CPUS)
    parser.add_argument("--cores-per-worker", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.cores_per_worker:
        args.num_workers = max(1, CPUS // args.cores_per_worker)
        ENV["OMP_NUM_THREAD"] = min(CPUS, 2 * args.cores_per_worker)
    if args.dry_run:
        args.num_workers = 1

    main(args)
