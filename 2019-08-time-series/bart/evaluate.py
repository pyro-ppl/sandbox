import argparse
import os
import subprocess

import torch

from preprocess import load_hourly_od


def config_to_basename(config):
    return '.'.join(arg.lstrip('-').replace('-', '_') for arg in config)


def make_splits(args, dataset):
    """
    Make train-test splits in time.
    """
    total_hours = len(dataset['counts'])
    if args.truncate:
        total_hours = min(total_hours, args.truncate)

    # Dataset starts on a Saturday.
    assert dataset['start_date'][0].strftime('%A') == 'Saturday'

    # Ridership is minimum early Sunday morning.
    split_hour_of_week = 29

    # We train HMM on at least one year of historical data.
    min_hours = 365 * 24

    stride = 24 * 7

    result = list(range(min_hours + split_hour_of_week,
                        total_hours - args.forecast_hours,
                        stride))
    assert result, 'truncated too short'
    return result


def forecast_one(args, config):
    basename = config_to_basename(config + ("forecast",))
    forecast_path = f'{args.results}/{basename}.pkl'

    if args.force or not os.path.exists(forecast_path):
        command = ['python'] if __debug__ else ['python', '-O']
        command.append('main.py')
        command.append('--pdb' if args.pdb else '--no-pdb')
        if args.verbose:
            command.append('--verbose')
        command.append(f'--num-steps={args.num_steps}')
        command.append('--param-store-filename=/dev/null')
        command.append('--forecaster-filename=/dev/null')
        command.append('--training-filename=/dev/null')
        command.extend(config)
        command.append(f'--forecast-filename={forecast_path}')
        print('# {}'.format(' '.join(config)))
        if args.verbose:
            print(' \\\n '.join(command))
        subprocess.check_call(command)

    return torch.load(forecast_path)


def main(args):
    dataset = load_hourly_od()
    if not os.path.exists(args.results):
        os.mkdir(args.results)

    variants = [
        (),
        # ('--funsor',),
        # ('--funsor', '--analytic-kl'),
    ]
    results = {}
    for config in variants:
        results[config] = []
        for truncate in make_splits(args, dataset):
            result = forecast_one(args, config + ('--truncate={}'.format(truncate),))
            results[config].append(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BART forecasting evaluation")
    parser.add_argument("--results", default="results")
    parser.add_argument("--truncate", default=0, type=int)
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("--forecast-hours", default=24 * 7, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--pdb", action="store_true")
    parser.add_argument("--no-pdb", dest="pdb", action="store_false")
    args = parser.parse_args()

    main(args)
