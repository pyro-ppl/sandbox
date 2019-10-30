import argparse
import logging
import multiprocessing
import os
import subprocess

import torch
from pyro.ops.stats import crps_empirical

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

    # We train HMM on at least six years of historical data.
    min_hours = 6 * 365 * 24

    stride = 24 * 7

    result = list(range(min_hours + split_hour_of_week,
                        total_hours - args.forecast_hours,
                        stride))
    logging.info(f'Created {len(result)} test/train splits')
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
        command.append(f'--forecast-hours={args.forecast_hours}')
        command.append(f'--num-samples={args.num_samples}')
        command.append('--param-store-filename=/dev/null')
        command.append('--forecaster-filename=/dev/null')
        command.append('--training-filename=/dev/null')
        command.extend(config)
        command.append(f'--forecast-filename={forecast_path}')
        logging.info('# {}'.format(' '.join(command)))
        logging.debug(' \\\n '.join(command))
        subprocess.check_call(command)

    return torch.load(forecast_path, map_location=args.device)


def eval_one(args, result):
    logging.debug('evaluating')
    pred = result['forecast']
    truth = result['truth']

    t, n, n = truth.shape
    assert pred.shape == (args.num_samples, t, n, n)

    # Evaluate point estimate using Mean Absolute Error.
    mae = float((pred.median(dim=0).values - truth).abs().mean())

    # Evaluate uncertainty using negative Continuous Ranked Probability Score.
    crps = float(crps_empirical(pred, truth).mean())

    result = {'MAE': mae, 'CRPS': crps, 'ELBO': result['log_prob']}
    logging.info(result)
    return result


def process_task(task):
    args, config, truncate = task
    logging.basicConfig(format='%(process) 5d %(relativeCreated) 9d %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    forecast = forecast_one(args, config + ('--truncate={}'.format(truncate),))
    metrics = eval_one(args, forecast)
    del forecast
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    return config, truncate, metrics


def main(args):
    dataset = load_hourly_od()
    if not os.path.exists(args.results):
        os.mkdir(args.results)

    configs = [
        # (),
        # ('--mean-field',),
        ('--state-dim=2', '--guide-rank=2'),
        ('--state-dim=2', '--guide-rank=2', '--mean-field'),
        ('--state-dim=4', '--guide-rank=4'),
        ('--state-dim=4', '--guide-rank=4', '--mean-field'),
        # ('--funsor',),
        # ('--funsor', '--analytic-kl'),
    ]
    splits = make_splits(args, dataset)
    results = {}
    map_ = map if args.parallel == 1 else multiprocessing.Pool(args.parallel).map

    results = list(map_(process_task, [
        (args, config, truncate)
        for config in configs
        for truncate in splits
    ]))

    # Group by config and by truncate.
    metrics = {}
    for config, truncate, metric in results:
        metrics.setdefault(config, {}).setdefault(truncate, metric)
    results = {'args': args, 'metrics': metrics}

    eval_filename = os.path.abspath(f'{args.results}/eval.pkl')
    logging.info(f'Saving results to {eval_filename}')
    torch.save(results, eval_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BART forecasting evaluation")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--results", default="results")
    parser.add_argument("--truncate", default=0, type=int)
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("--forecast-hours", default=24 * 7, type=int)
    parser.add_argument("--num-samples", default=99, type=int)
    parser.add_argument("--device", default="")
    parser.add_argument("--cuda", dest="device", action="store_const", const="cuda")
    parser.add_argument("-p", "--parallel", default=1, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--pdb", action="store_true")
    parser.add_argument("--no-pdb", dest="pdb", action="store_false")
    args = parser.parse_args()
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.parallel > 1 and args.device.startswith("cuda"):
        multiprocessing.set_start_method('forkserver')

    logging.basicConfig(format='%(process) 5d %(relativeCreated) 9d %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)

    main(args)
