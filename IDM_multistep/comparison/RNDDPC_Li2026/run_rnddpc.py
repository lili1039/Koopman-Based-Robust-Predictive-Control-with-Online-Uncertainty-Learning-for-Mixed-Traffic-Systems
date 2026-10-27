"""Single entry point for the maintained RNDDPC-HI baseline."""
import argparse
import contextlib
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['both', 'tracking', 'braking'], default='both')
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--coverage', type=float, default=0.10,
                        help='Coordinate-wise absolute residual quantile; no additional scaling.')
    args = parser.parse_args()
    if args.horizon < 1 or not 0 < args.coverage < 1:
        parser.error('horizon must be positive and coverage must lie between 0 and 1')

    base = Path(__file__).resolve().parent
    os.chdir(base)
    # The entry point owns the experiment configuration; stale shell settings
    # from other controllers must not silently change this baseline.
    for key in list(os.environ):
        if key.startswith(('KMPC_', 'RNDDPC_')):
            del os.environ[key]
    os.environ.update({
        'KMPC_SCENARIO': 'tracking', 'KMPC_NFUT': str(args.horizon),
        'KMPC_SUBPLATOON': '2,3,1,4,2', 'KMPC_CONTROL': 'RNDDPC',
        'KMPC_SIGNAL_DELAY': '0', 'KMPC_CAV_PRED_LOSS': '0',
        'RNDDPC_QCOVER': str(args.coverage),
        'MPLBACKEND': 'Agg', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
    })
    import torch
    import cvxpy as cp
    import functions as f
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    f.np.random.seed(0)
    if f.CONIC_SOLVER not in cp.installed_solvers():
        raise RuntimeError('CLARABEL is not installed in this Python environment')

    import RNDDPC as driver
    is_main = args.horizon == 5 and args.coverage == 0.10
    quantile_label = format(100 * args.coverage, '.8g').replace('.', 'p')
    root = base / 'results' / ('main_results' if is_main else
        f'feasibility_N{args.horizon}_residual{quantile_label}_{datetime.now():%Y%m%d_%H%M%S}')
    previous = root.with_name('.' + root.name + '_previous')
    if root.exists():
        if previous.exists():
            raise RuntimeError(f'Previous results await review: {previous}')
        root.rename(previous)
    root.mkdir(parents=True, exist_ok=False)
    print(f'Results: {root}', flush=True)
    summary = []
    scenarios = ['tracking', 'braking'] if args.scenario == 'both' else [args.scenario]
    for scenario in scenarios:
        destination = root / scenario
        destination.mkdir()
        f.scenario = driver.scenario = scenario
        f.output_dir = driver.output_dir = str(destination)
        print(f'Starting {scenario} (N={args.horizon}, quantile={args.coverage})', flush=True)
        with (destination / 'run.log').open('w', encoding='utf-8') as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                try:
                    driver.main()
                    row = {'scenario': scenario, 'completed': True}
                except Exception as error:
                    traceback.print_exc()
                    row = {'scenario': scenario, 'completed': False, 'message': str(error)}
        if row['completed']:
            import csv
            with (destination / 'metrics.csv').open(encoding='utf-8-sig') as stream:
                row['metrics'] = list(csv.DictReader(stream))
        summary.append(row)
        print(json.dumps(row), flush=True)
    return 0 if all(row['completed'] for row in summary) else 1


if __name__ == '__main__':
    # Keep the timing runs mutually exclusive, including direct Python launches.
    lock = Path(__file__).resolve().parent / '.rnddpc_running.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        sys.exit(f'RNDDPC is already running; see {lock}')
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        sys.exit(main())
    finally:
        lock.unlink(missing_ok=True)
