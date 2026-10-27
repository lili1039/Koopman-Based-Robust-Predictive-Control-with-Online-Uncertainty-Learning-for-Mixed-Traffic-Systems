"""Internal runner. Configure and launch uncertainty_ablation.bat."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import platform
from importlib.metadata import version

BASE = Path(__file__).resolve().parent
MODES = ('online', 'frozen', 'global', 'nominal')
THREAD_KEYS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'PYTHONHASHSEED', 'PYTHONUNBUFFERED', 'MPLBACKEND')


def configuration():
    """Validate BAT configuration without loading the simulation or solvers."""
    required = ('KMPC_BATCH_WORKERS', 'KMPC_BATCH_START_STEP', 'KMPC_BATCH_INTERVAL_STEPS',
                'KMPC_BATCH_COUNT', 'KMPC_BATCH_RESULT_PREFIX', 'KMPC_BATCH_LAUNCHER',
                'KMPC_CUTIN_GAP', 'KMPC_CUTIN_FINAL_SPEED', 'KMPC_CUTIN_BRAKE_DURATION',
                'KMPC_VEHICLE_CSV', 'KMPC_ABLATION_FREEZE_STEP', 'KMPC_NPOINT', 'KMPC_NFUT')
    missing = [k for k in required if not os.environ.get(k, '').strip()]
    if missing:
        raise ValueError('Launch uncertainty_ablation.bat; missing: ' + ', '.join(missing))
    env = {k: v for k, v in os.environ.items() if k.startswith('KMPC_') or k in THREAD_KEYS}
    workers = int(env['KMPC_BATCH_WORKERS'])
    start = int(env['KMPC_BATCH_START_STEP'])
    interval = int(env['KMPC_BATCH_INTERVAL_STEPS'])
    count = int(env['KMPC_BATCH_COUNT'])
    if not 1 <= workers <= 4:
        raise ValueError('Batch workers must be in [1, 4].')
    if start < 84 or interval <= 0 or count < 1:
        raise ValueError('Use a positive arithmetic grid starting at step 84 or later.')
    times = [round((start + i * interval) * 0.12, 2) for i in range(count)]
    gap, speed, duration = [float(env[k]) for k in
                          ('KMPC_CUTIN_GAP', 'KMPC_CUTIN_FINAL_SPEED', 'KMPC_CUTIN_BRAKE_DURATION')]
    if not all(math.isfinite(v) for v in (gap, speed, duration)) or gap <= 0 or not 0 <= speed < 26 or duration <= 0:
        raise ValueError('Require gap > 0, final speed in [0, 26), and braking duration > 0.')
    if times[-1] + duration >= 79.92:
        raise ValueError('All braking phases must end before the final 79.92 s sample.')
    if not 1 <= int(env['KMPC_NFUT']) <= 15:
        raise ValueError('Prediction horizon must be in [1, 15].')
    if int(env['KMPC_NPOINT']) < 1 or not 1 <= int(env['KMPC_EP_WINDOW']) <= 50:
        raise ValueError('Require a positive error window and an acceleration window in [1, 50].')
    if not 0 <= float(env['KMPC_LAM']) <= 1:
        raise ValueError('Alpha smoothing must be in [0, 1].')
    if int(env['KMPC_CORR_THREADS']) < 1 or int(env['KMPC_GRB_THREADS']) < 1:
        raise ValueError('Solver and correction thread counts must be positive.')
    if not int(env['KMPC_NFUT']) + int(env['KMPC_NPOINT']) <= int(env['KMPC_ABLATION_FREEZE_STEP']) < start:
        raise ValueError('Freeze after sufficient error samples and before the first cut-in.')
    if not float(env['KMPC_ACC_MIN']) < 0 < float(env['KMPC_ACC_MAX']):
        raise ValueError('Acceleration bounds must allow braking and acceleration.')
    if any(v not in ('1', '2', '3', '4') for v in env['KMPC_SUBPLATOON'].split(',')):
        raise ValueError('Subplatoon entries must be HDV counts/model ids 1..4.')
    for low, high in (('SPA_MIN', 'SPA_MAX'), ('VEL_MIN', 'VEL_MAX'), ('ACC_MIN', 'ACC_MAX')):
        if not float(env['KMPC_' + low]) < float(env['KMPC_' + high]):
            raise ValueError('Constraint lower bounds must be below upper bounds.')
    for mode in MODES:
        if env.get('KMPC_BATCH_SLACK_' + mode.upper(), '0') not in ('0', '1'):
            raise ValueError('Per-method slack flags must be 0 or 1.')
    if float(env['KMPC_SLACK_W']) <= 0 or float(env['KMPC_SLACK_W_L2']) < 0 or float(env['KMPC_SLACK_TOL']) <= 0:
        raise ValueError('Require positive L1 slack weight/tolerance and nonnegative L2 weight.')
    fixed = {'KMPC_SCENARIO': 'cutin', 'KMPC_CONTROL': 'Distributed MPC',
             'KMPC_DISTURBANCE_ESTIMATE': '1',
             'KMPC_SIGNAL_DELAY': '0', 'KMPC_CAV_PRED_LOSS': '0', 'KMPC_SIM_MAXTIME': '0'}
    if any(env.get(k) != v for k, v in fixed.items()):
        raise ValueError('Keep the fixed comparison protocol settings defined in the BAT.')
    prefix = env['KMPC_BATCH_RESULT_PREFIX']
    if not prefix.startswith('result_') or any(c in prefix for c in '/\\:') or prefix in ('.', '..'):
        raise ValueError('Use a simple folder prefix beginning with result_.')
    vehicle = Path(env['KMPC_VEHICLE_CSV'])
    if not vehicle.is_absolute():
        vehicle = BASE / vehicle
    with vehicle.open('rb'):
        pass
    env['KMPC_VEHICLE_CSV'] = str(vehicle.absolute())
    with Path(env['KMPC_BATCH_LAUNCHER']).open('rb'):
        pass
    return env, workers, times


def run_case(root, work, settings, mode, cutin_time, suffix=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith('KMPC_')}
    env.update({k: v for k, v in settings.items() if not k.startswith('KMPC_BATCH_')})
    suffix = suffix or '_' + mode + '_cutin_' + str(cutin_time).replace('.', 'p')
    case_work = work / suffix.lstrip('_')
    case_work.mkdir()
    env.update(KMPC_ABLATION_MODE=mode, KMPC_ABLATION_WORK_DIR=str(case_work),
               KMPC_RESULT_ROOT=str(root), KMPC_RUN_SUFFIX=suffix, KMPC_CUTIN_TIME=str(cutin_time))
    slack = settings.get('KMPC_BATCH_SLACK_' + mode.upper(), '0')
    env.update(KMPC_ENABLE_SLACK=slack, KMPC_SAVE_SLACK=slack)
    print(f'START {mode}, cutin={cutin_time} s', flush=True)
    log = case_work / 'run.log'
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, str(BASE / 'Koopman_MPC_IDM_learning_set.py')],
                                cwd=BASE, env=env, stdout=stream, stderr=subprocess.STDOUT)
    metrics = root / 'cutin' / ('Distributed_MPC' + suffix) / 'metrics.csv'
    if not metrics.exists():
        raise RuntimeError(f'{mode}, cutin={cutin_time}: exit={result.returncode}\n'
                           + log.read_text(encoding='utf-8', errors='replace')[-5000:])
    with metrics.open(encoding='utf-8-sig', newline='') as stream:
        data = {row['metric']: row['value'] for row in csv.DictReader(stream)}
    # Recorded optimization failures are experimental outcomes; other failures stop the batch.
    expected = ('infeasible', 'infeasible_inaccurate', 'user_limit', 'solver_error')
    if data.get('completed') != '1' and data.get('failure_status') not in expected:
        raise RuntimeError(f'Unexpected failure: {mode}, cutin={cutin_time}\n'
                           + log.read_text(encoding='utf-8', errors='replace')[-5000:])
    print(f'END {mode}, cutin={cutin_time}: completed={data.get("completed")}, '
          f'last_step={data.get("last_valid_step")}, status={data.get("failure_status")}', flush=True)


def save_batch_settings(root, settings, workers, times):
    """Save resolved settings and exact launcher text before starting any case."""
    lines = [f'timestamp: {datetime.now().isoformat(timespec="seconds")}',
             f'python: {sys.executable}', f'result_root: {root}',
             f'modes: {", ".join(MODES)}', f'cutin_times_s: {times}',
             f'workers: {workers}', f'timing_valid: {settings.get("KMPC_TIMING_RUN", "0") == "1"}',
             'time_step_s: 0.12 (learned model)', 'planned_final_sample_s: 79.92',
             'PV_initial_speed_mps: 20', 'PV_acceleration_mps2: 2',
             'PV_acceleration_duration_s: 3', 'PV_peak_speed_mps: 26',
             'added_learning_prefix_steps: 0', '', '[Resolved environment]']
    lines += [f'{k}={v}' for k, v in sorted(settings.items())]
    lines += ['', '[BAT source at launch]', Path(settings['KMPC_BATCH_LAUNCHER']).read_text(encoding='utf-8-sig')]
    (root / 'run_settings.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_serial_timing(root, work, settings, cutin_time):
    """Benchmark complete controllers one simulation at a time, without coverage LPs."""
    timing = dict(settings, KMPC_TIMING_RUN='1', KMPC_BATCH_WORKERS='1')
    timing_root = root / 'serial_timing'
    timing_root.mkdir()
    save_batch_settings(timing_root, timing, 1, [cutin_time])
    with (timing_root / 'run_settings.txt').open('a', encoding='utf-8') as stream:
        stream.write('\nTimingValid: True for this dedicated serial stage; coverage LPs disabled.\n'
                     'controller: CAV encoding through solved control, excluding plant simulation\n'
                     'fleet_sequential: sum of CAV controller times; not a parallel deployment estimate\n')
    for mode in MODES:
        active = dict(timing)
        from summarize_uncertainty_ablation import read_cases
        hard = next(r for r in read_cases(root/'hard_comparison')
                    if r['mode'] == mode and r['cutin_time_s'] == cutin_time)
        active['KMPC_BATCH_SLACK_'+mode.upper()] = '0' if hard['completed'] == 1 else '1'
        run_case(timing_root, work, active, mode, cutin_time)


def run_review_evidence(settings, workers, times, include_timing=True, hard_only=False):
    """Run a self-contained hard/soft comparison and dedicated serial timing stage."""
    if not hard_only and any(settings.get('KMPC_BATCH_SLACK_' + m.upper()) != '1' for m in MODES):
        raise ValueError('Review evidence requires the same slack policy for all four methods.')
    prefix = settings['KMPC_BATCH_RESULT_PREFIX']
    root = BASE / prefix
    number = 2
    while True:
        try:
            root.mkdir()
            break
        except FileExistsError:
            root = BASE / f'{prefix}_{number}'
            number += 1
    save_batch_settings(root, settings, workers, times)
    with (root / 'run_settings.txt').open('a', encoding='utf-8') as stream:
        stream.write('\n[Stages]\nperformance: fresh hard and soft comparisons; no reused baselines\n'
                     f'serial_timing: four methods; cutin={times[0]} s; one process at a time\n'
                     f'OS: {platform.platform()}\nCPU: {os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())}\n')
        for package in ('numpy', 'torch', 'cvxpy', 'osqp', 'gurobipy'):
            stream.write(f'{package}: {version(package)}\n')
    print('RESULT_ROOT ' + str(root), flush=True)
    with tempfile.TemporaryDirectory(prefix='.cutin_evidence_work_', dir=BASE) as temporary:
        work = Path(temporary)
        stages = [('hard_comparison', dict(settings, **{'KMPC_BATCH_SLACK_' + m.upper(): '0' for m in MODES}))]
        if not hard_only:
            stages.append(('soft_comparison', settings))
        for stage, stage_settings in stages:
            stage_root = root / stage
            stage_work = work / stage
            stage_work.mkdir()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = [(mode, t) for t in times for mode in MODES]
                if stage == 'soft_comparison':
                    from summarize_uncertainty_ablation import read_cases
                    failed = [r for r in read_cases(root/'hard_comparison') if r['completed'] != 1]
                    for case in failed:
                        attempts = __import__('pandas').read_csv(case['folder']/'solve_attempts.csv', keep_default_na=False)
                        last = attempts.iloc[-1]
                        if not (last['status'] in ('infeasible', 'infeasible_inaccurate') or
                                str(last.get('independent_feasibility_status', '')) in ('infeasible', 'infeasible_inaccurate')):
                            raise RuntimeError('Hard case stopped without confirmed infeasibility: '+str(case['folder']))
                    jobs = [(r['mode'], r['cutin_time_s']) for r in failed]
                futures = [pool.submit(run_case, stage_root, stage_work, stage_settings, mode, t)
                           for mode, t in jobs]
                for future in as_completed(futures):
                    future.result()
            save_batch_settings(stage_root, stage_settings, workers, times)
        if include_timing:
            run_serial_timing(root, work, settings, times[0])
    if hard_only:
        from summarize_uncertainty_ablation import summarize_hard_evidence
        summarize_hard_evidence(root)
    else:
        from summarize_uncertainty_ablation import summarize_review_evidence
        summarize_review_evidence(root)
    print(('REVIEW EVIDENCE FINISHED ' if include_timing else 'PERFORMANCE FINISHED ') + str(root), flush=True)


def main():
    args = sys.argv[1:]
    if args not in ([], ['--check-config'], ['--review-evidence'], ['--review-performance'], ['--review-hard']) and not (
            len(args) == 2 and args[0] == '--serial-timing'):
        raise ValueError('Use the BAT launcher: --check-config, --review-evidence, '
                         '--review-performance, --review-hard, or --serial-timing RESULT_ROOT.')
    settings, workers, times = configuration()
    print(f'CUTIN TIMES {times}; methods={MODES}; workers={workers}', flush=True)
    if args == ['--check-config']:
        print('Configuration valid; no simulation or result folder created.', flush=True)
        return
    if args in ([], ['--review-evidence'], ['--review-performance'], ['--review-hard']):
        run_review_evidence(settings, workers, times, include_timing=args in ([], ['--review-evidence']),
                            hard_only=args == ['--review-hard'])
        return
    if len(args) == 2 and args[0] == '--serial-timing':
        root = Path(args[1]).resolve()
        if root.parent != BASE or not list((root / 'hard_comparison').glob('cutin/*/metrics.csv')):
            raise ValueError('Select a completed review-performance folder in IDM_multistep.')
        previous = {}
        for line in (root / 'run_settings.txt').read_text(encoding='utf-8').splitlines():
            if line == '[BAT source at launch]':
                break
            if line.startswith('KMPC_') and '=' in line:
                key, value = line.split('=', 1)
                previous[key] = value
        for key, value in settings.items():
            if key.startswith('KMPC_') and not key.startswith('KMPC_BATCH_'):
                if previous.get(key) != value:
                    raise ValueError('Timing/performance configuration mismatch: ' + key)
        if (root / 'serial_timing').exists():
            raise ValueError('Dedicated timing already exists; existing results are never overwritten.')
        with tempfile.TemporaryDirectory(prefix='.cutin_timing_work_', dir=BASE) as temporary:
            run_serial_timing(root, Path(temporary), settings, times[0])
        from summarize_uncertainty_ablation import summarize_review_evidence
        summarize_review_evidence(root)
        print('REVIEW EVIDENCE FINISHED ' + str(root), flush=True)
        return


if __name__ == '__main__':
    main()
