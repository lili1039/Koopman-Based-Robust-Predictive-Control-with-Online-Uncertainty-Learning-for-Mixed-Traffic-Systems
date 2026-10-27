"""Fresh repeated cut-in robustness evaluation. Launch run_repeated_safety.bat."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
METHODS = {'all_hdvs': ('All HDVs', 'nominal'),
           'nominal': ('Distributed MPC', 'nominal'),
           'distributed': ('Distributed MPC', 'online')}
REPORT_METHODS = tuple(METHODS)
RANGES = dict(Tgap=(0.8, 2.0), v0_IDM=(35., 45.), veh_len=(4., 5.5),
              Tgap_rate=(-0.05, 0.05), v0_rate=(-0.02, 0.02),
              a_IDM=(1., 2.5), b_IDM=(3., 5.), s0_IDM=(5., 8.))
PARAMETER_SEEDS = [2026102001, 2026102002, *range(2026102301, 2026102314)]
BRAKES = [3., 3.5, 4., 4.5]
TIMES = [10.08, 11.28, 12.48, 13.68, 14.88, 16.08]
THREAD_KEYS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES')
CODE_FILES = ('run_repeated_safety.py', 'run_repeated_safety.bat',
              'Koopman_MPC_IDM_learning_set.py',
              '_heter_helperfuns_torch_IDM_learning_set.py', 'uncertainty_ablation.py')
PROTOCOL = ('Complete all controlled hard-constraint cases before any soft rerun. '
            'Only independently confirmed infeasible cases are rerun from the initial state '
            'with common slack penalties. Select each successful hard trajectory or its '
            'complete soft rerun. Simulate all-HDVs baselines afresh.')
SCHEMA = 'repeated_cutin_robustness_1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data, allow_nan=False):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str, allow_nan=allow_nan), encoding='utf-8')
    temporary.replace(path)


def configuration():
    settings = {k: v for k, v in os.environ.items()
                if k.startswith('KMPC_') or k in THREAD_KEYS or k == 'MPLBACKEND'}
    workers = int(settings['KMPC_REPEAT_WORKERS'])
    seeds = [int(x) for x in settings['KMPC_REPEAT_PARAMETER_SEEDS'].split(',')]
    brakes = [float(x) for x in settings['KMPC_REPEAT_BRAKES'].split(',')]
    times = [round((int(settings['KMPC_REPEAT_START_STEP']) + i *
                   int(settings['KMPC_REPEAT_INTERVAL_STEPS']))*.12, 2)
             for i in range(int(settings['KMPC_REPEAT_TIMING_COUNT']))]
    if seeds != PARAMETER_SEEDS or brakes != BRAKES or times != TIMES:
        raise ValueError('Keep the 15 parameter seeds, four braking intensities and six timings in the BAT.')
    fixed = {'KMPC_SCENARIO': 'cutin', 'KMPC_ENABLE_SLACK': '0', 'KMPC_SAVE_SLACK': '0',
             'KMPC_HARD_FIRST_SLACK_FALLBACK': '0', 'KMPC_TIME_VARYING': '1',
             'KMPC_DISTURBANCE_ESTIMATE': '1', 'KMPC_SIGNAL_DELAY': '0',
             'KMPC_CAV_PRED_LOSS': '0', 'KMPC_SAVE_PLOTS': '0',
             'KMPC_SUBPLATOON': '2,3,1,4,2', 'KMPC_SIM_MAXTIME': '0',
             'KMPC_REPEAT_EVALUATION_END_STEP': '333', 'KMPC_REPEAT_CI_SEED': '20261023',
             'KMPC_REPEAT_CI_RESAMPLES': '10000', 'KMPC_SIM_SEED': '20261020'}
    if any(settings.get(k) != v for k, v in fixed.items()) or not 1 <= workers <= 8:
        raise ValueError('Keep the fixed experimental protocol in the BAT; use 1..8 workers.')
    if float(settings['KMPC_CUTIN_FINAL_SPEED']) != 4 or float(settings['KMPC_CUTIN_GAP']) != 25.5:
        raise ValueError('Keep final speed 4 m/s and post-cut-in gap 25.5 m.')
    return settings, workers, seeds, brakes, times


def input_checksums():
    """Enumerate every required model, predictor matrix and uncertainty-set CSV."""
    helper = ast.parse((BASE/'_heter_helperfuns_torch_IDM_learning_set.py').read_text(encoding='utf-8-sig'))
    models = [node.value.value for node in helper.body
              if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
              and any(isinstance(t, ast.Name) and t.id.startswith('HDV_')
                      and t.id.endswith('_MODEL_PATH') for t in node.targets)]
    if len(models) != 4:
        raise ValueError('Expected the four heterogeneous multi-step model bundles.')
    inputs = set()
    for name in models:
        model_root = BASE.parent/'Deep_Koop_v1'/'model_highd_multistep'
        inputs.add(model_root/name)
        date = name.split('/')[0]
        weights = model_root/date/'multistep_weights'
        zono = BASE.parent/'Zonosets_and_preparations'/'_5_multi_step'/'error_points_multistep'/date/'zonotopes'
        inputs.add(weights/'decoder_weight.csv')
        inputs.update(zono/f'Zono_gamma_{part}.csv' for part in ('c', 'G'))
        for step in range(1, 16):
            inputs.add(weights/f'G0_step{step:02d}.csv')
            for lag in range(step):
                inputs.update(weights/f'{kind}_step{step:02d}_lag{lag:02d}.csv' for kind in ('Gu', 'Ge'))
            inputs.update(zono/f'Zono_omega_step{step:02d}_{part}.csv' for part in ('c', 'G'))
    for path in sorted(inputs):
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError('Required simulation input missing or empty: '+str(path))
    return {str(p.relative_to(BASE.parent)): digest(p) for p in sorted(inputs)}


def prepare(root, settings, seeds, brakes, times, inputs):
    params = root/'parameters'
    params.mkdir()
    composition = [int(x) for x in settings['KMPC_SUBPLATOON'].split(',')]
    count = sum(composition) + len(composition)
    profiles = []
    for profile, seed in enumerate(seeds, 1):
        rng = np.random.default_rng(seed)
        data = {key: np.round(rng.uniform(low, high, count), 4)
                for key, (low, high) in RANGES.items()}
        path = params/f'profile_{profile}.csv'
        pd.DataFrame(data).to_csv(path, index=False, float_format='%.4f')
        profiles.append(dict(profile=profile, seed=seed, source='numpy PCG64; uniform; round 4 dp',
                             file=path.relative_to(root).as_posix(), sha256=digest(path)))
    cases = [dict(case_id=f'p{p}_b{b:g}_t{round(t/.12):03d}_{method}'.replace('.', 'p'),
                  profile=p, parameter_seed=seed, brake_mps2=b, cutin_time_s=t,
                  brake_duration_s=22./b, method=method, simulation_seed=int(settings['KMPC_SIM_SEED']))
             for p, seed in enumerate(seeds, 1) for b in brakes for t in times for method in REPORT_METHODS]
    pd.DataFrame(cases).to_csv(root/'manifest.csv', index=False)
    metadata = dict(schema=SCHEMA, created=datetime.now().astimezone().isoformat(),
                    protocol=PROTOCOL, settings=settings, parameter_ranges=RANGES, profiles=profiles,
                    methods=list(REPORT_METHODS), braking_magnitudes_mps2=brakes, cutin_times_s=times,
                    cases=len(cases), python=sys.version, platform=platform.platform(),
                    packages={k: version(k) for k in ('numpy','pandas','scipy','torch','cvxpy','osqp','gurobipy')},
                    code_sha256={name: digest(BASE/name) for name in CODE_FILES}, input_sha256=inputs,
                    manifest_sha256=digest(root/'manifest.csv'),
                    confidence_intervals=dict(level=.95, method='percentile cluster bootstrap',
                        resamples=10000, seed=20261023, independent_units=15))
    dump(root/'run_settings.json', metadata)
    (root/'run_settings.txt').write_text(
        '\n'.join(f'{k}={v}' for k,v in sorted(settings.items()))+'\n\n[BAT at launch]\n'+
        (BASE/'run_repeated_safety.bat').read_text(encoding='utf-8-sig'), encoding='utf-8')
    (root/'README.md').write_text('# Repeated cut-in robustness evaluation\n\n'+PROTOCOL+'\n\n'
        '15 random IDM profiles x four braking magnitudes x six timings x three methods = 1080 final cases. '
        'Evaluate 0--39.96 s after the 50-step initialization. Each final case uses its successful hard '
        'trajectory or its complete soft rerun. All-HDVs results are generated in this batch.\n\n'
        'case_sources.csv records the selected stage; case_metrics.csv contains the per-case metrics; '
        'reviewer_summary.csv contains the 90-run comparisons and 95% profile-cluster bootstrap intervals. '
        'Spacing violation duration is averaged only over runs with spacing violations. '
        'Slack activation counts successful control solves with slack above 0.001.\n', encoding='utf-8')
    return cases


def verify_saved(root):
    saved = json.loads((root/'run_settings.json').read_text())
    if saved.get('schema') != SCHEMA:
        raise ValueError('Resume requires a batch created by this standalone runner.')
    checks = [(BASE/name, checksum) for name, checksum in saved['code_sha256'].items()]
    checks += [(BASE.parent/name, checksum) for name, checksum in saved['input_sha256'].items()]
    checks += [(root/item['file'], item['sha256']) for item in saved['profiles']]
    checks.append((root/'manifest.csv', saved['manifest_sha256']))
    for path, checksum in checks:
        if digest(path) != checksum:
            raise ValueError('Batch input changed since preparation: '+str(path))
    return saved


def confirmed_infeasible(folder):
    """A numerical stop or an independently feasible QP never qualifies for a soft rerun."""
    attempts = pd.read_csv(folder/'solve_attempts.csv', keep_default_na=False)
    if attempts.empty:
        return False
    last = attempts.iloc[-1]
    return (str(last.get('success')) in ('0', '0.0') and
            str(last.get('independent_feasibility_status', '')) in ('infeasible', 'infeasible_inaccurate'))


def run_case(root, work, settings, case):
    work.mkdir(parents=True, exist_ok=True)
    case_lock = work/(case['case_id']+'.lock')
    # Probe before reading or moving files that an orphaned child may still be writing.
    with file_lock(case_lock):
        pass
    control, mode = METHODS[case['method']]
    suffix = '_'+case['case_id']
    output = root/'cutin'/(control.replace(' ','_')+suffix)
    final = root/'cases'/case['case_id']
    if (final/'case_metrics.json').exists():
        row = measure(final, case, settings)
        if row['completed'] or (settings['KMPC_ENABLE_SLACK'] == '0' and confirmed_infeasible(final)):
            return row
        raise RuntimeError('Saved numerical stop or incomplete soft run requires inspection: '+str(final))
    for partial in (output, final):
        if partial.exists():
            diagnostics = root/'interrupted_cases'
            diagnostics.mkdir(exist_ok=True)
            partial.rename(diagnostics/(partial.name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    env = {k:v for k,v in os.environ.items() if not k.startswith('KMPC_')}
    env.update({k:v for k,v in settings.items() if not k.startswith('KMPC_REPEAT_')})
    # All jobs share a parent so the simulator's process slots also cover surviving children.
    job = work/(case['case_id']+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    job.mkdir(parents=True)
    env.update(KMPC_REPEATED_RUN='1', KMPC_CASE_LOCK=str(case_lock.resolve()), KMPC_ABLATION_MODE=mode,
               KMPC_ABLATION_WORK_DIR=str(job), KMPC_RESULT_ROOT=str(root),
               KMPC_RUN_SUFFIX=suffix, KMPC_CONTROL=control,
               KMPC_VEHICLE_CSV=str(root/'parameters'/f'profile_{case["profile"]}.csv'),
               KMPC_CUTIN_TIME=str(case['cutin_time_s']),
               KMPC_CUTIN_BRAKE_DURATION=str(case['brake_duration_s']),
               KMPC_SIM_MAXTIME='0', KMPC_SIM_SEED=str(case['simulation_seed']))
    log = job/'run.log'
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-B', str(BASE/'Koopman_MPC_IDM_learning_set.py')],
                                cwd=BASE, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if not (output/'metrics.csv').exists():
        raise RuntimeError(f'{case["case_id"]}, exit={result.returncode}:\n'+log.read_text(errors='replace')[-4000:])
    row = measure(output, case, settings)
    if result.returncode and not row['failure_status']:
        raise RuntimeError(f'{case["case_id"]}: unexpected exit {result.returncode}')
    final.parent.mkdir(exist_ok=True)
    output.rename(final)
    # An early hard stop can have no closing pair yet (infinite TTC); retain its diagnostics.
    dump(final/'case_metrics.json', row, allow_nan=True)
    (final/'metrics.csv').unlink()  # The resolved case_metrics.json supersedes the worker status file.
    return row


def safety(states, lengths, indices, settings):
    """Only the safety quantities used in the reviewer comparison table."""
    spacing = states[:, :-1, 0] - states[:, 1:, 0]
    net = spacing - lengths
    closing = states[:, 1:, 1] - states[:, :-1, 1]
    ttc = np.full_like(net, np.inf)
    np.divide(net, closing, out=ttc, where=closing > 1e-6)
    ttc[net <= 0] = 0
    gaps = spacing[:, indices-1]
    velocity = states[:, indices, 1]
    acceleration = states[:-1, indices, 2]
    violation = False
    for values, lower, upper in [
        (gaps, 'KMPC_SPA_MIN', 'KMPC_SPA_MAX'),
        (velocity, 'KMPC_VEL_MIN', 'KMPC_VEL_MAX'),
        (acceleration, 'KMPC_ACC_MIN', 'KMPC_ACC_MAX')]:
        violation |= bool(((values < float(settings[lower])-1e-3) |
                           (values > float(settings[upper])+1e-3)).any())
    return dict(focus_min_spacing_m=float(gaps.min()),
                focus_min_TTC_s=float(ttc[:, indices-1].min()),
                all_collision_trial=int((net <= 0).any()),
                focus_constraint_violation_trial=int(violation))


def measure(folder, case, settings):
    """Read the common-window trajectory and the solver/slack evidence."""
    folder = Path(folder)
    if (folder/'metrics.csv').exists():
        recorded = pd.read_csv(folder/'metrics.csv', keep_default_na=False)
        status = dict(zip(recorded.metric, recorded.value))
    else:
        status = json.loads((folder/'case_metrics.json').read_text())
    with np.load(folder/'trajectory.npz') as data:
        states = data['states'][int(data['history_steps']):]
        dt = float(data['dt'])
        lengths, indices = data['vehicle_lengths'], data['nominal_cav_indices']
    end = int(settings.get('KMPC_REPEAT_EVALUATION_END_STEP', 333))
    states = states[:end+1]
    if not len(states) or not np.isfinite(states).all() or not np.isclose(dt, .12):
        raise ValueError('Nonfinite trajectory: '+case['case_id'])
    completed = int(len(states)-1 >= end)
    row = dict(case, completed=completed, last_valid_step=len(states)-1,
               planned_steps=end, evaluation_seconds=(len(states)-1)*dt,
               failure_status='' if completed else status.get('failure_status', ''),
               failure_reason='' if completed else status.get('failure_reason', ''))
    row.update(safety(states, lengths, indices, settings))
    gaps = states[:, indices-1, 0] - states[:, indices, 0]
    lower, upper = float(settings['KMPC_SPA_MIN']), float(settings['KMPC_SPA_MAX'])
    bad = ((gaps < lower-1e-3) | (gaps > upper+1e-3)).any(axis=1)
    row.update(max_spacing_violation_m=float(np.maximum(np.maximum(lower-gaps, gaps-upper), 0).max()),
               spacing_violation_duration_s=float(bad[:-1].sum()*dt))
    attempts = pd.read_csv(folder/'solve_attempts.csv')
    attempts = attempts[attempts.step < len(states)-1]
    slack = pd.read_csv(folder/'slack_records.csv')
    slack = slack[(slack.step < len(states)-1) &
                  (slack.slack > float(settings['KMPC_SLACK_TOL']))]
    row.update(solve_attempts=len(attempts), solve_successes=int(attempts.success.sum()),
               slack_triggered_solves=len(slack[['step', 'subplatoon']].drop_duplicates()))
    return row


def summarize(root, rows, settings):
    frame = pd.DataFrame(rows).sort_values(['profile','brake_mps2','cutin_time_s','method'])
    if len(frame) != 1080 or frame.case_id.duplicated().any() or not frame.completed.eq(1).all():
        raise RuntimeError('Final summary requires all 1080 selected complete cases.')
    frame.to_csv(root/'case_metrics.csv', index=False)
    draws = np.random.default_rng(int(settings['KMPC_REPEAT_CI_SEED'])).integers(
        0, 15, size=(int(settings['KMPC_REPEAT_CI_RESAMPLES']), 15))
    records = []
    for brake in BRAKES:
        for method in REPORT_METHODS:
            done = frame[(frame.brake_mps2 == brake) & (frame.method == method)]
            counts = done.groupby('profile').size().reindex(range(1,16), fill_value=0)
            if len(done) != 90 or not counts.eq(6).all():
                raise RuntimeError('Final comparison requires 15 profiles and six timings per group.')
            attempts, success = int(done.solve_attempts.sum()), int(done.solve_successes.sum())
            row = dict(method=method, brake_mps2=brake, cases=90, planned_cases=90, completed=90,
                complete_profiles=15, mean_min_spacing_m=float(done.focus_min_spacing_m.mean()),
                mean_min_TTC_s=float(done.focus_min_TTC_s.mean()),
                collision_trial_rate=float(done.all_collision_trial.mean()),
                constraint_violation_trial_rate=float(done.focus_constraint_violation_trial.mean()),
                hard_constraint_completed_cases=int(done.hard_completed.sum()),
                soft_rerun_cases=int(done.selected_stage.eq('soft').sum()),
                solve_success_rate=success/attempts if attempts else np.nan,
                slack_solve_rate=float(done.slack_triggered_solves.sum()/success) if success else np.nan,
                max_spacing_violation_m=float(done.max_spacing_violation_m.max()),
                mean_spacing_violation_duration_s=float(done.loc[
                    done.spacing_violation_duration_s > 0, 'spacing_violation_duration_s'].mean()))
            for metric, label in [('focus_min_spacing_m','spacing'), ('focus_min_TTC_s','TTC')]:
                blocks = done.groupby('profile')[metric].mean().reindex(range(1,16)).to_numpy()
                if not np.isfinite(blocks).all():
                    raise ValueError('Nonfinite spacing/TTC block mean; cannot form a bootstrap interval.')
                row[label+'_bootstrap_p2p5'], row[label+'_bootstrap_p97p5'] = np.quantile(
                    blocks[draws].mean(axis=1), [.025,.975])
            records.append(row)
    result = pd.DataFrame(records)
    result.to_csv(root/'reviewer_summary.csv', index=False)
    return result


def status(root, phase, **extra):
    dump(root/'status.json', dict(phase=phase, pid=os.getpid(),
         updated=datetime.now().astimezone().isoformat(), **extra))


def prepare_stage(root, name, settings, workers):
    stage = root/name
    stage.mkdir(exist_ok=True)
    (stage/'parameters').mkdir(exist_ok=True)
    for profile in range(1,16):
        source = root/'parameters'/f'profile_{profile}.csv'
        target = stage/'parameters'/source.name
        if target.exists() and digest(target) != digest(source):
            raise ValueError('Stage parameter file changed: '+str(target))
        if not target.exists():
            shutil.copyfile(source, target)
    dump(stage/'execution_policy.json', dict(active_process_limit=workers,
         evaluation_end_step=int(settings['KMPC_REPEAT_EVALUATION_END_STEP']), simulation_max_time_s=40.08))
    return stage


def run_stage(root, name, cases, settings, workers, allow_infeasible=False):
    stage = prepare_stage(root, name, settings, workers)
    rows, errors = [], []
    status(root, name, completed=0, total=len(cases))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(run_case, stage, root/'work'/name, settings, case):case for case in cases}
        for future in as_completed(jobs):
            case = jobs[future]
            try:
                row = future.result()
                if not row['completed'] and not (allow_infeasible and
                        confirmed_infeasible(stage/'cases'/case['case_id'])):
                    raise RuntimeError('Unconfirmed numerical/solver stop or incomplete run: '+case['case_id'])
                rows.append(row)
                print(f'{name} {len(rows)}/{len(cases)} {case["case_id"]} completed={row["completed"]}', flush=True)
            except Exception as exc:
                errors.append(f'{case["case_id"]}: {exc}')
                print('ERROR '+errors[-1], flush=True)
            status(root, name, completed=len(rows), total=len(cases), errors=len(errors))
    if errors:
        (stage/'errors.txt').write_text('\n\n'.join(errors), encoding='utf-8')
        raise RuntimeError(f'{len(errors)} cases require inspection; see {stage / "errors.txt"}. No later stage started.')
    if rows:
        pd.DataFrame(rows).to_csv(stage/'case_metrics.csv', index=False)
    return {row['case_id']:row for row in rows}


def copy_selected(source, dest):
    """Publish a complete case directory; rebuild interrupted copies on resume."""
    expected = [p.relative_to(source) for p in source.rglob('*')
                if p.is_file() and p.name != 'case_metrics.json']
    dest.parent.mkdir(exist_ok=True)
    if dest.exists() and all((dest/p).is_file() and digest(dest/p) == digest(source/p) for p in expected):
        return
    staging = dest.with_name(dest.name+'.partial')
    for partial in (dest, staging):
        if partial.exists():
            diagnostics = dest.parent.parent/'interrupted_copies'
            diagnostics.mkdir(exist_ok=True)
            partial.rename(diagnostics/(partial.name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    shutil.copytree(source, staging)
    staging.rename(dest)


def select_results(root, cases, hard, soft, baselines, settings):
    rows, sources = [], []
    for case in cases:
        key = case['case_id']
        if case['method'] == 'all_hdvs':
            name, selected, hard_completed = 'baseline', baselines[key], 1
            label = 'all_hdvs'
        else:
            hard_completed = int(hard[key]['completed'])
            label = 'hard' if hard_completed else 'soft'
            name = 'hard_comparison' if hard_completed else 'soft_reruns'
            selected = hard[key] if hard_completed else soft[key]
        source = root/name/'cases'/key
        dest = root/'cases'/key
        copy_selected(source, dest)
        if selected['completed'] != 1:
            raise RuntimeError('Cannot report an incomplete selected case: '+key)
        row = dict(selected, selected_stage=label, hard_completed=hard_completed)
        dump(dest/'case_metrics.json', row, allow_nan=True)
        rows.append(row)
        sources.append(dict(case_id=key, selected_stage=label, hard_completed=hard_completed,
                            source=source.relative_to(root).as_posix()))
    pd.DataFrame(sources).to_csv(root/'case_sources.csv', index=False)
    return summarize(root, rows, settings)


def run_batch(root, cases, settings, workers):
    controlled = [c for c in cases if c['method'] != 'all_hdvs']
    hard_settings = dict(settings, KMPC_ENABLE_SLACK='0', KMPC_SAVE_SLACK='0', KMPC_HARD_FIRST_SLACK_FALLBACK='0')
    soft_settings = dict(settings, KMPC_ENABLE_SLACK='1', KMPC_SAVE_SLACK='1', KMPC_HARD_FIRST_SLACK_FALLBACK='0')
    hard = run_stage(root, 'hard_comparison', controlled, hard_settings, workers, allow_infeasible=True)
    hard_frame = pd.DataFrame(hard.values())
    hard_frame.groupby(['method','brake_mps2']).agg(
        cases=('case_id','size'), completed_without_slack=('completed','sum')).reset_index().to_csv(
            root/'hard_comparison'/'completion_summary.csv', index=False)
    baselines = run_stage(root, 'baseline', [c for c in cases if c['method'] == 'all_hdvs'], hard_settings, workers)
    failed = [c for c in controlled if not hard[c['case_id']]['completed']]
    soft = run_stage(root, 'soft_reruns', failed, soft_settings, workers)
    select_results(root, cases, hard, soft, baselines, settings)
    status(root, 'complete', final_cases=len(cases), hard_cases=len(controlled),
           hard_completed=sum(r['completed'] for r in hard.values()), soft_reruns=len(failed), baseline_cases=len(baselines))
    print('FINISHED '+str(root), flush=True)


@contextmanager
def file_lock(path):
    # Windows releases the lock when the runner exits or is interrupted.
    import msvcrt
    with path.open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'\0'); lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('A runner or simulation still holds this lock: '+str(path)) from exc
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-config', action='store_true', help='Validate without simulating.')
    parser.add_argument('--resume', type=Path, help='Continue case-level checkpoints from this runner.')
    parser.add_argument('--workers', type=int, choices=range(1,9), help='Parallel case limit (default: BAT setting).')
    args = parser.parse_args()
    settings, workers, seeds, brakes, times = configuration()
    workers = args.workers or workers
    settings['KMPC_REPEAT_WORKERS'] = str(workers)
    if args.resume:
        root = args.resume.resolve()
        with file_lock(root/'runner.lock'):
            saved = verify_saved(root)
            settings = dict(saved['settings'], KMPC_REPEAT_WORKERS=str(workers))
            cases = pd.read_csv(root/'manifest.csv').to_dict('records')
            if not args.check_config:
                print('RESULT_ROOT '+str(root), flush=True)
                try:
                    run_batch(root, cases, settings, workers)
                except BaseException as exc:
                    status(root, 'error', error=repr(exc)); raise
        if args.check_config:
            print('VALID: saved parameters, code and predictor inputs match; no simulations.', flush=True)
        return
    inputs = input_checksums()
    if args.check_config:
        print(f'VALID: 15 profiles x 4 brakes x 6 timings x 3 methods = 1080 final cases; '
              f'720 controlled hard cases, 360 fresh baselines, failed-only soft reruns; '
              f'{len(inputs)} model inputs; workers={workers}. No results created.', flush=True)
        return
    prefix = 'result_cutin_robustness_15profiles'
    root = BASE / prefix
    number = 2
    while True:
        try:
            root.mkdir()
            break
        except FileExistsError:
            root = BASE / f'{prefix}_{number}'
            number += 1
    with file_lock(root/'runner.lock'):
        try:
            cases = prepare(root, settings, seeds, brakes, times, inputs)
            print('RESULT_ROOT '+str(root), flush=True)
            run_batch(root, cases, settings, workers)
        except BaseException as exc:
            status(root, 'error', error=repr(exc)); raise


if __name__ == '__main__':
    main()
