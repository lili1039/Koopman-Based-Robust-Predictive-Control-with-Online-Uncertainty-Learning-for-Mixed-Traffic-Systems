"""Rebuild only the uncertainty-treatment comparison table from saved case records."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODES = ('nominal', 'global', 'frozen', 'online')
CASE_FIELDS = ('mode', 'scenario', 'completed', 'step_seconds', 'history_steps',
               'last_valid_step', 'cutin_time_s', 'SlackEnabled', 'TimingValid',
               'RMSVE', 'RMSSE', 'CAV_spacing_violation_count',
               'CAV_velocity_violation_count', 'slack_triggered_solves',
               'failure_status', 'failure_reason')
COVERAGE_FIELDS = ('step', 'subplatoon', 'horizon', 'covered_at_prediction_origin')
TIGHTENING_FIELDS = ('step', 'subplatoon', 'horizon', 'spacing_lower', 'spacing_upper')
ATTEMPT_FIELDS = ('step', 'subplatoon', 'status', 'success', 'error',
                  'independent_feasibility_status', 'independent_feasibility_error')


def read_cases(root):
    cases = []
    for path in sorted(Path(root).glob('cutin/*/metrics.csv')):
        frame = pd.read_csv(path, keep_default_na=False)
        values = dict(zip(frame.metric, frame.value))
        row = {'folder': path.parent}
        for key in CASE_FIELDS:
            if key not in values:
                continue
            try:
                row[key] = float(values[key])
            except (ValueError, TypeError):
                row[key] = values[key]
        cases.append(row)
    return cases


def select_completed_cases(hard_cases, soft_cases):
    """Always use successful hard cases or their complete matching soft reruns."""
    hard_times = [r['cutin_time_s'] for r in hard_cases]
    soft_times = [r['cutin_time_s'] for r in soft_cases]
    if len(set(hard_times)) != len(hard_times) or len(set(soft_times)) != len(soft_times):
        raise ValueError('Duplicate cut-in case in the hard or soft stage.')
    if set(soft_times) - set(hard_times):
        raise ValueError('A soft rerun must have a matching hard-constraint case.')
    replacements = {r['cutin_time_s']: r for r in soft_cases}
    selected = []
    for case in hard_cases:
        replacement = replacements.get(case['cutin_time_s'])
        if case['completed'] == 1:
            if replacement is not None:
                raise ValueError('A hard-feasible case must not be rerun with slack.')
            selected.append(case)
        else:
            if replacement is None or replacement['completed'] != 1:
                raise ValueError('An incomplete hard case requires a complete matching soft rerun.')
            selected.append(replacement)
    return selected


def summarize_review_evidence(root):
    """One row per method, using the existing table definitions and rounding scope."""
    root = Path(root)
    hard = read_cases(root/'hard_comparison')
    soft = read_cases(root/'soft_comparison')
    timing = read_cases(root/'serial_timing')
    records = []
    for mode in MODES:
        hard_cases = [r for r in hard if r['mode'] == mode]
        soft_cases = [r for r in soft if r['mode'] == mode]
        if not hard_cases and not soft_cases:
            continue
        selected = select_completed_cases(hard_cases, soft_cases)
        row = dict(mode=mode, cases=len(hard_cases or soft_cases),
                   hard_constraint_completed_cases=int(sum(r['completed'] for r in hard_cases)))
        if selected:
            row.update(total_slack_triggered_solves=int(sum(r['slack_triggered_solves'] for r in selected)),
                       total_CAV_spacing_violation_count=int(sum(r['CAV_spacing_violation_count'] for r in selected)),
                       total_CAV_velocity_violation_count=int(sum(r['CAV_velocity_violation_count'] for r in selected)),
                       mean_RMSVE=float(np.mean([r['RMSVE'] for r in selected])),
                       mean_RMSSE=float(np.mean([r['RMSSE'] for r in selected])))
            covered = samples = phase_covered = phase_samples = 0
            tightening = []
            for case in selected:
                folder = case['folder']
                margins = pd.read_csv(folder/'tightening.csv', usecols=['spacing_lower', 'spacing_upper'])
                tightening.append((margins.spacing_lower + margins.spacing_upper).mean())
                if mode != 'nominal':
                    errors = pd.read_csv(folder/'coverage.csv', usecols=list(COVERAGE_FIELDS))
                    observed = errors.covered_at_prediction_origin
                    covered += observed.sum()
                    samples += observed.notna().sum()
                    event = round(case['cutin_time_s']/case['step_seconds'])
                    origin = errors.step - errors.horizon
                    phase = observed[(origin >= event) &
                                     (origin <= event + round(15/case['step_seconds']))]
                    phase_covered += phase.sum()
                    phase_samples += phase.notna().sum()
            row.update(full_origin_coverage=float(covered/samples) if samples else np.nan,
                       cutin_15s_origin_coverage=float(phase_covered/phase_samples) if phase_samples else np.nan,
                       full_mean_spacing_tightening_m=float(np.mean(tightening)))
        timed = [r for r in timing if r['mode'] == mode]
        if timed:
            if len(timed) != 1 or timed[0]['TimingValid'] != 1 or timed[0]['completed'] != 1:
                raise ValueError('Expected one completed dedicated timing case per method')
            values = np.loadtxt(timed[0]['folder']/'controller_computation_time.csv', delimiter=',')
            row['controller_mean_ms'] = float(values.mean()*1000)
        records.append(row)
    result = pd.DataFrame(records)
    result.to_csv(root/'reviewer_summary.csv', index=False, encoding='utf-8-sig')
    return result


def summarize_hard_evidence(root):
    """Report hard-stage completion before any soft reruns exist."""
    root = Path(root)
    hard = read_cases(root/'hard_comparison')
    records = []
    for mode in MODES:
        cases = [r for r in hard if r['mode'] == mode]
        if cases:
            records.append(dict(mode=mode, cases=len(cases),
                hard_constraint_completed_cases=int(sum(r['completed'] for r in cases))))
    result = pd.DataFrame(records)
    result.to_csv(root/'reviewer_summary.csv', index=False, encoding='utf-8-sig')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root.resolve()
    if (root/'hard_comparison').exists() or (root/'soft_comparison').exists():
        print(summarize_review_evidence(root).to_string(index=False))
    else:
        # A standalone stage already stores the required fields in each metrics.csv.
        print(pd.DataFrame([{k:v for k,v in row.items() if k != 'folder'}
                            for row in read_cases(root)]).to_string(index=False))


if __name__ == '__main__':
    main()
