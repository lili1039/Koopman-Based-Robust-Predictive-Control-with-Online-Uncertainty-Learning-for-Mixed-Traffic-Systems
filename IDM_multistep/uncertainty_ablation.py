"""Uncertainty ablation on the original scenario timeline, with standard CSV outputs."""
import atexit
import csv
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np


class UncertaintyAblation:
    MODES = ('nominal', 'global', 'frozen', 'online')

    @classmethod
    def from_environment(cls):
        mode = os.environ.get('KMPC_ABLATION_MODE', '').strip()
        return cls(mode) if mode else None

    def __init__(self, mode):
        if mode not in self.MODES:
            raise ValueError(f'Unknown uncertainty ablation mode: {mode}')
        self.mode = mode
        self.repeated_run = os.environ.get('KMPC_REPEATED_RUN', '0') == '1'
        self.timing_run = os.environ.get('KMPC_TIMING_RUN', '0') == '1'
        self.collect_table = not self.repeated_run and not self.timing_run
        # Repeated-safety and timing runs retain their existing protocols.
        self.hard_first_fallback = (not self.repeated_run and not self.timing_run
            and os.environ.get('KMPC_ENABLE_SLACK', '0') == '0'
            and os.environ.get('KMPC_HARD_FIRST_SLACK_FALLBACK', '1') == '1')
        self.soft_solvers = {}
        self.freeze_step = int(os.environ.get('KMPC_ABLATION_FREEZE_STEP', '50'))
        self.step = -1
        self.finished = False
        self.ns = None
        self.failure = None
        self.attempts = []
        self.coverage = []
        self.origin_sets = {}
        self.tightening = []
        self.frozen_signature = None
        self.metrics = {}
        self.work = Path(os.environ['KMPC_ABLATION_WORK_DIR'])
        self.work.mkdir(parents=True, exist_ok=True)

    @property
    def nominal_active(self):
        return self.mode == 'nominal'

    @property
    def should_update(self):
        return self.mode == 'online' or (self.mode == 'frozen' and self.step < self.freeze_step)

    def start(self, namespace):
        self.ns = namespace
        if not namespace['disturbance_estimate']:
            raise ValueError('Ablation requires the shared nominal forecast path.')
        if self.freeze_step < namespace['Nfut'] + namespace['N_point']:
            raise ValueError('Freeze step must allow sufficient prediction-error samples.')
        self.out = Path(namespace['output_dir'])
        if self.hard_first_fallback:
            with (self.out / 'run_settings.txt').open('a', encoding='utf-8') as stream:
                stream.write('\nhard_first_slack_fallback: True\n')
        # Coverage measurements must not change the optimizer state used by learning.
        self.coverage_solvers = [] if not self.collect_table or self.mode == 'nominal' else [
            [namespace['Whether_in_Zonotope_Solver'](c.shape, G.shape, G)
             for c, G in zip(cs, gs)]
            for cs, gs in zip(namespace['sub_ctemplate'], namespace['sub_Gtemplate'])]
        for p, solver in enumerate(namespace['sub_solver']):
            original = solver.solve

            def wrapped(*args, _original=original, _solver=solver, _p=p, **kwargs):
                row = {'step': self.step, 'subplatoon': _p + 1}
                used_solver = _solver
                try:
                    if self.hard_first_fallback:
                        result, used_solver = self._solve_with_fallback(
                            _original, _solver, _p, row, args, kwargs)
                    else:
                        result = self._solve_with_numerical_retry(_original, _solver, row, args, kwargs)
                    status = str(used_solver.prob.status)
                    ok = status in ('optimal', 'optimal_inaccurate') and result[2] is not None
                    row.update(status=status, success=int(ok))
                    if not ok:
                        raise RuntimeError(f'Control solve failed: {status}')
                except Exception as exc:
                    if row.get('fallback_used') and _p in self.soft_solvers:
                        used_solver = self.soft_solvers[_p]
                    status = str(used_solver.prob.status)
                    if status in ('optimal', 'optimal_inaccurate', 'None'):
                        status = 'solver_error'
                    row.update(status=status, success=0, error=str(exc))
                    # Independently distinguish infeasibility from numerical failure after stopping;
                    # this diagnostic solution is never applied to the vehicles.
                    if status in ('infeasible','user_limit','solver_error','infeasible_inaccurate'):
                        try:
                            import cvxpy as cp
                            check=cp.Problem(cp.Minimize(0),used_solver.prob.constraints)
                            check.solve(solver=cp.GUROBI,verbose=False,Threads=1,DualReductions=0)
                            row['independent_feasibility_status']=str(check.status)
                        except Exception as verification_error:
                            row['independent_feasibility_status']='verification_error'
                            row['independent_feasibility_error']=str(verification_error)
                    self.attempts.append(row)
                    self.failure = dict(step=self.step, subplatoon=_p+1, status=status, error=str(exc))
                    self.failure['independent_feasibility_status']=row.get('independent_feasibility_status','')
                    self.finish(completed=False)
                    raise
                self.attempts.append(row)
                if self.repeated_run:
                    return result
                if self.collect_table:
                    if self.mode != 'nominal':
                        for k in range(1, _solver.Nfut + 1):
                            c, G, _, _ = self.ns['total_disturbance_set'][_p][k-1]
                            self.origin_sets[(_p, self.step+k, k)] = (c.copy(), G.copy())
                    for h in range(1, _solver.Nfut + 1):
                        self.tightening.append((self.step, _p+1, h,
                            float(_solver.Spa_lb_param.value[h]-_solver.SpaMin),
                            float(_solver.SpaMax-_solver.Spa_ub_param.value[h])))
                return result

            solver.solve = wrapped
        old_hook = sys.excepthook

        def exception_hook(kind, value, tb):
            if not self.finished:
                self.failure = dict(step=self.step, status=kind.__name__, error=str(value))
                self.finish(completed=False)
            old_hook(kind, value, tb)

        sys.excepthook = exception_hook
        atexit.register(lambda: self.finish(completed=False) if not self.finished else None)
        self._save_progress()

    def _solve_with_numerical_retry(self, original, solver, row, args, kwargs):
        """Retry the identical optimization problem after a numerical OSQP stop.

        No constraints or penalties are changed, and this is not a slack fallback.
        Infeasibility is still independently diagnosed by the caller.
        """
        import cvxpy as cp
        started = time.perf_counter()
        try:
            result = original(*args, **kwargs)
            if str(solver.prob.status) in ('optimal', 'optimal_inaccurate') and result[2] is not None:
                return result
            if str(solver.prob.status) in ('infeasible', 'infeasible_inaccurate', 'unbounded'):
                return result
        except Exception:
            if str(solver.prob.status) in ('infeasible', 'infeasible_inaccurate', 'unbounded'):
                raise
        row['primary_status'] = str(solver.prob.status)
        row['numerical_retry_solver'] = 'GUROBI'
        solver.prob.solve(solver=cp.GUROBI, verbose=False, warm_start=True,
                          Threads=1, DualReductions=0, FeasibilityTol=1e-8, OptimalityTol=1e-8)
        row['numerical_retry_status'] = str(solver.prob.status)
        slack = (solver._slack_report() if self.ns['enable_slack'] and self.ns['save_slack']
                 and str(solver.prob.status) in ('optimal', 'optimal_inaccurate') else None)
        return (solver.x.value, solver.s.value, solver.u.value,
                time.perf_counter()-started, slack)

    def _solve_with_fallback(self, original, hard_solver, p, row, args, kwargs):
        """Try hard constraints first, then a matching slack problem at the same state."""
        import cvxpy as cp
        try:
            result = original(*args, **kwargs)
            status = str(hard_solver.prob.status)
            row['hard_status'] = status
            if status in ('optimal', 'optimal_inaccurate') and result[2] is not None:
                row['fallback_used'] = 0
                return result, hard_solver
            row['hard_error'] = 'Hard solve returned no applicable control.'
        except Exception as exc:
            row['hard_status'] = str(hard_solver.prob.status)
            row['hard_error'] = str(exc)
        # Keep infeasibility distinct from a numerical solver failure.
        try:
            check = cp.Problem(cp.Minimize(0), hard_solver.prob.constraints)
            check.solve(solver=cp.GUROBI, verbose=False, Threads=1, DualReductions=0)
            row['hard_feasibility_status'] = str(check.status)
        except Exception as exc:
            row['hard_feasibility_status'] = 'verification_error'
            row['hard_feasibility_error'] = str(exc)
        row['fallback_used'] = 1
        n = self.ns
        previous_slack, previous_save = n['enable_slack'], n['save_slack']
        try:
            # Constructors/solvers use these globals; restore them before the next hard attempt.
            n['enable_slack'] = n['save_slack'] = True
            if p not in self.soft_solvers:
                self.soft_solvers[p] = type(hard_solver)(
                    n['sub_G0'][p], n['sub_Gu'][p], n['sub_Ge'][p], n['sub_D'][p],
                    n['SpaMax'], n['SpaMin'], n['VelMax'], n['VelMin'],
                    n['AccMin'], n['AccMax'], n['sub_nhigh'][p], n['sub_nlow'][p],
                    n['Nfut'], n['Tstep'], n['scenario'])
            soft_solver = self.soft_solvers[p]
            result = soft_solver.solve(*args, **kwargs)
            row['fallback_status'] = str(soft_solver.prob.status)
            if row['fallback_status'] not in ('optimal', 'optimal_inaccurate') or result[2] is None:
                raise RuntimeError('Fallback solve failed: ' + row['fallback_status'])
            n['_record_slack'](p, self.step, result[4])
            return result, soft_solver
        finally:
            n['enable_slack'], n['save_slack'] = previous_slack, previous_save

    def before_step(self, step):
        self.step = step
        n = self.ns
        event = int(round(n.get('cutin_time', 10.08) / n['Tstep']))
        if n['scenario'] == 'cutin' and event <= step <= event + int(n['ep_window']):
            acceleration, low, high = n['helper'].lv_acceleration_estimate(
                n['S'], step, 0, n['scenario'], n['Tstep'], n['Npst'])
            row = dict(step=step, time_s=step*n['Tstep'], nominal_acceleration=acceleration,
                       acceleration_min=low, acceleration_max=high,
                       post_cutin_samples=min(max(1, step-event), int(n['ep_window'])))
            path = self.out / 'lv_prediction.csv'
            first = not path.exists()
            with path.open('a', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if first:
                    writer.writeheader()
                writer.writerow(row)
        if self.mode == 'frozen' and step == self.freeze_step:
            self.frozen_signature = self._set_signature()
        if self.mode == 'frozen' and step >= self.freeze_step:
            assert self._set_signature() == self.frozen_signature, 'Frozen set was modified'
        if step % 20 == 0:
            self._save_progress()

    def _set_signature(self):
        h = hashlib.sha256()
        for sets in self.ns['total_disturbance_set']:
            for item in sets:
                for value in item:
                    h.update(np.asarray(value).tobytes())
        return h.hexdigest()

    def observe_error(self, p, k, error):
        if not self.collect_table or self.mode == 'nominal':
            return
        origin = self.origin_sets.pop((p, self.step, k), None)
        covered = None
        if origin is not None:
            inside, _ = self.coverage_solvers[p][k-1].solve(error, *origin)
            covered = int(inside)
        self.coverage.append((self.step, p+1, k, covered))

    def _save_progress(self):
        self.write_csv(self.work/'progress.csv', [dict(mode=self.mode,step=self.step,
                       planned_steps=self.ns['total_time_steps']-1,attempts=len(self.attempts))],
                       ['mode','step','planned_steps','attempts'])

    @staticmethod
    def write_csv(path, rows, fields):
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)

    def finish(self, completed):
        if self.finished:
            return
        self.finished = True
        n = self.ns
        end = n['total_time_steps']-1 if completed else max(self.step,0)
        states = n['S'][n['Npst']:n['Npst']+end+1].copy()
        if self.repeated_run:
            # The repeated experiment needs trajectories and solver/slack evidence only.
            # Keep its output compact; the batch runner computes all safety statistics.
            np.savez_compressed(self.out/'trajectory.npz',
                states=n['S'][:n['Npst']+end+1], history_steps=n['Npst'],
                dt=n['Tstep'], vehicle_lengths=n['veh_len'],
                nominal_cav_indices=np.asarray(n['sub_start'])+1)
            self.write_csv(self.out/'solve_attempts.csv', self.attempts,
                           ['step','subplatoon','status','success','error',
                            'independent_feasibility_status','independent_feasibility_error',
                            'primary_status','numerical_retry_solver','numerical_retry_status'])
            self.write_csv(self.out/'slack_records.csv',
                [dict(subplatoon=r[0]+1,step=r[1],constraint=r[2],horizon=r[3],slack=r[4])
                 for r in n['slack_records']],
                ['subplatoon','step','constraint','horizon','slack'])
            m = dict(completed=int(completed), last_valid_step=end,
                     planned_steps=n['total_time_steps']-1,
                     failure_status=self.failure.get('status','') if self.failure else '',
                     failure_reason=self.failure.get('error','') if self.failure else '')
            n['save_metrics'](m, filename=str(self.out/'metrics.csv'))
            self.metrics = m
            self._save_progress()
            return
        m = dict(mode=self.mode, scenario=n['scenario'], completed=int(completed),
                 step_seconds=n['Tstep'], history_steps=n['Npst'], last_valid_step=end,
                 cutin_time_s=n.get('cutin_time'), SlackEnabled=int(n['enable_slack']),
                 TimingValid=int(self.timing_run),
                 failure_status=self.failure.get('status', '') if self.failure else '',
                 failure_reason=self.failure.get('error', '') if self.failure else '')
        if self.collect_table:
            tracking = n['compute_tracking_metrics'](states, n['scenario'], sdes=n['SDES'])
            m.update({key: tracking[key] for key in ('RMSVE', 'RMSSE')})
            cavs = np.asarray(n['sub_start']) + 1
            gap = states[:, cavs-1, 0] - states[:, cavs, 0]
            velocity = states[:, cavs, 1]
            m['CAV_spacing_violation_count'] = int(((gap < n['SpaMin']-1e-3) |
                                                   (gap > n['SpaMax']+1e-3)).sum())
            m['CAV_velocity_violation_count'] = int(((velocity < n['VelMin']-1e-3) |
                                                    (velocity > n['VelMax']+1e-3)).sum())
            slack_rows = n['slack_records']
            m['slack_triggered_solves'] = len({(r[0], r[1]) for r in slack_rows})
            self.write_csv(self.out/'slack_records.csv',
                [dict(subplatoon=r[0]+1, step=r[1], constraint=r[2], horizon=r[3], slack=r[4])
                 for r in slack_rows], ['subplatoon', 'step', 'constraint', 'horizon', 'slack'])
            for name, rows, headers in [
                ('coverage', self.coverage, ['step', 'subplatoon', 'horizon', 'covered_at_prediction_origin']),
                ('tightening', self.tightening, ['step', 'subplatoon', 'horizon', 'spacing_lower', 'spacing_upper'])]:
                if name == 'coverage' and self.mode == 'nominal':
                    continue
                with (self.out/(name+'.csv')).open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(headers)
                    writer.writerows(rows)
        if self.timing_run:
            np.savetxt(self.out/'controller_computation_time.csv',
                       n['controller_compute_time_collection'][:, :end], delimiter=',')
        if not self.timing_run and (not n['enable_slack'] or not completed):
            self.write_csv(self.out/'solve_attempts.csv', self.attempts,
                ['step', 'subplatoon', 'status', 'success', 'error',
                 'independent_feasibility_status', 'independent_feasibility_error',
                 'primary_status', 'numerical_retry_solver', 'numerical_retry_status'])
        self.metrics = m
        n['save_metrics'](m, filename=str(self.out/'metrics.csv'))
        full = n['S'][:n['Npst']+end+1].copy()
        n['save_vehicle_data_to_csv'](full, filename_prefix=str(self.out/'vehicle_data'))
        if n['save_plots'] and len(full) > 1:
            n['plot'](n['args'], n['scenario'], full, n['Tstep'], n['sim_subplatoon_num'],
                      n['sub_numHDV'], out_dir=str(self.out))
        self._save_progress()
        print(f'[ablation] {self.mode}: completed={completed}, last valid step={end}', flush=True)
