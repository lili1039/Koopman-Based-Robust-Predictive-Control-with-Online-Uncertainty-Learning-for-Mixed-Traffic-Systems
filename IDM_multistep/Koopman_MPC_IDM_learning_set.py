# Basic imports
import os
scenario = os.environ.get('KMPC_SCENARIO', 'braking')
if scenario not in ('tracking', 'braking', 'cutin'):
    raise ValueError('Single-run scenarios are tracking and braking.')
if scenario == 'cutin' and not os.environ.get('KMPC_ABLATION_MODE', '').strip():
    raise ValueError('Run uncertainty_ablation.bat or run_repeated_safety.bat for cut-in experiments.')

# Bound heavy workers before torch/cvxpy imports; Windows releases file locks on exit.
if os.environ.get('KMPC_REPEATED_RUN') == '1':
    import json
    import msvcrt
    import time as _slot_time
    from pathlib import Path as _SlotPath
    _case_lock_name = os.environ.get('KMPC_CASE_LOCK', '').strip()
    if _case_lock_name:
        _case_lock_path = _SlotPath(_case_lock_name)
        if not _case_lock_path.is_absolute():
            raise ValueError('KMPC_CASE_LOCK must be an absolute file path.')
        _case_lock_handle = _case_lock_path.open('a+b')
        try:
            if os.fstat(_case_lock_handle.fileno()).st_size == 0:
                _case_lock_handle.write(b'\0')
                _case_lock_handle.flush()
            _case_lock_handle.seek(0)
            msvcrt.locking(_case_lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            _case_lock_handle.close()
            raise RuntimeError(f'Case already running; lock unavailable: {_case_lock_path}') from None
    _policy_path = _SlotPath(os.environ['KMPC_RESULT_ROOT'])/'execution_policy.json'
    if _policy_path.exists():
        _active_limit = int(json.loads(_policy_path.read_text())['active_process_limit'])
        _slot_dir = _SlotPath(os.environ['KMPC_ABLATION_WORK_DIR']).parent/'slots'
        _slot_dir.mkdir(exist_ok=True)
        _repeat_slot_handle = None
        while _repeat_slot_handle is None:
            for _slot in range(_active_limit):
                _handle = (_slot_dir/f'{_slot}.lock').open('a+b')
                _handle.seek(0)
                _handle.seek(0)
                try:
                    msvcrt.locking(_handle.fileno(), msvcrt.LK_NBLCK, 1)
                    _repeat_slot_handle = _handle
                    break
                except OSError:
                    _handle.close()
            if _repeat_slot_handle is None:
                _slot_time.sleep(1)
import time
import numpy as np
import cvxpy as cp
from concurrent.futures import ThreadPoolExecutor

# Network-related imports
import torch
from uncertainty_ablation import UncertaintyAblation

ablation = UncertaintyAblation.from_environment()
if ablation:
    torch.set_num_threads(1)
    if ablation.repeated_run:
        import random
        simulation_seed = int(os.environ['KMPC_SIM_SEED'])
        random.seed(simulation_seed)
        np.random.seed(simulation_seed)
        torch.manual_seed(simulation_seed)
        torch.use_deterministic_algorithms(True)

import _heter_helperfuns_torch_IDM_learning_set as helper
from _heter_helperfuns_torch_IDM_learning_set import (
    Min_Zonotope_fixed_vG_Solver,
    Online_learning_Zonotope_Solver,
    Whether_in_Zonotope_Solver,
    _env_flag,
    _env_float,
    _env_int,
    _env_int_list,
    compute_tracking_metrics,
    plot,
    plot_alpha_evolution,
    predict_future_ep_and_constraints,
    read_csv_2_np,
    save_computation_time,
    save_disturbance_set_data,
    save_metrics,
    save_vehicle_data_to_csv,
)

# ====================== Tunable parameters (same order as run_mpc.bat) ======================
# core
control_method = os.environ.get('KMPC_CONTROL', 'Distributed MPC')
disturbance_estimate = _env_flag('KMPC_DISTURBANCE_ESTIMATE', False)
Nfut = _env_int('KMPC_NFUT', 15)
# Platoon composition: comma-separated HDV counts for each subplatoon behind the PV; also the HDV model index.
subplatoon_spec = _env_int_list('KMPC_SUBPLATOON', [3, 1, 2, 4])

# spacing / constraint bounds
SDES = _env_float('KMPC_SDES', 40.0)
SpaMin = _env_float('KMPC_SPA_MIN', 10)
SpaMax = _env_float('KMPC_SPA_MAX', 80)
VelMin = _env_float('KMPC_VEL_MIN', 0)
VelMax = _env_float('KMPC_VEL_MAX', 35)
AccMin = _env_float('KMPC_ACC_MIN', -6)
AccMax = _env_float('KMPC_ACC_MAX', 4)

# online correction / error set
lam = _env_float('KMPC_LAM', 0.7)              # Exponential smoothing coefficient for alpha; larger values give stronger inertia.
N_point = _env_int('KMPC_NPOINT', 10)          # Number of historical error points in the refit sliding window.
ep_window = _env_int('KMPC_EP_WINDOW', 20)
corr_threads = _env_int('KMPC_CORR_THREADS', 8)

# soft-constraint / slack
# enable_slack controls whether slack is added to the optimization; save_slack only controls detailed slack logging.
enable_slack = _env_flag('KMPC_ENABLE_SLACK', True)
W_SLACK = _env_float('KMPC_SLACK_W', 1.0e4)    # L1 slack penalty weight; large enough approximates hard constraints.
W_SLACK_L2 = _env_float('KMPC_SLACK_W_L2', 1.0e2)  # Additional small L2 term to improve numerical conditioning.
save_slack = _env_flag('KMPC_SAVE_SLACK', False)
SLACK_TOL = _env_float('KMPC_SLACK_TOL', 1.0e-6)   # Slack trigger threshold: log a single-step constraint relaxation above this value when save_slack is enabled.
verbose_step = _env_flag('KMPC_VERBOSE_STEP', True)
save_plots = _env_flag('KMPC_SAVE_PLOTS', True)

# output / vehicle params
result_root = os.environ.get('KMPC_RESULT_ROOT', 'result')
run_suffix = os.environ.get('KMPC_RUN_SUFFIX', '')
vehicle_csv = os.environ.get('KMPC_VEHICLE_CSV', 'vehicle_parameters.csv')

# optional: signal delay (tracking + Distributed MPC)
signal_delay_requested = _env_flag('KMPC_SIGNAL_DELAY', False)
delay_seed = _env_int('KMPC_DELAY_SEED', 202606)
delay_count = _env_int('KMPC_DELAY_COUNT', 0)
delay_min_duration = _env_int('KMPC_DELAY_MIN_DURATION', 1)
delay_max_duration = _env_int('KMPC_DELAY_MAX_DURATION', delay_min_duration)
delay_targets_spec = os.environ.get('KMPC_DELAY_HDVS', '')

def _env_float_first(names, default):
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return float(value)
    return default

# Shared MPC weights for tracking, braking and the batch cut-in experiments.
dec_q_spa = _env_float_first(['KMPC_DEC_Q_SPA', 'KMPC_Q_SPA'], 2.0)
dec_q_vel = _env_float_first(['KMPC_DEC_Q_VEL', 'KMPC_Q_VEL'], 0.8)
dec_qa_spa = _env_float_first(['KMPC_DEC_QA_SPA', 'KMPC_QA_SPA'], 0.0)
dec_qa_vel = _env_float_first(['KMPC_DEC_QA_VEL', 'KMPC_QA_VEL'], 0.0)
dec_r = _env_float_first(['KMPC_DEC_R', 'KMPC_R'], 20.0)
dec_rd = _env_float_first(['KMPC_DEC_RD', 'KMPC_RD'], 0.0)

dist_q1_spa = _env_float_first(['KMPC_DIST_Q1_SPA', 'KMPC_DIST_Q_SPA'], 0.1)
dist_q1_vel = _env_float_first(['KMPC_DIST_Q1_VEL', 'KMPC_DIST_Q_VEL'], 0.25)
dist_q2_spa = _env_float_first(['KMPC_DIST_Q2_SPA', 'KMPC_DIST_Q_SPA'], 0.1)
dist_q2_vel = _env_float_first(['KMPC_DIST_Q2_VEL', 'KMPC_DIST_Q_VEL'], 0.25)
dist_qa_spa = _env_float_first(['KMPC_DIST_QA_SPA', 'KMPC_DIST_Q3_SPA'], 0.1)
dist_qa_vel = _env_float_first(['KMPC_DIST_QA_VEL', 'KMPC_DIST_Q3_VEL'], 0.5)
dist_q4_spa = _env_float_first(['KMPC_DIST_Q4_SPA', 'KMPC_DIST_Q_SPA'], 0.1)
dist_q4_vel = _env_float_first(['KMPC_DIST_Q4_VEL', 'KMPC_DIST_Q_VEL'], 0.5)
dist_r = _env_float_first(['KMPC_DIST_R', 'KMPC_R'], 25.0)
dist_rd = _env_float_first(['KMPC_DIST_RD', 'KMPC_RD'], 0.0)
# Reference selection for the Distributed terminal-speed constraint, active only for tracking/braking/cutin:
#   'weighted' = normalized weighted average using reference velocity weights (Q1v,Q2v,Q3v,Q4v), consistent with cost priorities.
#   Comma subsets such as 'ref' or 'pv,ref' use an equal average over selected references (pv=PV, preceed=previous CAV, ref=direct leader, assume=previous self prediction).
#   Default 'pv,preceed,ref,assume' keeps the original equal average over four references.
dist_term_refs = os.environ.get('KMPC_DIST_TERM_REFS', 'pv,preceed,ref,assume').strip()

# Optional: random packet loss for upstream CAV optimal-prediction messages in Distributed MPC.
# Receiver indices are 1-based subplatoon numbers; receiver 2 loses messages from CAV1, receiver 3 from CAV2, etc.
cav_pred_loss_requested = _env_flag('KMPC_CAV_PRED_LOSS', False)
cav_pred_loss_count = _env_int('KMPC_CAV_PRED_LOSS_COUNT', 0)
cav_pred_loss_min_duration = _env_int('KMPC_CAV_PRED_LOSS_MIN_DURATION', 1)
cav_pred_loss_max_duration = _env_int('KMPC_CAV_PRED_LOSS_MAX_DURATION', cav_pred_loss_min_duration)
cav_pred_loss_seed = _env_int('KMPC_CAV_PRED_LOSS_SEED', 202607)
cav_pred_loss_receivers_spec = os.environ.get('KMPC_CAV_PRED_LOSS_RECEIVERS', '')

mpc_weight_settings = {
    'dec_q_spa': dec_q_spa,
    'dec_q_vel': dec_q_vel,
    'dec_qa_spa': dec_qa_spa,
    'dec_qa_vel': dec_qa_vel,
    'dec_r': dec_r,
    'dec_rd': dec_rd,
    'dist_q1_spa': dist_q1_spa,
    'dist_q1_vel': dist_q1_vel,
    'dist_q2_spa': dist_q2_spa,
    'dist_q2_vel': dist_q2_vel,
    'dist_qa_spa': dist_qa_spa,
    'dist_qa_vel': dist_qa_vel,
    'dist_q4_spa': dist_q4_spa,
    'dist_q4_vel': dist_q4_vel,
    'dist_r': dist_r,
    'dist_rd': dist_rd,
    'dist_term_refs': dist_term_refs,
}

if not run_suffix.strip():
    # Keep readable case names and preserve earlier results when a method is rerun.
    method_name = control_method.replace(' ', '_')
    number = 2
    while os.path.exists(os.path.join(result_root, scenario, method_name + run_suffix)):
        run_suffix = '_' + str(number)
        number += 1
output_dir = os.path.join(result_root, scenario, control_method.replace(' ', '_') + run_suffix)

# ====================== Internal fixed settings ======================
time_varying = _env_flag('KMPC_TIME_VARYING', True)
disturbance_update_method = 'refit'
# Cut-in settings are supplied by the uncertainty-comparison and repeated-safety launchers.
if scenario == 'cutin':
    cutin_time = float(os.environ['KMPC_CUTIN_TIME'])
    cutin_gap_target = float(os.environ['KMPC_CUTIN_GAP'])
    cutin_brake_duration = float(os.environ['KMPC_CUTIN_BRAKE_DURATION'])
    cutin_final_speed = float(os.environ['KMPC_CUTIN_FINAL_SPEED'])
    if cutin_gap_target <= 0:
        raise ValueError('KMPC_CUTIN_GAP must be positive.')


def _inject_cutin_position(states, index):
    """Set the post-cut-in gap used by the batch experiments."""
    previous_position = float(states[index, 0, 0])
    states[index, 0, 0] = states[index, 1, 0] + cutin_gap_target
    return previous_position - float(states[index, 0, 0])


def _parse_cav_pred_loss_receivers(spec, sim_subplatoon_num):
    """Parse receiving subplatoons for upstream CAV prediction packet loss."""
    if sim_subplatoon_num <= 1:
        return []
    if spec.strip():
        receivers = []
        for item in spec.split(','):
            item = item.strip()
            if not item:
                continue
            receiver = int(item) - 1
            if receiver <= 0 or receiver >= sim_subplatoon_num:
                raise ValueError(f'KMPC_CAV_PRED_LOSS_RECEIVERS must use receiving subplatoons 2..{sim_subplatoon_num}: {item}')
            receivers.append(receiver)
        return sorted(set(receivers))
    return list(range(1, sim_subplatoon_num))


def _build_random_cav_pred_loss_windows(count, min_duration, max_duration, total_time_steps, receivers, seed):
    """Generate random packet-loss windows as (start_step, end_step, receiver_idx)."""
    if count <= 0 or not receivers:
        return []
    if min_duration <= 0 or max_duration <= 0:
        raise ValueError('KMPC_CAV_PRED_LOSS_MIN_DURATION and KMPC_CAV_PRED_LOSS_MAX_DURATION must be positive.')
    if min_duration > max_duration:
        min_duration, max_duration = max_duration, min_duration
    max_step = max(total_time_steps - 2, 1)
    min_start = 2 if max_step >= 2 else 1
    rng = np.random.default_rng(seed)
    windows = []
    for _ in range(count):
        receiver = int(rng.choice(receivers))
        start = int(rng.integers(min_start, max_step + 1))
        duration = int(rng.integers(min_duration, max_duration + 1))
        end = min(max_step, start + duration - 1)
        windows.append((start, end, receiver))
    return sorted(windows)


def _steps_by_receiver_from_windows(windows):
    steps_by_receiver = {}
    for start, end, receiver in windows:
        steps_by_receiver.setdefault(receiver, set()).update(range(start, end + 1))
    return steps_by_receiver


def _parse_hdv_delay_candidates(spec, sub_numHDV):
    """Parse optional HDV delay candidates; blank means every HDV in every subplatoon."""
    if spec.strip():
        candidates = []
        for item in spec.split(','):
            item = item.strip()
            if not item:
                continue
            sub_s, hdv_s = item.split(':', 1)
            sub_idx = int(sub_s) - 1
            hdv_idx = int(hdv_s)
            if sub_idx < 0 or sub_idx >= len(sub_numHDV):
                raise ValueError(f'Subplatoon index out of range in KMPC_DELAY_HDVS: {item}')
            if hdv_idx < 1 or hdv_idx > sub_numHDV[sub_idx]:
                raise ValueError(f'HDV index out of range in KMPC_DELAY_HDVS: {item}')
            candidates.append((sub_idx, hdv_idx))
        return sorted(set(candidates))
    return [
        (sub_idx, hdv_idx)
        for sub_idx, hdv_count in enumerate(sub_numHDV)
        for hdv_idx in range(1, hdv_count + 1)
    ]


def _build_random_hdv_delay_windows(count, min_duration, max_duration, total_time_steps, candidates, seed):
    """Generate random HDV signal-delay windows as (start_step, end_step, subplatoon_idx, hdv_idx)."""
    if count <= 0 or not candidates:
        return []
    if min_duration <= 0 or max_duration <= 0:
        raise ValueError('KMPC_DELAY_MIN_DURATION and KMPC_DELAY_MAX_DURATION must be positive.')
    if min_duration > max_duration:
        min_duration, max_duration = max_duration, min_duration
    max_step = max(total_time_steps - 2, 1)
    rng = np.random.default_rng(seed)
    windows = []
    for _ in range(count):
        target_idx = int(rng.integers(0, len(candidates)))
        sub_idx, hdv_idx = candidates[target_idx]
        start = int(rng.integers(1, max_step + 1))
        duration = int(rng.integers(min_duration, max_duration + 1))
        end = min(max_step, start + duration - 1)
        windows.append((start, end, sub_idx, hdv_idx))
    return sorted(windows)


def _hdv_delay_targets_by_step(windows):
    targets_by_step = {}
    for start, end, sub_idx, hdv_idx in windows:
        for step in range(start, end + 1):
            targets_by_step.setdefault(step, {}).setdefault(sub_idx, set()).add(hdv_idx)
    return targets_by_step


def _shift_prediction_once(prediction):
    """One-step left shift for an optimal prediction; repeat the tail column."""
    return np.concatenate([prediction[:, 1:], prediction[:, -1:]], axis=1)

# ============================= MPC Solvers =================================
# --------- Decentralized Solver ---------
def _project_terminal_speed(target, lower, upper):
    """Project the terminal target onto the last prediction-step velocity bounds."""
    if lower > upper:
        raise RuntimeError(f'Empty terminal velocity interval: [{lower}, {upper}]')
    return float(np.clip(float(target), lower, upper))


class Decentralized_VehicleSolver:
    def __init__(self, G0, Gu, Ge, D, SpaMax, SpaMin, VelMax, VelMin, AccMin, AccMax, n_high, n_low, Nfut, Tstep, scenario):
        self.Tstep = Tstep
        assert len(G0) >= Nfut, f"Exported model steps {len(G0)} < MPC prediction steps Nfut={Nfut}"
        # ===== Parameter definitions; values are updated at each solve. =====
        self.s_t = cp.Parameter((n_high,), name="s_t")
        self.x_ref = cp.Parameter((n_low,Nfut+1), name="x_ref")
        self.x_assume = cp.Parameter((n_low,Nfut+1), name="x_assume")
        self.ep  = cp.Parameter((1,Nfut+1), name="ep")
        self.Spa_lb_param = cp.Parameter((Nfut+1,))
        self.Spa_ub_param = cp.Parameter((Nfut+1,))
        self.Vel_lb_param = cp.Parameter((Nfut+1,))
        self.Vel_ub_param = cp.Parameter((Nfut+1,))
        self.u_lb_param = cp.Parameter((Nfut,))
        self.u_ub_param = cp.Parameter((Nfut,))
        self.u_prev = cp.Parameter((1,), name="u_prev")   # Previously applied control, used for the delta-u penalty.

        # ===== Decision variables =====
        self.s = cp.Variable((n_high, Nfut+1))
        self.u = cp.Variable((1, Nfut))
        self.x = cp.Variable((n_low, Nfut+1))

        # ===== Soft-constraint slack variables for spacing/velocity state constraints and terminal velocity; acceleration remains hard-constrained. =====
        self.sl_spa_lb = cp.Variable((Nfut,), nonneg=True)
        self.sl_spa_ub = cp.Variable((Nfut,), nonneg=True)
        self.sl_vel_lb = cp.Variable((Nfut,), nonneg=True)
        self.sl_vel_ub = cp.Variable((Nfut,), nonneg=True)
        self.sl_term   = cp.Variable(nonneg=True)   # Symmetric slack for the terminal velocity equality.

        # ===== Weights, adapted to n_low by repeating [spacing, velocity] weights for each vehicle. =====
        Q = np.diag([dec_q_spa, dec_q_vel] * (n_low//2))
        Qa = np.diag([dec_qa_spa, dec_qa_vel] * (n_low//2))  # Stay close to the previous optimal predicted trajectory to suppress receding-horizon jumps.
        R = dec_r*np.eye(1)      # Control input.
        Rd = dec_rd*np.eye(1)     # Delta-u control-increment penalty.

        # ===== Constraints and objective =====
        constraints = []
        self.SpaMin = SpaMin
        self.SpaMax = SpaMax
        self.VelMin = VelMin
        self.VelMax = VelMax
        self.AccMin = AccMin
        self.AccMax = AccMax
        self.Nfut = Nfut
        self.n_high = n_high
        self.n_low = n_low
        objective = 0

        # Direct multi-step prediction: g(t+j) = G0[j-1]@s0 + sum_l(Gu[j-1][l]@u[l] + Ge[j-1][l]@ep[l]).
        for j in range(1, Nfut+1):
            pred = G0[j-1] @ self.s[:, 0]
            for l in range(j):
                pred = pred + Gu[j-1][l] @ self.u[:, l] + Ge[j-1][l] @ self.ep[:, l]
            step_constraints = [
                self.s[:, j] == pred,
                self.x[:, j] == D @ self.s[:, j],
                self.u[0, j-1] >= self.u_lb_param[j-1],
                self.u[0, j-1] <= self.u_ub_param[j-1],
            ]
            if enable_slack:
                step_constraints += [
                    self.x[0, j] <= self.Spa_ub_param[j] + self.sl_spa_ub[j-1],
                    self.x[0, j] >= self.Spa_lb_param[j] - self.sl_spa_lb[j-1],
                    self.x[1, j] <= self.Vel_ub_param[j] + self.sl_vel_ub[j-1],
                    self.x[1, j] >= self.Vel_lb_param[j] - self.sl_vel_lb[j-1],
                ]
            else:
                step_constraints += [
                    self.x[0, j] <= self.Spa_ub_param[j],
                    self.x[0, j] >= self.Spa_lb_param[j],
                    self.x[1, j] <= self.Vel_ub_param[j],
                    self.x[1, j] >= self.Vel_lb_param[j],
                ]
            constraints += step_constraints
            objective += (cp.quad_form((self.x[:,j]-self.x_ref[:,j]), Q)
                          + cp.quad_form((self.x[:,j]-self.x_assume[:,j]), Qa)
                          + cp.quad_form((self.u[:,j-1]), R))

        # ===== Delta-u control-increment penalty, including the difference from the previously applied control. =====
        objective += cp.quad_form(self.u[:, 0] - self.u_prev, Rd)
        for l in range(1, Nfut):
            objective += cp.quad_form(self.u[:, l] - self.u[:, l-1], Rd)

        # ===== Common L1/L2 penalties discourage relaxation; finite penalties can trade slack against tracking cost. =====
        if enable_slack:
            slack_all = (cp.sum(self.sl_spa_lb) + cp.sum(self.sl_spa_ub)
                         + cp.sum(self.sl_vel_lb) + cp.sum(self.sl_vel_ub) + self.sl_term)
            objective += W_SLACK * slack_all
            objective += W_SLACK_L2 * (cp.sum_squares(self.sl_spa_lb) + cp.sum_squares(self.sl_spa_ub)
                                       + cp.sum_squares(self.sl_vel_lb) + cp.sum_squares(self.sl_vel_ub)
                                       + cp.square(self.sl_term))

        constraints += [self.s[:, 0] == self.s_t]    # Initial state.
        self.terminal_target_raw_expr = self.x_ref[1,Nfut]
        self.terminal_target_param = cp.Parameter(name='terminal_target')
        if enable_slack:
            constraints += [
                self.x[1,Nfut] <= self.terminal_target_param + self.sl_term,   # Terminal state, softened.
                self.x[1,Nfut] >= self.terminal_target_param - self.sl_term,
            ]
        else:
            constraints += [self.x[1,Nfut] == self.terminal_target_param]

        # ===== Build the optimization problem =====
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def _slack_report(self):
        """Return soft-constraint slack values from this solve for simulation-side logging."""
        if not enable_slack:
            return None
        return {
            'spa_lb': np.asarray(self.sl_spa_lb.value).ravel(),
            'spa_ub': np.asarray(self.sl_spa_ub.value).ravel(),
            'vel_lb': np.asarray(self.sl_vel_lb.value).ravel(),
            'vel_ub': np.asarray(self.sl_vel_ub.value).ravel(),
            'term': float(self.sl_term.value),
        }

    def solve(self, s_t_val, x_ref_val, x_assume_val, ep_val, Ex_interval, u_prev_val):
        # Update parameters.
        self.s_t.value = s_t_val
        self.x_ref.value = x_ref_val
        self.x_assume.value = x_assume_val
        self.ep.value = ep_val
        self.u_prev.value = u_prev_val

        # Update constraints.
        Spa_lb = np.zeros((self.Nfut+1,))
        Spa_ub = np.zeros((self.Nfut+1,))
        Vel_lb = np.zeros((self.Nfut+1,))
        Vel_ub = np.zeros((self.Nfut+1,))
        u_lb = np.zeros((self.Nfut,))
        u_ub = np.zeros((self.Nfut,))

        if disturbance_estimate == False or (ablation and ablation.nominal_active): # Do not tighten constraints.
            for i in range(self.Nfut+1):
                Spa_lb[i] = self.SpaMin
                Spa_ub[i] = self.SpaMax
                Vel_lb[i] = self.VelMin
                Vel_ub[i] = self.VelMax
                if i < self.Nfut:
                    u_lb[i] = self.AccMin
                    u_ub[i] = self.AccMax
        else:
            for i in range(self.Nfut+1):
                if i == 0:
                    Spa_lb[i] = self.SpaMin
                    Spa_ub[i] = self.SpaMax
                    Vel_lb[i] = self.VelMin
                    Vel_ub[i] = self.VelMax
                else:
                    Spa_lb[i] = self.SpaMin - Ex_interval[i-1].left_limit[0]
                    Spa_ub[i] = self.SpaMax - Ex_interval[i-1].right_limit[0]
                    Vel_lb[i] = self.VelMin - Ex_interval[i-1].left_limit[1]
                    Vel_ub[i] = self.VelMax - Ex_interval[i-1].right_limit[1]
                if i < self.Nfut:
                    u_lb[i] = self.AccMin
                    u_ub[i] = self.AccMax
        
        self.Spa_lb_param.value = Spa_lb
        self.Spa_ub_param.value = Spa_ub
        self.Vel_lb_param.value = Vel_lb
        self.Vel_ub_param.value = Vel_ub
        self.u_lb_param.value = u_lb
        self.u_ub_param.value = u_ub

        self.terminal_target_param.value = _project_terminal_speed(
            self.terminal_target_raw_expr.value, Vel_lb[self.Nfut], Vel_ub[self.Nfut])

        # Start timing.
        t0 = time.time()

        # Solve.
        self.prob.solve(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False
        )

        compute_time = time.time() - t0

        if self.prob.status in [cp.INFEASIBLE, cp.UNBOUNDED]:
            raise RuntimeError(f"Optimization failed: {self.prob.status}")

        return (self.x.value,
                self.s.value,
                self.u.value,
                compute_time,
                self._slack_report() if (enable_slack and save_slack) else None)

# --------- Distributed Solver ---------
class Distributed_VehicleSolver:
    def __init__(self, G0, Gu, Ge, D, SpaMax, SpaMin, VelMax, VelMin, AccMin, AccMax, n_high, n_low, Nfut, Tstep, scenario):
        self.Tstep = Tstep
        assert len(G0) >= Nfut, f"Exported model steps {len(G0)} < MPC prediction steps Nfut={Nfut}"
        # ===== Parameter definitions; values are updated at each solve. =====
        self.s_t = cp.Parameter((n_high,), name="s_t")
        self.x_pv = cp.Parameter((n_low,Nfut+1), name="x_pv")
        self.x_preceed = cp.Parameter((n_low,Nfut+1), name="x_preceed")
        self.x_assume = cp.Parameter((n_low, Nfut+1), name="x_assume")
        self.x_ref = cp.Parameter((n_low, Nfut+1), name="x_ref")
        self.ep  = cp.Parameter((1,Nfut+1), name="ep")
        self.Spa_lb_param = cp.Parameter((Nfut+1,))
        self.Spa_ub_param = cp.Parameter((Nfut+1,))
        self.Vel_lb_param = cp.Parameter((Nfut+1,))
        self.Vel_ub_param = cp.Parameter((Nfut+1,))
        self.u_lb_param = cp.Parameter((Nfut,))
        self.u_ub_param = cp.Parameter((Nfut,))
        self.u_prev = cp.Parameter((1,), name="u_prev")   # Previously applied control, used for the delta-u penalty.

        # ===== Decision variables =====
        self.s = cp.Variable((n_high, Nfut+1))
        self.u = cp.Variable((1, Nfut))
        self.x = cp.Variable((n_low, Nfut+1))

        # ===== Soft-constraint slack variables for spacing/velocity state constraints and terminal velocity; acceleration remains hard-constrained. =====
        self.sl_spa_lb = cp.Variable((Nfut,), nonneg=True)
        self.sl_spa_ub = cp.Variable((Nfut,), nonneg=True)
        self.sl_vel_lb = cp.Variable((Nfut,), nonneg=True)
        self.sl_vel_ub = cp.Variable((Nfut,), nonneg=True)
        self.sl_term   = cp.Variable(nonneg=True)   # Symmetric slack for the terminal velocity equality.

        # ===== Weights, adapted to n_low by repeating [spacing, velocity] weights for each vehicle. =====
        Q1 = np.diag([dist_q1_spa, dist_q1_vel] * (n_low//2))  # Reference tracking.
        Q2 = np.diag([dist_q2_spa, dist_q2_vel] * (n_low//2))
        Q3 = np.diag([dist_qa_spa, dist_qa_vel] * (n_low//2))
        Q4 = np.diag([dist_q4_spa, dist_q4_vel] * (n_low//2))
        R = dist_r*np.eye(1)
        Rd = dist_rd*np.eye(1)     # Delta-u control-increment penalty.


        self.n_high = n_high
        self.n_low = n_low

        # ===== Constraints and objective =====
        constraints = []
        self.SpaMin = SpaMin
        self.SpaMax = SpaMax
        self.VelMin = VelMin
        self.VelMax = VelMax
        self.AccMin = AccMin
        self.AccMax = AccMax
        self.Nfut = Nfut
        objective = 0

        # Direct multi-step prediction: g(t+j) = G0[j-1]@s0 + sum_l(Gu[j-1][l]@u[l] + Ge[j-1][l]@ep[l]).
        for j in range(1, Nfut+1):
            pred = G0[j-1] @ self.s[:, 0]
            for l in range(j):
                pred = pred + Gu[j-1][l] @ self.u[:, l] + Ge[j-1][l] @ self.ep[:, l]
            step_constraints = [
                self.s[:, j] == pred,
                self.x[:, j] == D @ self.s[:, j],
                self.u[0, j-1] >= self.u_lb_param[j-1],
                self.u[0, j-1] <= self.u_ub_param[j-1],
            ]
            if enable_slack:
                step_constraints += [
                    self.x[0, j] <= self.Spa_ub_param[j] + self.sl_spa_ub[j-1],
                    self.x[0, j] >= self.Spa_lb_param[j] - self.sl_spa_lb[j-1],
                    self.x[1, j] <= self.Vel_ub_param[j] + self.sl_vel_ub[j-1],
                    self.x[1, j] >= self.Vel_lb_param[j] - self.sl_vel_lb[j-1],
                ]
            else:
                step_constraints += [
                    self.x[0, j] <= self.Spa_ub_param[j],
                    self.x[0, j] >= self.Spa_lb_param[j],
                    self.x[1, j] <= self.Vel_ub_param[j],
                    self.x[1, j] >= self.Vel_lb_param[j],
                ]
            constraints += step_constraints
            objective += (cp.quad_form((self.x[:,j]-self.x_pv[:,j]), Q1) +
                         cp.quad_form((self.x[:,j]-self.x_preceed[:,j]), Q2) +
                         cp.quad_form((self.x[:,j]-self.x_assume[:,j]), Q3) +
                         cp.quad_form((self.x[:,j]-self.x_ref[:,j]), Q4) +
                         cp.quad_form((self.u[:,j-1]), R)
                         )

        # ===== Delta-u control-increment penalty, including the difference from the previously applied control. =====
        objective += cp.quad_form(self.u[:, 0] - self.u_prev, Rd)
        for l in range(1, Nfut):
            objective += cp.quad_form(self.u[:, l] - self.u[:, l-1], Rd)

        # ===== Common L1/L2 penalties discourage relaxation; finite penalties can trade slack against tracking cost. =====
        if enable_slack:
            slack_all = (cp.sum(self.sl_spa_lb) + cp.sum(self.sl_spa_ub)
                         + cp.sum(self.sl_vel_lb) + cp.sum(self.sl_vel_ub) + self.sl_term)
            objective += W_SLACK * slack_all
            objective += W_SLACK_L2 * (cp.sum_squares(self.sl_spa_lb) + cp.sum_squares(self.sl_spa_ub)
                                       + cp.sum_squares(self.sl_vel_lb) + cp.sum_squares(self.sl_vel_ub)
                                       + cp.square(self.sl_term))

        # Initial state.
        constraints += [self.s[:, 0] == self.s_t,]
        # Terminal state with symmetric slack sl_term.
        # Terminal velocity target selected by dist_term_refs; see KMPC_DIST_TERM_REFS parsing near the top of the file.
        #   'weighted' uses the normalized weighted average of reference velocity weights (Q1v,Q2v,Q3v,Q4v), consistent with cost priorities.
        #   A comma list, such as 'ref' or 'pv,preceed,ref,assume', uses an equal average over selected references; default uses all four.
        _tr = dist_term_refs
        if _tr == 'weighted':
            _wv = np.array([Q1[1, 1], Q2[1, 1], Q3[1, 1], Q4[1, 1]], dtype=float)  # Velocity weights for each reference.
            if _wv.sum() <= 0:
                _wv = np.ones(4)
            _wv = _wv / _wv.sum()
            term_target = (_wv[0] * self.x_pv[1, Nfut] + _wv[1] * self.x_preceed[1, Nfut]
                           + _wv[2] * self.x_assume[1, Nfut] + _wv[3] * self.x_ref[1, Nfut])
        else:
            _map = {'pv': self.x_pv[1, Nfut], 'preceed': self.x_preceed[1, Nfut],
                    'ref': self.x_ref[1, Nfut], 'assume': self.x_assume[1, Nfut]}
            _act = [_map[k.strip()] for k in _tr.split(',') if k.strip() in _map]
            term_target = sum(_act) / len(_act)
        self.terminal_target_raw_expr = term_target
        self.terminal_target_param = cp.Parameter(name='terminal_target')
        if enable_slack:
            constraints += [
                self.x[1,Nfut] <= self.terminal_target_param + self.sl_term,
                self.x[1,Nfut] >= self.terminal_target_param - self.sl_term,
            ]
        else:
            constraints += [self.x[1,Nfut] == self.terminal_target_param]

        # ===== Build the optimization problem =====
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def _slack_report(self):
        """Return soft-constraint slack values from this solve for simulation-side logging."""
        if not enable_slack:
            return None
        return {
            'spa_lb': np.asarray(self.sl_spa_lb.value).ravel(),
            'spa_ub': np.asarray(self.sl_spa_ub.value).ravel(),
            'vel_lb': np.asarray(self.sl_vel_lb.value).ravel(),
            'vel_ub': np.asarray(self.sl_vel_ub.value).ravel(),
            'term': float(self.sl_term.value),
        }

    def solve(self, s_t_val, x_pv_val, x_preceed_val, x_assume_val, x_ref_val, ep_val, Ex_interval, u_prev_val):
        # Update parameters.
        self.s_t.value = s_t_val
        self.x_pv.value = x_pv_val
        self.x_preceed.value = x_preceed_val
        self.x_assume.value = x_assume_val
        self.x_ref.value = x_ref_val
        self.ep.value = ep_val
        self.u_prev.value = u_prev_val

        # Update constraints.
        Spa_lb = np.zeros((self.Nfut+1,))
        Spa_ub = np.zeros((self.Nfut+1,))
        Vel_lb = np.zeros((self.Nfut+1,))
        Vel_ub = np.zeros((self.Nfut+1,))
        u_lb = np.zeros((self.Nfut,))
        u_ub = np.zeros((self.Nfut,))

        if disturbance_estimate == False or (ablation and ablation.nominal_active): # Do not tighten constraints.
            for i in range(self.Nfut+1):
                Spa_lb[i] = self.SpaMin
                Spa_ub[i] = self.SpaMax
                Vel_lb[i] = self.VelMin
                Vel_ub[i] = self.VelMax
                if i < self.Nfut:
                    u_lb[i] = self.AccMin
                    u_ub[i] = self.AccMax
        else:
            for i in range(self.Nfut+1):
                if i == 0:
                    Spa_lb[i] = self.SpaMin
                    Spa_ub[i] = self.SpaMax
                    Vel_lb[i] = self.VelMin
                    Vel_ub[i] = self.VelMax
                else:
                    Spa_lb[i] = self.SpaMin - Ex_interval[i-1].left_limit[0]
                    Spa_ub[i] = self.SpaMax - Ex_interval[i-1].right_limit[0]
                    Vel_lb[i] = self.VelMin - Ex_interval[i-1].left_limit[1]
                    Vel_ub[i] = self.VelMax - Ex_interval[i-1].right_limit[1]
                if i < self.Nfut:
                    u_lb[i] = self.AccMin
                    u_ub[i] = self.AccMax

        self.Spa_lb_param.value = Spa_lb
        self.Spa_ub_param.value = Spa_ub
        self.Vel_lb_param.value = Vel_lb
        self.Vel_ub_param.value = Vel_ub
        self.u_lb_param.value = u_lb
        self.u_ub_param.value = u_ub

        self.terminal_target_param.value = _project_terminal_speed(
            self.terminal_target_raw_expr.value, Vel_lb[self.Nfut], Vel_ub[self.Nfut])

        # Start timing.
        t0 = time.time()

        # Solve.
        self.prob.solve(
            solver=cp.OSQP,
            warm_start=True,
            # verbose=True,
            verbose=False,
        )

        compute_time = time.time() - t0
        
        if self.prob.status in [cp.INFEASIBLE, cp.UNBOUNDED]:
            raise RuntimeError(f"Optimization failed: {self.prob.status}")

        return (self.x.value,
                self.s.value,
                self.u.value,
                compute_time,
                self._slack_report() if (enable_slack and save_slack) else None)

if __name__ == '__main__':
    args = helper.get_args_local(False)
    args = vars(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if disturbance_estimate:
        print("Disturbance set / constraint tightening: ON")
    else:
        print("Disturbance set / constraint tightening: OFF")
    print(f"Soft constraints / slack in optimization: {'ON' if enable_slack else 'OFF'}")

    # ================ Platoon configuration ================================
    model_collection_path = args['model_collection_path']
    num_CAV = 1
    # Predefine model files for HDV 1/2/3/4 and build prediction models according to the configuration.
    HDV_MODEL_PATHS = {
        1: helper.HDV_1_MODEL_PATH,
        2: helper.HDV_2_MODEL_PATH,
        3: helper.HDV_3_MODEL_PATH,
        4: helper.HDV_4_MODEL_PATH,
    }
    subplatoon_config = [
        {'num_HDV': nh, 'model_file': HDV_MODEL_PATHS[nh]} for nh in subplatoon_spec
    ]
    sim_subplatoon_num = len(subplatoon_config)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _load_matrix_col(path):
        """Read a CSV and orient it for the column-vector convention (M @ s): returns (n_high, in_dim)."""
        return np.atleast_2d(read_csv_2_np(path)).T

    def _load_zono_csv(path, as_generator=False):
        """Read a zonotope c/G CSV; reshape a single generator into a 2D (dim, 1) column."""
        arr = np.loadtxt(path, delimiter=',')
        if as_generator and arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _load_model_bundle(cfg):
        """Load model, args, multi-step predictor matrices, per-step zonotopes, and set solvers for one subplatoon."""
        nh = cfg['num_HDV']
        # The date directory defaults to the first segment of model_file, e.g., '2026_06_29_00_39_02/past_models/...'.
        date_folder = cfg['model_file'].split('/')[0]
        weight_dir = cfg.get('weight_dir', date_folder + '/multistep_weights/')
        zono_dir   = cfg.get('zono_dir', date_folder)
        bundle_args = vars(helper.get_args_local(False, num_HDV_override=nh))
        model_Nfut = bundle_args['Nfut']           # Number of exported steps after training (=15), used as the read limit.
        model_path = model_collection_path + 'model_highd_multistep/' + cfg['model_file']
        m = helper.load_trained_model(bundle_args, model_path, device)
        wdir = model_collection_path + 'model_highd_multistep/' + weight_dir

        # Multi-step predictor matrices: read G0/Gu/Ge for steps k=1..model_Nfut, with column-vector convention M @ s.
        # G0_list[k-1]:(n_high,n_high)  Gu_list[k-1][lag]:(n_high,u_dim)  Ge_list[k-1][lag]:(n_high,ep_dim)
        G0_list, Gu_list, Ge_list = [], [], []
        for k in range(1, model_Nfut + 1):
            sname = f"step{k:02d}"
            G0_list.append(_load_matrix_col(os.path.join(wdir, f"G0_{sname}.csv")))
            Gu_k, Ge_k = [], []
            for lag in range(k):
                lname = f"lag{lag:02d}"
                Gu_k.append(_load_matrix_col(os.path.join(wdir, f"Gu_{sname}_{lname}.csv")))
                Ge_k.append(_load_matrix_col(os.path.join(wdir, f"Ge_{sname}_{lname}.csv")))
            Gu_list.append(Gu_k)
            Ge_list.append(Ge_k)
        D_ = read_csv_2_np(os.path.join(wdir, 'decoder_weight.csv'))
        n_high = G0_list[0].shape[0]
        n_low = D_.shape[0]
        print(f"[load] num_HDV={nh}: {model_path}")
        print(f"        multi-step matrices step1..{model_Nfut}: G0{G0_list[0].shape} Gu{Gu_list[0][0].shape} Ge{Ge_list[0][0].shape} D{D_.shape} -> n_high={n_high}, n_low={n_low}")
        bundle = {
            'args': bundle_args, 'model': m,
            'G0': G0_list, 'Gu': Gu_list, 'Ge': Ge_list, 'D': D_,
            'n_high': n_high, 'n_low': n_low, 'model_Nfut': model_Nfut,
        }
        if disturbance_estimate:
            zdir = os.path.join(project_root, 'Zonosets_and_preparations', '_5_multi_step',
                                'error_points_multistep', zono_dir, 'zonotopes')
            if not os.path.isdir(zdir):
                raise FileNotFoundError(
                    f"Per-step zonotope directory not found: {zdir}\n"
                    f"Run error_statistic_multistep.py and enclosing_set.m first to generate "
                    f"Zono_omega_stepKK_*.csv and Zono_gamma_*.csv.")
            # omega is prediction-step dependent; gamma is the shared autoencoder error set, with the same c/G reused for all steps.
            c_omega_list, G_omega_list, c_gamma_list, G_gamma_list = [], [], [], []
            check_list, online_list, refit_list = [], [], []
            cg = _load_zono_csv(os.path.join(zdir, "Zono_gamma_c.csv"))
            Gg = _load_zono_csv(os.path.join(zdir, "Zono_gamma_G.csv"), as_generator=True)
            for k in range(1, model_Nfut + 1):
                sname = f"step{k:02d}"
                ct = _load_zono_csv(os.path.join(zdir, f"Zono_omega_{sname}_c.csv"))
                Gt = _load_zono_csv(os.path.join(zdir, f"Zono_omega_{sname}_G.csv"), as_generator=True)
                c_omega_list.append(ct); G_omega_list.append(Gt)
                c_gamma_list.append(cg); G_gamma_list.append(Gg)
                needs_updates = not (ablation and ablation.repeated_run and
                    (not ablation.should_update or control_method == 'All HDVs'))
                check_list.append(Whether_in_Zonotope_Solver(ct.shape, Gt.shape, Gt) if needs_updates else None)
                online_list.append(Online_learning_Zonotope_Solver(ct, Gt) if needs_updates else None)
                refit_list.append(Min_Zonotope_fixed_vG_Solver(ct, Gt, N_point)
                                  if needs_updates and disturbance_update_method == 'refit' else None)
            bundle.update({
                'c_template': c_omega_list, 'G_template': G_omega_list,
                'c_gamma': c_gamma_list, 'G_gamma': G_gamma_list,
                'check_solver': check_list, 'online_solver': online_list, 'refit_solver': refit_list,
            })
        else:
            bundle.update({
                'c_template': None, 'G_template': None, 'c_gamma': None, 'G_gamma': None,
                'check_solver': None, 'online_solver': None, 'refit_solver': None,
            })
        return bundle

    # Load with a model_file cache and expand into per-subplatoon lists.
    _bundle_cache = {}
    sub_bundle = []
    for cfg in subplatoon_config:
        if cfg['model_file'] not in _bundle_cache:
            _bundle_cache[cfg['model_file']] = _load_model_bundle(cfg)
        sub_bundle.append(_bundle_cache[cfg['model_file']])

    sub_numHDV = [cfg['num_HDV'] for cfg in subplatoon_config]
    sub_numveh = [num_CAV + nh for nh in sub_numHDV]
    sub_sdim   = [2*nv for nv in sub_numveh]
    sub_args   = [b['args'] for b in sub_bundle]
    sub_model  = [b['model'] for b in sub_bundle]
    sub_G0 = [b['G0'] for b in sub_bundle]
    sub_Gu = [b['Gu'] for b in sub_bundle]
    sub_Ge = [b['Ge'] for b in sub_bundle]
    sub_D = [b['D'] for b in sub_bundle]
    sub_nhigh = [b['n_high'] for b in sub_bundle]
    sub_nlow  = [b['n_low'] for b in sub_bundle]
    sub_ctemplate = [b['c_template'] for b in sub_bundle]
    sub_Gtemplate = [b['G_template'] for b in sub_bundle]
    sub_cgamma = [b['c_gamma'] for b in sub_bundle]
    sub_Ggamma = [b['G_gamma'] for b in sub_bundle]
    sub_check_solver  = [b['check_solver'] for b in sub_bundle]
    sub_online_solver = [b['online_solver'] for b in sub_bundle]
    sub_refit_solver  = [b['refit_solver'] for b in sub_bundle]

    # Global leader index uses cumulative offsets. Subplatoon k occupies sub_start[k]+1 .. sub_start[k]+sub_numveh[k], with its CAV at sub_start[k]+1.
    sub_start = [0]*sim_subplatoon_num
    for k in range(1, sim_subplatoon_num):
        sub_start[k] = sub_start[k-1] + sub_numveh[k-1]
    sim_num_veh = sum(sub_numveh)
    print(f"Platoon composition: {['1C%dH' % h for h in sub_numHDV]}, total following vehicles (excluding PV)={sim_num_veh}, sub_start={sub_start}")

    # ============================ Initialize each vehicle ===================================================
    Tstep = args['Time_step']
    if scenario == 'tracking':
        time_list, PV_vel = helper.generate_NEDC_velocity_profile(Tstep)
    elif scenario == 'braking':
        time_list, PV_vel = helper.generate_braking_velocity_profile(Tstep)
    elif scenario == 'cutin':
        # Scene settings come from the launcher; the position jump is applied in the simulation loop.
        time_list, PV_vel = helper.generate_cutin_velocity_profile(
            Tstep, cutin_time=cutin_time, brake_duration=cutin_brake_duration, final_speed=cutin_final_speed)
        cutin_step = int(round(cutin_time / Tstep))  # Inject the leader position drop when i+1 reaches this step.

    total_time_steps = len(time_list)
    # Optional simulation-time cap for quick iteration/tuning; default 0 means no cap and runs the full velocity profile.
    _sim_maxtime = _env_float('KMPC_SIM_MAXTIME', 0.0)
    if ablation and ablation.repeated_run and _policy_path.exists():
        _policy_cap = float(json.loads(_policy_path.read_text()).get('simulation_max_time_s', 0))
        if _policy_cap > 0:
            _sim_maxtime = min(_sim_maxtime, _policy_cap) if _sim_maxtime > 0 else _policy_cap
    if _sim_maxtime > 0:
        _cap = int(round(_sim_maxtime / Tstep))
        if 0 < _cap < total_time_steps:
            time_list = time_list[:_cap]
            PV_vel = PV_vel[:_cap]
            total_time_steps = _cap
            print(f"[sim-cap] Truncated simulation to {_sim_maxtime:.1f}s ({total_time_steps} steps)")
    signal_delay_active = (
        signal_delay_requested
        and scenario == 'tracking'
        and control_method == 'Distributed MPC'
    )
    delay_candidates = _parse_hdv_delay_candidates(delay_targets_spec, sub_numHDV) if signal_delay_active else []
    delay_windows = _build_random_hdv_delay_windows(
        delay_count,
        delay_min_duration,
        delay_max_duration,
        total_time_steps,
        delay_candidates,
        delay_seed,
    ) if signal_delay_active else []
    delay_targets_by_step = _hdv_delay_targets_by_step(delay_windows)
    if signal_delay_requested and not signal_delay_active:
        print("[signal-delay] Current settings are not tracking + Distributed MPC; delay scenario is disabled.")
    elif signal_delay_active:
        printable_windows = [(s, e, p + 1, h) for s, e, p, h in delay_windows]
        print(f"[signal-delay] Enabled random HDV signal delay: seed={delay_seed}, windows={printable_windows}")

    cav_pred_loss_active = (
        cav_pred_loss_requested
        and control_method == 'Distributed MPC'
        and sim_subplatoon_num > 1
    )
    cav_pred_loss_receivers = _parse_cav_pred_loss_receivers(cav_pred_loss_receivers_spec, sim_subplatoon_num) if cav_pred_loss_active else []
    cav_pred_loss_windows = _build_random_cav_pred_loss_windows(
        cav_pred_loss_count,
        cav_pred_loss_min_duration,
        cav_pred_loss_max_duration,
        total_time_steps,
        cav_pred_loss_receivers,
        cav_pred_loss_seed,
    ) if cav_pred_loss_active else []
    cav_pred_loss_steps_by_receiver = _steps_by_receiver_from_windows(cav_pred_loss_windows)
    if cav_pred_loss_requested and not cav_pred_loss_active:
        print("[cav-pred-loss] Current settings are not Distributed MPC with at least two subplatoons; packet-loss scenario is disabled.")
    elif cav_pred_loss_active:
        printable_windows = [(s, e, r + 1) for s, e, r in cav_pred_loss_windows]
        print(f"[cav-pred-loss] Enabled random CAV prediction packet loss: seed={cav_pred_loss_seed}, windows={printable_windows}")

    # Load parameters from CSV in global vehicle order; number of rows must be >= sim_num_veh.
    print(f"Vehicle parameter file: {vehicle_csv}")
    param_data = np.genfromtxt(vehicle_csv, delimiter=',', skip_header=1)
    if param_data.shape[0] < sim_num_veh:
        raise ValueError(f"vehicle_parameters.csv has {param_data.shape[0]} rows < required vehicle count {sim_num_veh}")
    # Use only the first sim_num_veh rows in global vehicle order so parameter arrays match the number of following vehicles.
    Tgap = param_data[:sim_num_veh, 0]
    v0_IDM = param_data[:sim_num_veh, 1]
    veh_len = param_data[:sim_num_veh, 2]
    # Heterogeneous, time-invariant IDM parameters: maximum acceleration a, comfortable deceleration b, and minimum spacing s0.
    a_IDM = param_data[:sim_num_veh, 5]
    b_IDM = param_data[:sim_num_veh, 6]
    s0_IDM = param_data[:sim_num_veh, 7]
    if time_varying:
        Tgap_rate = param_data[:sim_num_veh, 3]
        v0_rate = param_data[:sim_num_veh, 4]
    else:
        Tgap_rate = np.zeros(sim_num_veh)
        v0_rate = np.zeros(sim_num_veh)


    Npst = args['Npst']
    # Constraint bounds and desired spacing are read near the top of the file in run_mpc.bat order.
    print(f"Constraint bounds: Spa[{SpaMin},{SpaMax}] Vel[{VelMin},{VelMax}] Acc[{AccMin},{AccMax}] Nfut={Nfut} sdes={SDES}")

    # ===== Save all key settings for this run to the result folder for comparison across settings. =====
    os.makedirs(output_dir, exist_ok=True)
    run_settings = {
        'timestamp':                 time.strftime('%Y-%m-%d %H:%M:%S'),
        'scenario':                  scenario,
        'control_method':            control_method,
        'disturbance_estimate':      disturbance_estimate,
        'disturbance_update_method': disturbance_update_method,
        'time_varying':              time_varying,
        'Nfut':                      Nfut,
        'Npst':                      Npst,
        'Tstep':                     Tstep,
        'SDES(desired/equilibrium spacing)': SDES,
        'Spa[min,max]':              [SpaMin, SpaMax],
        'Vel[min,max]':              [VelMin, VelMax],
        'Acc[min,max]':              [AccMin, AccMax],
        'lam':                       lam,
        'N_point':                   N_point,
        'corr_threads':              corr_threads,
        'ep_window':                 ep_window,
        'vehicle_csv':               vehicle_csv,
        'signal_delay':              signal_delay_requested,
        'delay_count':               delay_count,
        'delay_duration':            [delay_min_duration, delay_max_duration],
        'delay_seed':                delay_seed,
        'delay_hdv_candidates':      delay_targets_spec,
        'delay_windows':             [(s, e, p + 1, h) for s, e, p, h in delay_windows],
        'cav_pred_loss':             cav_pred_loss_requested,
        'cav_pred_loss_count':       cav_pred_loss_count,
        'cav_pred_loss_duration':    [cav_pred_loss_min_duration, cav_pred_loss_max_duration],
        'cav_pred_loss_seed':        cav_pred_loss_seed,
        'cav_pred_loss_receivers':   cav_pred_loss_receivers_spec,
        'cav_pred_loss_windows':     [(s, e, r + 1) for s, e, r in cav_pred_loss_windows],
        'enable_slack':              enable_slack,
        'save_slack':                save_slack,
        'slack_L1_weight':           W_SLACK,
        'slack_L2_weight':           W_SLACK_L2,
        'slack_record_tolerance':    SLACK_TOL,
        'verbose_step':              verbose_step,
        'save_plots':                save_plots,
        'mpc_weights':               mpc_weight_settings,
        'subplatoon_config':         [(c['num_HDV'], c['model_file']) for c in subplatoon_config],
        'result_root':               result_root,
        'run_suffix':                run_suffix,
        'output_dir':                output_dir,
    }
    if scenario == 'cutin':
        run_settings.update(cutin_time=cutin_time, cutin_gap_target=cutin_gap_target,
                            cutin_spacing_mode='fixed_after_cutin_gap',
                            cutin_brake_duration=cutin_brake_duration,
                            cutin_final_speed=cutin_final_speed,
                            cutin_brake_acceleration=(cutin_final_speed - 26.0) / cutin_brake_duration)
    if ablation:
        run_settings.update(uncertainty_mode=ablation.mode, freeze_step=ablation.freeze_step,
                            freeze_time_s=ablation.freeze_step*Tstep, added_prefix_steps=0,
                            nominal_and_global_active_from_step=0, timing_valid=ablation.timing_run,
                            coverage_evaluation=not ablation.timing_run)
    _settings_path = os.path.join(output_dir, 'run_settings.txt')
    with open(_settings_path, 'w', encoding='utf-8') as _f:
        for _k, _v in run_settings.items():
            _f.write(f"{_k}: {_v}\n")
    print(f"Run settings saved to '{_settings_path}'")

    # Compute equilibrium values from initial parameters and start the simulation from equilibrium.
    v_init = np.zeros(sim_num_veh)
    s_init = np.zeros(sim_num_veh)
    for i in range(sim_num_veh):
        v_init[i] = PV_vel[0]
        s_init[i] = helper.compute_IDM_steady_s(PV_vel[0], PV_vel[0], v0_IDM[i], Tgap[i], a_IDM[i], b_IDM[i], s0_IDM[i]) + veh_len[i] 

    # Per-subplatoon historical sequences with possibly different dimensions; x_history_seq_list[k]=(1,Npst,slice_dim,numHDV_k), x_seq_list[k]=(1,2*num_veh_k).
    S_history, x_history_seq_list, x_seq_list = helper.generate_history_seq(
        args, sim_subplatoon_num, sim_num_veh, sub_start, sub_numHDV,
        PV_vel[0], v_init, s_init, Tgap, v0_IDM, Tgap_rate, v0_rate, veh_len, Tstep, a_IDM, b_IDM, s0_IDM)

    # Each subsystem and prediction step maintains its own error set [c,G,alpha,v] for per-step online correction.
    # total_disturbance_set[p][k-1] corresponds to step k of subplatoon p; set to None when disturbance_estimate is disabled.
    if disturbance_estimate:
        total_disturbance_set = []
        omega_buffer = []   # omega_buffer[p][k-1]: sliding window with shape (n_high, N_point).
        for p in range(sim_subplatoon_num):
            mN = sub_bundle[p]['model_Nfut']
            total_disturbance_set.append([
                [sub_ctemplate[p][k].copy(), sub_Gtemplate[p][k].copy(), 1.0, np.zeros(sub_nhigh[p])]
                for k in range(mN)
            ])
            if disturbance_update_method == 'refit':
                omega_buffer.append([np.zeros((sub_nhigh[p], N_point)) for _ in range(mN)])
            else:
                omega_buffer.append(None)
    else:
        total_disturbance_set = [None for _ in range(sim_subplatoon_num)]

    # Prediction records: pred_records[p][target_time] = [(k, pred_vec), ...].
    # Used to compare the k-step-ahead prediction made k steps earlier for current time t against the true lifted state s_t, yielding real-time prediction error omega.
    # Feed this into the per-step online correction, replacing the old single-step total_g_pred_list.
    pred_records = [dict() for _ in range(sim_subplatoon_num)]

    # Define S to store simulation states.
    S = np.zeros([Npst+total_time_steps, sim_num_veh+1, 3]) # pos/vel/acc
    S[0:Npst+1,:,:] = S_history
    S[Npst:,0,1] = PV_vel
    # Compute PV acceleration.
    for i in range(Npst+total_time_steps-1):
        S[i,0,2] = (S[i+1,0,1]-S[i,0,1])/Tstep

    # x_next stores the previous optimal predicted state sequence for use as ep by the downstream subsystem at the next time step.
    x_next_list = []
    x_next_list_temp = []
    cav_pred_loss_cache = [None for _ in range(sim_subplatoon_num)]
    cav_pred_loss_records = []

    # ============================= Initialize VehicleSolver ========================
    sub_solver = []
    if control_method == 'Decentralized MPC':
        for k in range(sim_subplatoon_num):
            sub_solver.append(Decentralized_VehicleSolver(sub_G0[k], sub_Gu[k], sub_Ge[k], sub_D[k], SpaMax, SpaMin, VelMax, VelMin, AccMin, AccMax, sub_nhigh[k], sub_nlow[k], Nfut, Tstep, scenario))
    elif control_method == 'Distributed MPC':
        for k in range(sim_subplatoon_num):
            sub_solver.append(Distributed_VehicleSolver(sub_G0[k], sub_Gu[k], sub_Ge[k], sub_D[k], SpaMax, SpaMin, VelMax, VelMin, AccMin, AccMax, sub_nhigh[k], sub_nlow[k], Nfut, Tstep, scenario))

    # Convert x_history_seq_list and x_seq_list to tensors.
    if control_method != 'All HDVs':
        x_history_seq_tensor_list = []
        x_seq_tensor_list = []
        for platoon_idx in range(sim_subplatoon_num):
            x_history_seq_tensor_list.append(torch.from_numpy(x_history_seq_list[platoon_idx]).float().to(device))
            x_seq_tensor_list.append(torch.from_numpy(x_seq_list[platoon_idx]).float().to(device))

    # ============================= Simulation =============================================================
    update_compute_time_collection = np.zeros((sim_subplatoon_num, total_time_steps-1))
    control_compute_time_collection = np.zeros((sim_subplatoon_num, total_time_steps-1))
    learning_compute_time_collection = np.zeros_like(control_compute_time_collection)
    controller_compute_time_collection = np.zeros_like(control_compute_time_collection)

    # ===== Soft-constraint slack trigger log, optional via KMPC_SAVE_SLACK: log each relaxation above threshold per step and save nonempty logs after simulation. =====
    slack_records = []   # Each row: (subplatoon, step, constraint, horizon, value).
    def _record_slack(platoon_idx, step, slack_vals):
        if not save_slack or slack_vals is None:
            return
        for key in ('spa_lb', 'spa_ub', 'vel_lb', 'vel_ub'):
            for j, val in enumerate(slack_vals[key]):
                if val > SLACK_TOL:
                    slack_records.append((platoon_idx, step, key, j + 1, float(val)))
        if slack_vals['term'] > SLACK_TOL:
            slack_records.append((platoon_idx, step, 'term', Nfut, float(slack_vals['term'])))

    delay_records = []
    if disturbance_estimate:
        # alpha_collection: 2D array (subplatoon, time) storing the mean alpha over prediction steps for existing plot/save code.
        alpha_collection = np.zeros((sim_subplatoon_num, total_time_steps-1))
        # alpha_step_collection[p]: full per-step alpha with shape (model_Nfut, T-1), saved separately as CSV.
        alpha_step_collection = [np.ones((sub_bundle[k]['model_Nfut'], total_time_steps-1)) for k in range(sim_subplatoon_num)]
        tightened_constraint_collection = [np.zeros((total_time_steps - 1, Nfut*4)) for _ in range(sim_subplatoon_num)]

    def _update_step_set(platoon_idx, k, omega_new):
        """Apply online correction to the 1-based step-k template for one subplatoon.
        Operates on that step's template, buffer, and solver instances."""
        idx = k - 1
        if ablation and not ablation.should_update:
            return
        check_solver = sub_check_solver[platoon_idx][idx]
        online_solver = sub_online_solver[platoon_idx][idx]
        refit_solver  = sub_refit_solver[platoon_idx][idx]
        c_template = sub_ctemplate[platoon_idx][idx]
        G_template = sub_Gtemplate[platoon_idx][idx]
        n_high = c_template.shape[0]

        is_inside, _ = check_solver.solve(omega_new, c_template, G_template)
        if is_inside:
            if disturbance_update_method == 'refit':
                alpha_last_val = total_disturbance_set[platoon_idx][idx][2]
                v_last         = total_disturbance_set[platoon_idx][idx][3]
                c_last         = total_disturbance_set[platoon_idx][idx][0]
                G_last         = total_disturbance_set[platoon_idx][idx][1]

                omega_buffer[platoon_idx][idx] = np.roll(omega_buffer[platoon_idx][idx], -1, axis=1)
                omega_buffer[platoon_idx][idx][:, -1] = omega_new

                is_in_W_last, _ = check_solver.solve(omega_new, c_last, G_last)
                if is_in_W_last:
                    alpha_star, _ = refit_solver.solve(omega_buffer[platoon_idx][idx], v_last)
                    if alpha_star is not None:
                        alpha_new = lam * alpha_last_val + (1 - lam) * alpha_star
                        c_new = alpha_new * c_template + (1 - alpha_new) * v_last
                        G_new = alpha_new * G_template
                        total_disturbance_set[platoon_idx][idx] = [c_new, G_new, alpha_new, v_last]
                    else:
                        raise RuntimeError(f"Subplatoon {platoon_idx}, step {k}, refit solve failed.")
                else:
                    tau_new, y_new, _, _, _ = online_solver.solve(omega_new, alpha_last_val, v_last)
                    if tau_new is not None:
                        alpha_new = 1 - tau_new
                        vec_new   = y_new / tau_new
                        c_new     = alpha_new * c_template + tau_new * vec_new
                        G_new     = alpha_new * G_template
                        total_disturbance_set[platoon_idx][idx] = [c_new, G_new, alpha_new, vec_new]
                    else:
                        raise RuntimeError(f"Subplatoon {platoon_idx}, step {k}, expansion update failed.")
        else:
            # If the error point lies outside the template set, use the full template directly with alpha=1.
            total_disturbance_set[platoon_idx][idx] = [c_template, G_template, 1.0, np.zeros((n_high,))]

    # Online-correction thread pool: each (platoon, step) correction is independent and can run in parallel; persistent Gurobi solves release the GIL.
    # KMPC_CORR_THREADS=1 disables parallelism; the pool is created only when disturbance sets are built.
    _corr_threads = corr_threads
    _corr_pool = ThreadPoolExecutor(max_workers=_corr_threads) if (disturbance_estimate and _corr_threads > 1) else None

    def _correct_and_record(platoon_idx, s_t, i):
        """Recompute pure Koopman k-step open-loop predictions with realized inputs and record omega.
        pred_records[p][i] stores the lifted source state s_src for each source time t=i-k.
        The k-step prediction is recomputed with realized u/ep from S so omega contains only
        Koopman modeling error; ep error is handled by Eep, and replanning mismatch is excluded.
        Corrections for different horizons are independent, with separate templates, buffers,
        solver instances, and disjoint (p,k) write slots, so they can run in a thread pool.
        KMPC_CORR_THREADS=1 falls back to serial execution.
        s_t: lifted state at current time i, shape (n_high,).
        """
        if (not disturbance_estimate) or i == 0:
            return 0.0
        if ablation and (ablation.timing_run or ablation.repeated_run) and not ablation.should_update:
            pred_records[platoon_idx].pop(i, None)
            return 0.0
        t0 = time.time()
        realized = pred_records[platoon_idx].pop(i, [])   # [(k, s_src), ...], k=1..min(i,model_Nfut)
        lead_idx = sub_start[platoon_idx]; cav_idx = lead_idx + 1

        def _do_one(kp):
            k, s_src = kp                            # s_src: initial lifted state at source time t=i-k.
            t = i - k
            # Recompute the k-step open-loop prediction using the truly applied u and true leader speed ep, isolating the pure Koopman error from ep/control estimation errors.
            pred = sub_G0[platoon_idx][k-1] @ s_src
            for l in range(k):
                u_l  = S[Npst + t + l, cav_idx, 2]   # Actually applied CAV control.
                ep_l = S[Npst + t + l, lead_idx, 1]  # True leader speed.
                pred = pred + sub_Gu[platoon_idx][k-1][l] @ np.array([u_l]) \
                            + sub_Ge[platoon_idx][k-1][l] @ np.array([ep_l])
            omega_new = s_t - pred
            if ablation and not ablation.timing_run and not ablation.repeated_run:
                ablation.observe_error(platoon_idx, k, omega_new)
            _update_step_set(platoon_idx, k, omega_new)

        if _corr_pool is not None and len(realized) > 1:
            list(_corr_pool.map(_do_one, realized))
        else:
            for kp in realized:
                _do_one(kp)
        return time.time() - t0

    if ablation:
        ablation.start(globals())

    for i in range(total_time_steps-1):
        if ablation:
            ablation.before_step(i)
        Tgap_this_step = Tgap * (1 + Tgap_rate/100)**(i+Npst)
        v0_IDM_this_step = v0_IDM * (1 + v0_rate/100)**(i+Npst)
        # print(f"Step {i}: Tgap_this_step: {Tgap_this_step}, v0_IDM_this_step: {v0_IDM_this_step}")

        if control_method == 'All HDVs':
            acel = helper.IDM_dynamics(S[Npst+i,:,:],Tgap_this_step,v0_IDM_this_step,veh_len,a_IDM,b_IDM,s0_IDM)
            S[Npst+i,1:,2] = acel
            # Original next-step speed.
            v_next = S[Npst+i,1:,1] + Tstep * S[Npst+i,1:,2]
            # Prevent backward vehicle motion: clamp speed at 0 so hard braking stops the vehicle rather than reversing it, and adjust recorded acceleration accordingly.
            # Keep position, speed, and acceleration consistent; position uses the current speed, which is always nonnegative.
            v_next = np.maximum(v_next, 0.0)
            S[Npst+i,1:,2] = (v_next - S[Npst+i,1:,1]) / Tstep
            # Update position and speed.
            S[Npst+i+1,:,0] = S[Npst+i,:,0] + Tstep * S[Npst+i,:,1]
            S[Npst+i+1,1:,1] = v_next

            # ===== Inject cut-in event: the leader position shifts backward instantaneously, causing a sharp spacing reduction. =====
            # All-HDV baseline: next-step IDM reads directly from S, so injecting after the position update is enough for following vehicles to sense it.
            if scenario == 'cutin' and (i + 1) == cutin_step:
                cutin_applied_drop = _inject_cutin_position(S, Npst+i+1)
                print(f"[cut-in] Step {i+1}: leader position shifted backward by {cutin_applied_drop:.1f} m; CAV1-PV gap {S[Npst+i+1,0,0]-S[Npst+i+1,1,0]:.3f} m")

        elif control_method == 'Decentralized MPC':
            # Update HDV acceleration according to IDM.
            acel = helper.IDM_dynamics(S[Npst+i,:,:],Tgap_this_step,v0_IDM_this_step,veh_len,a_IDM,b_IDM,s0_IDM)
            S[Npst+i,1:,2] = acel

            # Apply MPC control separately to the CAV in each subplatoon.
            for platoon_idx in range(sim_subplatoon_num):
                # Select the model, dimensions, and global indices for this subplatoon under the heterogeneous layout.
                controller_t0 = time.perf_counter()
                model = sub_model[platoon_idx]
                c_gamma = sub_cgamma[platoon_idx]
                G_gamma = sub_Ggamma[platoon_idx]
                Ge = sub_Ge[platoon_idx]; D = sub_D[platoon_idx]
                lead_idx = sub_start[platoon_idx]      # Global leader index.
                cav_idx = lead_idx + 1                 # Global CAV index for this segment.

                x_t = x_seq_list[platoon_idx]
                g_t = model.encode(x_seq_tensor_list[platoon_idx],x_history_seq_tensor_list[platoon_idx])
                s_t = g_t.detach().cpu().numpy().squeeze()
                ep_t = S[Npst+i,lead_idx,1] # Decentralized case: ep_t is the speed of the vehicle immediately ahead of the subsystem CAV.

                # Correct each prediction-step template online using the difference between the past multi-step prediction for the current time and the real-time s_t.
                online_update_set_time = _correct_and_record(platoon_idx, s_t, i)
                learning_compute_time_collection[platoon_idx,i] = online_update_set_time

                # Record per-step alpha and step-mean alpha for existing plot/save code.
                if disturbance_estimate:
                    alphas = np.array([dset[2] for dset in total_disturbance_set[platoon_idx]])
                    alpha_step_collection[platoon_idx][:, i] = alphas
                    alpha_collection[platoon_idx, i] = float(np.mean(alphas))

                # ============ Current per-step omega template list after online correction, used for error reachable sets and constraint tightening.
                if disturbance_estimate:
                    c_omega_now = [total_disturbance_set[platoon_idx][k][0] for k in range(Nfut)]
                    G_omega_now = [total_disturbance_set[platoon_idx][k][1] for k in range(Nfut)]
                else:
                    c_omega_now = G_omega_now = None

                start_time = time.time()
                ep, Ex_interval = predict_future_ep_and_constraints(scenario, control_method, disturbance_estimate, c_omega_now, G_omega_now, c_gamma, G_gamma, Nfut, lead_idx, platoon_idx, ep_t, S, i, Tstep, Ge, D)
                preparation_time = time.time()-start_time + online_update_set_time

                # Save tightened constraints.
                if disturbance_estimate:
                    for k in range(Nfut):
                        tightened_constraint_collection[platoon_idx][i,k*4:(k+1)*4] = [
                            Ex_interval[k].left_limit[0],
                            Ex_interval[k].right_limit[0],
                            Ex_interval[k].left_limit[1],
                            Ex_interval[k].right_limit[1],
                        ]
                    # Print spacing/velocity shrinkage as the maximum interval width over the prediction horizon, with limited decimals for concise logs.
                    spa_contraction = max(Ex_interval[k].right_limit[0] - Ex_interval[k].left_limit[0] for k in range(Nfut))
                    vel_contraction = max(Ex_interval[k].right_limit[1] - Ex_interval[k].left_limit[1] for k in range(Nfut))
                    if verbose_step:
                        print(f"  [sub{platoon_idx} step{i}] max contraction spa={spa_contraction:.3f} m  vel={vel_contraction:.3f} m/s")

                # =========== Desired state.
                # Four desired states in the cost.
                sdes = SDES
                x_ref = np.zeros((sub_sdim[platoon_idx], Nfut+1))
                x_assume = np.zeros((sub_sdim[platoon_idx], Nfut+1))
                x_ref[:,0] = x_t
                for j in range(1,Nfut+1):
                    x_ref[0::2,j] = sdes
                    # x_ref[1::2,j] = ep[0,j]  # Current LV speed when disturbance_estimate=False; predicted LV speed when True.
                    x_ref[1::2,j] = ep_t # reference velocity = current measured LV velocity
                if i == 0:
                    x_assume[0::2,:] = sdes
                    x_assume[1::2,:] = x_t[0,1::2].reshape(-1,1)
                else:
                    x_assume = np.concatenate(
                        [x_next_list[platoon_idx][:,1:], x_next_list[platoon_idx][:,-1:]],
                        axis=1
                    )

                # ============= Solve.
                u_prev = np.array([S[Npst+i-1, cav_idx, 2]])   # Previously applied control for the delta-u penalty.
                x_next, s_next, u_next, compute_time, slack_vals = sub_solver[platoon_idx].solve(s_t, x_ref, x_assume, ep, Ex_interval, u_prev)
                controller_compute_time_collection[platoon_idx,i] = time.perf_counter() - controller_t0
                if verbose_step:
                    print(f"Subplatoon {platoon_idx}, step {i}: u_next: {u_next[0,0]}, total_time: {preparation_time + compute_time}")
                x_next_list_temp.append(x_next)
                update_compute_time_collection[platoon_idx,i] = preparation_time
                control_compute_time_collection[platoon_idx,i] = compute_time
                _record_slack(platoon_idx, i, slack_vals)
                S[Npst+i,cav_idx,2] = u_next[0,0]

                # ============= Record source-time initial state s_t for recomputing pure Koopman error with true inputs at t=i+j.
                if disturbance_estimate:
                    for j in range(1, Nfut+1):
                        pred_records[platoon_idx].setdefault(i+j, []).append((j, s_t.copy()))

            # Update position and speed.
            S[Npst+i+1,:,0] = S[Npst+i,:,0] + Tstep * S[Npst+i,:,1]
            S[Npst+i+1,1:,1] = S[Npst+i,1:,1] + Tstep * S[Npst+i,1:,2]

            # ===== Inject cut-in event: the leader position shifts backward instantaneously, causing a sharp CAV1-PV gap reduction. =====
            # Inject before recomputing x_seq so the CAV can sense the sudden spacing change at the next encoding step.
            if scenario == 'cutin' and (i + 1) == cutin_step:
                cutin_applied_drop = _inject_cutin_position(S, Npst+i+1)
                print(f"[cut-in] Step {i+1}: leader position shifted backward by {cutin_applied_drop:.1f} m; CAV1-PV gap {S[Npst+i+1,0,0]-S[Npst+i+1,1,0]:.3f} m")

            # Update x_seq and x_history_seq.
            for platoon_idx in range(sim_subplatoon_num):
                base = sub_start[platoon_idx]
                nv = sub_numveh[platoon_idx]
                x_new_slice = np.zeros([1,1,args['HV_hist_slice_dim'], sub_numHDV[platoon_idx]])
                for j in range(sub_numHDV[platoon_idx]):
                    x_new_slice[0,0,:,j] = x_seq_list[platoon_idx][0,(1+2*j):(4+2*j)]
                x_history_seq = np.concatenate(
                    (x_history_seq_list[platoon_idx][:,1:,:,:], x_new_slice),axis=1
                )
                x_history_seq_list[platoon_idx] = x_history_seq

                x_seq_list[platoon_idx][0,0::2] = S[Npst+i+1,base:base+nv,0] - S[Npst+i+1,base+1:base+nv+1,0]
                x_seq_list[platoon_idx][0,1::2] = S[Npst+i+1,base+1:base+nv+1,1]

                x_history_seq_tensor_list[platoon_idx] = torch.from_numpy(x_history_seq).float().to(device)
                x_seq_tensor_list[platoon_idx] = torch.from_numpy(x_seq_list[platoon_idx]).float().to(device)

            x_next_list = x_next_list_temp
            x_next_list_temp = []

        elif control_method == 'Distributed MPC':
            # Update HDV acceleration according to IDM.
            acel = helper.IDM_dynamics(S[Npst+i,:,:],Tgap_this_step,v0_IDM_this_step,veh_len,a_IDM,b_IDM,s0_IDM)
            S[Npst+i,1:,2] = acel

            # Apply MPC control separately to the CAV in each subplatoon.
            for platoon_idx in range(sim_subplatoon_num):
                # Select the model, dimensions, and global indices for this subplatoon under the heterogeneous layout.
                controller_t0 = time.perf_counter()
                model = sub_model[platoon_idx]
                c_gamma = sub_cgamma[platoon_idx]
                G_gamma = sub_Ggamma[platoon_idx]
                Ge = sub_Ge[platoon_idx]; D = sub_D[platoon_idx]
                lead_idx = sub_start[platoon_idx]      # Global leader index.
                cav_idx = lead_idx + 1                 # Global CAV index for this segment.

                x_t = x_seq_list[platoon_idx]
                x_seq_tensor_for_control = x_seq_tensor_list[platoon_idx]
                delay_targets_this_step = delay_targets_by_step.get(i, {}).get(platoon_idx, set()) if signal_delay_active else set()
                if delay_targets_this_step and i > 0:
                    actual_x_t = x_seq_list[platoon_idx].copy()
                    delayed_x_t = actual_x_t.copy()
                    predicted_current = x_next_list[platoon_idx][:, 1]
                    for hdv_idx in sorted(delay_targets_this_step):
                        pair_idx = hdv_idx  # Pair 0 is the CAV; HDV1 corresponds to pair 1.
                        state_slice = slice(2 * pair_idx, 2 * pair_idx + 2)
                        actual_pair = actual_x_t[0, state_slice].copy()
                        substitute_pair = predicted_current[state_slice].copy()
                        delayed_x_t[0, state_slice] = substitute_pair
                        delay_records.append([
                            i,
                            platoon_idx + 1,
                            hdv_idx,
                            actual_pair[0],
                            actual_pair[1],
                            substitute_pair[0],
                            substitute_pair[1],
                        ])
                        print(
                            f"[signal-delay] step {i}, subplatoon {platoon_idx + 1}, HDV{hdv_idx}: "
                            f"use predicted [s,v]=[{substitute_pair[0]:.3f}, {substitute_pair[1]:.3f}] "
                            f"instead of measured [{actual_pair[0]:.3f}, {actual_pair[1]:.3f}]"
                        )
                    # Temporarily replace the input signal only for the current control step; do not write it back to x_seq_list.
                    # Thus, after leaving the delay window, the history window returns to the true measured sequence.
                    x_t = delayed_x_t
                    x_seq_tensor_for_control = torch.from_numpy(delayed_x_t).float().to(device)

                # print(f"Subplatoon {platoon_idx}, step: {i}, x_t: {x_t.squeeze()}")
                g_t = model.encode(x_seq_tensor_for_control,x_history_seq_tensor_list[platoon_idx])
                s_t = g_t.detach().cpu().numpy().squeeze()
                lv_t = S[Npst+i,lead_idx,1]  # Speed of the direct leader (LV) of the CAV in this subplatoon.
                pv_t = S[Npst+i,0,1]         # PV speed of the full platoon.
                ep_t = lv_t                  # Koopman exogenous input ep always denotes the LV speed of this subplatoon.
                pv_seq = np.full((1, Nfut+1), pv_t)
                if disturbance_estimate:
                    a_pv_t, _, _ = helper.lv_acceleration_estimate(S, i, 0, scenario, Tstep, Npst)
                    for j in range(1, Nfut+1):
                        pv_seq[0,j] = max(0.0, pv_seq[0,j-1] + a_pv_t * Tstep)

                # Correct each prediction-step template online using the difference between the past multi-step prediction for the current time and the real-time s_t.
                online_update_set_time = _correct_and_record(platoon_idx, s_t, i)
                learning_compute_time_collection[platoon_idx,i] = online_update_set_time

                # Record per-step alpha and step-mean alpha for existing plot/save code.
                if disturbance_estimate:
                    alphas = np.array([dset[2] for dset in total_disturbance_set[platoon_idx]])
                    alpha_step_collection[platoon_idx][:, i] = alphas
                    alpha_collection[platoon_idx, i] = float(np.mean(alphas))

                # ============ Current per-step omega template list after online correction, used for error reachable sets and constraint tightening.
                if disturbance_estimate:
                    c_omega_now = [total_disturbance_set[platoon_idx][k][0] for k in range(Nfut)]
                    G_omega_now = [total_disturbance_set[platoon_idx][k][1] for k in range(Nfut)]
                else:
                    c_omega_now = G_omega_now = None

                start_time = time.time()
                ep, Ex_interval= predict_future_ep_and_constraints(scenario, control_method, disturbance_estimate, c_omega_now, G_omega_now, c_gamma, G_gamma, Nfut, lead_idx, platoon_idx, ep_t, S, i, Tstep, Ge, D)
                upstream_prediction_for_refs = None
                if platoon_idx != 0 and i != 0:
                    packet_lost = cav_pred_loss_active and i in cav_pred_loss_steps_by_receiver.get(platoon_idx, set())
                    if packet_lost:
                        cache_bootstrap = cav_pred_loss_cache[platoon_idx] is None
                        if cav_pred_loss_cache[platoon_idx] is None:
                            # Bootstrap only if the first attempted upstream packet is lost before any cached prediction exists.
                            cav_pred_loss_cache[platoon_idx] = x_next_list[platoon_idx-1].copy()
                        upstream_prediction_for_refs = _shift_prediction_once(cav_pred_loss_cache[platoon_idx])
                        cav_pred_loss_cache[platoon_idx] = upstream_prediction_for_refs.copy()
                        cav_pred_loss_records.append([
                            i,
                            platoon_idx + 1,
                            platoon_idx,
                            int(cache_bootstrap),
                        ])
                        if verbose_step:
                            print(
                                f"[cav-pred-loss] step {i}, subplatoon {platoon_idx + 1}: "
                                f"use shifted cached optimal prediction from CAV{platoon_idx}"
                            )
                    else:
                        cav_pred_loss_cache[platoon_idx] = x_next_list[platoon_idx-1].copy()
                        upstream_prediction_for_refs = _shift_prediction_once(x_next_list[platoon_idx-1])
                    ep = upstream_prediction_for_refs[-1:,:] # ep is the predicted speed of the vehicle ahead of the subsystem CAV.
                preparation_time = time.time() - start_time + online_update_set_time

                # Save tightened constraints.
                if disturbance_estimate:
                    for k in range(Nfut):
                        tightened_constraint_collection[platoon_idx][i,k*4:(k+1)*4] = [
                            Ex_interval[k].left_limit[0],
                            Ex_interval[k].right_limit[0],
                            Ex_interval[k].left_limit[1],
                            Ex_interval[k].right_limit[1],
                        ]
                    # Print spacing/velocity shrinkage as the maximum interval width over the prediction horizon, with limited decimals for concise logs.
                    spa_contraction = max(Ex_interval[k].right_limit[0] - Ex_interval[k].left_limit[0] for k in range(Nfut))
                    vel_contraction = max(Ex_interval[k].right_limit[1] - Ex_interval[k].left_limit[1] for k in range(Nfut))
                    if verbose_step:
                        print(f"  [sub{platoon_idx} step{i}] max contraction spa={spa_contraction:.3f} m  vel={vel_contraction:.3f} m/s")

                # =========== Desired state.
                # Four desired states in the cost.
                sdes = SDES
                x_pv = np.zeros((sub_sdim[platoon_idx], Nfut+1))
                x_preceed = np.zeros((sub_sdim[platoon_idx], Nfut+1))
                x_assume = np.zeros((sub_sdim[platoon_idx], Nfut+1))
                x_ref = np.zeros((sub_sdim[platoon_idx], Nfut+1))

                # x_assume is the previous optimal predicted state sequence; the current sequence should stay close to it, for velocity only.
                if i == 0:
                    # Set all x_assume velocities to the current value because no previous time step is available.
                    x_assume[0::2,:] = sdes
                    x_assume[1::2,:] = x_t[0,1::2].reshape(-1,1)
                    # x_pv velocity is the PV speed sequence of the full platoon, sent by CAV1 to other vehicles.
                    x_pv[:,0] = x_t
                    x_pv[0::2,1:] = sdes
                    x_pv[1::2,1:] = pv_seq[0,1:]
                    # Set x_preceed velocities from the shifted previous optimal predicted velocity sequence of CAV i-1; use current values when no previous step is available.
                    x_preceed[0::2,:] = sdes
                    if platoon_idx == 0:
                        # x_preceed[1::2,:] = ep  # ep is the predicted leader speed.
                        x_preceed[1::2,:] = ep_t # ep_t is the current leader speed.
                    elif platoon_idx != 0:
                        x_preceed[1::2,:] = S[Npst+i,sub_start[platoon_idx-1]+1,1]   # Previous segment CAV speed, scalar.
                    # Set x_ref velocity to the LV i speed sequence from the previous optimal predicted sequence; use the current speed at the first step.
                    x_ref[0::2,:] = sdes
                    if platoon_idx == 0:
                        # x_ref[1::2,:] = ep
                        x_ref[1::2,:] = ep_t
                    elif platoon_idx != 0:
                        x_ref[1::2,:] = S[Npst+i,lead_idx,1]   # Current segment leader speed, scalar.
                else:
                    # x_assume velocity is the previous optimal predicted velocity sequence, encouraging adjacent optimization sequences to stay close.
                    x_assume[0::2,:]=sdes
                    x_assume[1::2,:]=np.concatenate([x_next_list[platoon_idx][:,1:], x_next_list[platoon_idx][:,-1:]], axis=1)[1::2,:]
                    # x_pv velocity is the PV speed sequence of the full platoon, sent by CAV1 to other vehicles.
                    x_pv[:,0] = x_t
                    x_pv[0::2,1:] = sdes
                    x_pv[1::2,1:] = pv_seq[0,1:]
                    # x_preceed velocity is the shifted previous optimal predicted velocity sequence of the previous subplatoon CAV; for the first subplatoon, it is generated from ep_t.
                    x_preceed[0::2,:] = sdes
                    if platoon_idx == 0:
                        # x_preceed[1::2,:] = ep
                        x_preceed[1::2,:] = ep_t
                    elif platoon_idx != 0: 
                        x_preceed[1::2,:] = np.tile(upstream_prediction_for_refs[1:2,:], (sub_numveh[platoon_idx], 1))
                    # Set x_ref velocity to the speed sequence of the vehicle ahead of the current subplatoon CAV, taken from the previous optimal predicted sequence.
                    x_ref[0::2,:] = sdes
                    if platoon_idx == 0:
                        # x_ref[1::2,:] = ep
                        x_ref[1::2,:] = ep_t
                    elif platoon_idx != 0:
                        x_ref[1::2,:] = np.tile(upstream_prediction_for_refs[-1:,:], (sub_numveh[platoon_idx], 1))


                # ============= Solve.
                u_prev = np.array([S[Npst+i-1, cav_idx, 2]])   # Previously applied control for the delta-u penalty.
                x_next, s_next, u_next, compute_time, slack_vals = sub_solver[platoon_idx].solve(s_t, x_pv, x_preceed, x_assume, x_ref, ep, Ex_interval, u_prev)
                controller_compute_time_collection[platoon_idx,i] = time.perf_counter() - controller_t0
                alpha_mean = float(np.mean([dset[2] for dset in total_disturbance_set[platoon_idx]])) if disturbance_estimate else 'N/A'
                if verbose_step:
                    print(f"Subplatoon {platoon_idx}, step {i}: u_next: {u_next[0,0]}, total_consume_time: {compute_time+preparation_time}, alpha(mean): {alpha_mean}")
                x_next_list_temp.append(x_next)
                update_compute_time_collection[platoon_idx,i] = preparation_time
                control_compute_time_collection[platoon_idx,i] = compute_time
                _record_slack(platoon_idx, i, slack_vals)
                S[Npst+i,cav_idx,2] = u_next[0,0]

                # ============= Record source-time initial state s_t for recomputing pure Koopman error with true inputs at t=i+j.
                if disturbance_estimate:
                    for j in range(1, Nfut+1):
                        pred_records[platoon_idx].setdefault(i+j, []).append((j, s_t.copy()))

            # Update position and speed.
            S[Npst+i+1,:,0] = S[Npst+i,:,0] + Tstep * S[Npst+i,:,1]
            S[Npst+i+1,1:,1] = S[Npst+i,1:,1] + Tstep * S[Npst+i,1:,2]

            # ===== Inject cut-in event: the leader position shifts backward instantaneously, causing a sharp CAV1-PV gap reduction. =====
            # Inject before recomputing x_seq so the CAV can sense the sudden spacing change at the next encoding step.
            if scenario == 'cutin' and (i + 1) == cutin_step:
                cutin_applied_drop = _inject_cutin_position(S, Npst+i+1)
                print(f"[cut-in] Step {i+1}: leader position shifted backward by {cutin_applied_drop:.1f} m; CAV1-PV gap {S[Npst+i+1,0,0]-S[Npst+i+1,1,0]:.3f} m")

            # Update x_seq and x_history_seq.
            for platoon_idx in range(sim_subplatoon_num):
                base = sub_start[platoon_idx]
                nv = sub_numveh[platoon_idx]
                x_new_slice = np.zeros([1,1,args['HV_hist_slice_dim'], sub_numHDV[platoon_idx]])
                for j in range(sub_numHDV[platoon_idx]):
                    x_new_slice[0,0,:,j] = x_seq_list[platoon_idx][0,(1+2*j):(4+2*j)]
                x_history_seq = np.concatenate((x_history_seq_list[platoon_idx][:,1:,:,:], x_new_slice),axis=1)
                x_history_seq_list[platoon_idx] = x_history_seq

                x_seq_list[platoon_idx][0,0::2] = S[Npst+i+1,base:base+nv,0] - S[Npst+i+1,base+1:base+nv+1,0]
                x_seq_list[platoon_idx][0,1::2] = S[Npst+i+1,base+1:base+nv+1,1]

                x_history_seq_tensor_list[platoon_idx] = torch.from_numpy(x_history_seq).float().to(device)
                x_seq_tensor_list[platoon_idx] = torch.from_numpy(x_seq_list[platoon_idx]).float().to(device)

            x_next_list = x_next_list_temp
            x_next_list_temp=[]

    if ablation:
        ablation.finish(completed=True)
        raise SystemExit(0)

    print(f"Result output directory: {output_dir}")
    if save_plots:
        plot(args, scenario, S, Tstep, sim_subplatoon_num, sub_numHDV, out_dir=output_dir)

    # Save the simulation results
    save_vehicle_data_to_csv(S, filename_prefix=os.path.join(output_dir, 'vehicle_data'))

    # ===== Metrics: RMSVE/RMSSE excluding the PV and evaluating only the control segment S[Npst:] after IDM warm-up, plus mean MPC solve time. =====
    metrics = compute_tracking_metrics(S[Npst:], scenario, sdes=SDES)
    metrics.update(mpc_weight_settings)
    if control_method != 'All HDVs':
        metrics['AvgControlTime_s'] = float(np.mean(control_compute_time_collection))
        metrics['AvgUpdateTime_s']  = float(np.mean(update_compute_time_collection))
        metrics['AvgTotalTime_s']   = float(np.mean(control_compute_time_collection + update_compute_time_collection))
        metrics['SignalDelayEnabled'] = int(signal_delay_active)
        metrics['SignalDelayEvents'] = int(len(delay_records))
        metrics['SignalDelayWindows'] = int(len(delay_windows))
        metrics['CAVPredLossEnabled'] = int(cav_pred_loss_active)
        metrics['CAVPredLossEvents'] = int(len(cav_pred_loss_records))
        metrics['SlackEnabled'] = int(enable_slack)

    # ===== Optional soft-constraint slack via KMPC_SAVE_SLACK: save details to the result folder and write metrics summary if slack is triggered. =====
    if enable_slack and save_slack and control_method != 'All HDVs' and not ablation:
        if slack_records:
            slack_csv = os.path.join(output_dir, 'slack_records.csv')
            with open(slack_csv, 'w', newline='') as f:
                f.write('subplatoon,step,constraint,horizon,slack\n')
                for r in slack_records:
                    f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.6g}\n")
            slack_max = max(r[4] for r in slack_records)
            metrics['SlackTriggered'] = 1
            metrics['SlackEntries'] = len(slack_records)
            metrics['SlackMax'] = float(slack_max)
            print(f"[slack] Soft-constraint slack triggered {len(slack_records)} entries, max {slack_max:.4g}; saved to {slack_csv}")
        else:
            metrics['SlackTriggered'] = 0
            metrics['SlackEntries'] = 0
            metrics['SlackMax'] = 0.0
            print("[slack] No soft-constraint slack was triggered.")

    print("==================== Evaluation Metrics ====================")
    print(f"  RMSVE = {metrics['RMSVE']:.4f} m/s   RMSSE = {metrics['RMSSE']:.4f} m")
    if control_method != 'All HDVs':
        print(f"  Avg control solve time = {metrics['AvgControlTime_s']*1000:.3f} ms | "
              f"Avg set update = {metrics['AvgUpdateTime_s']*1000:.3f} ms | "
              f"Avg total = {metrics['AvgTotalTime_s']*1000:.3f} ms")
    print("=================================================")
    if ablation:
        metrics.update(ablation.metrics)
    save_metrics(metrics, filename=os.path.join(output_dir, 'metrics.csv'))

    # Save the computation time collection
    if control_method != 'All HDVs':
        save_computation_time(update_compute_time_collection,"update", out_dir=output_dir)
        save_computation_time(control_compute_time_collection,"control", out_dir=output_dir)
        if signal_delay_active:
            delay_windows_csv = os.path.join(output_dir, 'signal_delay_windows.csv')
            delay_windows_arr = (np.array([
                [
                    start,
                    start * Tstep,
                    end,
                    end * Tstep,
                    end - start + 1,
                    (end - start + 1) * Tstep,
                    sub_idx + 1,
                    hdv_idx,
                    sub_start[sub_idx] + hdv_idx + 1,
                ]
                for start, end, sub_idx, hdv_idx in delay_windows
            ], dtype=float) if delay_windows else np.empty((0, 9)))
            np.savetxt(
                delay_windows_csv,
                delay_windows_arr,
                delimiter=',',
                fmt=['%d', '%.6f', '%d', '%.6f', '%d', '%.6f', '%d', '%d', '%d'],
                header='start_step,start_time_s,end_step,end_time_s,duration_steps,duration_s,subplatoon,hdv_idx,global_vehicle_idx',
                comments='',
            )
            delay_csv = os.path.join(output_dir, 'signal_delay_records.csv')
            delay_arr = (np.array(delay_records, dtype=float)
                         if delay_records else np.empty((0, 7)))
            np.savetxt(
                delay_csv,
                delay_arr,
                delimiter=',',
                header='step,subplatoon,hdv_idx,actual_spacing,actual_velocity,substitute_spacing,substitute_velocity',
                comments='',
            )
            print(f"Signal-delay random windows saved to '{delay_windows_csv}' ({len(delay_windows)} windows)")
            print(f"Signal-delay substitution records saved to '{delay_csv}' ({len(delay_records)} entries)")
        if cav_pred_loss_active:
            loss_windows_csv = os.path.join(output_dir, 'cav_prediction_loss_windows.csv')
            loss_windows_arr = (np.array([(s, e, r + 1, r) for s, e, r in cav_pred_loss_windows], dtype=int)
                                if cav_pred_loss_windows else np.empty((0, 4), dtype=int))
            np.savetxt(
                loss_windows_csv,
                loss_windows_arr,
                delimiter=',',
                fmt='%d',
                header='start_step,end_step,receiver_subplatoon,source_cav_subplatoon',
                comments='',
            )
            loss_records_csv = os.path.join(output_dir, 'cav_prediction_loss_records.csv')
            loss_records_arr = (np.array(cav_pred_loss_records, dtype=int)
                                if cav_pred_loss_records else np.empty((0, 4), dtype=int))
            np.savetxt(
                loss_records_csv,
                loss_records_arr,
                delimiter=',',
                fmt='%d',
                header='step,receiver_subplatoon,source_cav_subplatoon,cache_bootstrap',
                comments='',
            )
            print(f"CAV prediction packet-loss records saved to '{loss_records_csv}' ({len(cav_pred_loss_records)} entries)")
        if disturbance_estimate:
            save_disturbance_set_data(alpha_collection, tightened_constraint_collection, out_dir=output_dir)
            if save_plots:
                plot_alpha_evolution(alpha_collection, Tstep, out_dir=output_dir)
            # Full per-step alpha records, one row per prediction step and one column per time step.
            ds_dir = os.path.join(output_dir, 'disturbance_set')
            os.makedirs(ds_dir, exist_ok=True)
            for platoon_idx in range(sim_subplatoon_num):
                np.savetxt(os.path.join(ds_dir, f'subplatoon_{platoon_idx}_alpha_per_step.csv'),
                           alpha_step_collection[platoon_idx], delimiter=',')
            print(f"Per-step alpha saved to '{ds_dir}/'")
