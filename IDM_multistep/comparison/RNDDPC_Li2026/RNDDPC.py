"""Closed-loop RNDDPC-HI simulation; launch with run_rnddpc.py.

History at time t ends at t-1, consistent with offline excitation data.
All physical CAV bounds are hard; no terminal-spacing penalty or slack.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import time
import numpy as np
import functions as f
from functions import helper, scenario, output_dir, Tstep, Nfut, SpaMax, SpaMin, VelMax, VelMin, AccMax, AccMin, sdes, EPS_MAX, EP_EXTRAPOLATE, WITH_HISTORY_MODEL_PATHS, subplatoon_spec, vehicle_csv_path, rnddpc_q_spa, rnddpc_q_vel, rnddpc_r, rnddpc_rd, CONIC_SOLVER, MZ_KEEP_GENERATORS, build_rnddpc_bundle, IDM_dynamics, compute_IDM_steady_s, generate_history_seq, plot, compute_tracking_metrics, save_vehicle_data_to_csv, save_metrics, save_computation_time
from solver import RNDDPCSolver
import json
from pathlib import Path
time_varying = True

def save_run_settings(path, settings):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fobj:
        for (key, value) in settings.items():
            fobj.write(f'{key}: {value}\n')
    print(f"Run settings saved to '{path}'")


def solve_local(solver, inputs, states, past_steps, step, subsystem, preparation_time):
    """Record accepted solves, or save the precise state/input at a failure."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        result = solver.solve(*inputs)
        row = {'step': step, 'subsystem': subsystem, 'status': solver.prob.status,
               'projection': solver.last_projection, 'control': result[2][0].tolist(),
               'control_time_s': result[3], 'solver_core_time_s': result[4],
               'preparation_time_s': preparation_time,
               'total_time_s': preparation_time + result[3]}
        with (directory / 'solve_status.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        return result
    except Exception as error:
        row = {'step': step, 'time_s': step*Tstep, 'subsystem': subsystem,
               'subsystem_one_based': subsystem+1, 'message': str(error),
               'status': solver.prob.status, 'projection': solver.last_projection}
        (directory / 'failure.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        np.savez_compressed(directory / 'failure_inputs.npz', **dict(zip(
            ['z_t','x0','x_ref','ep','u_prev'], inputs)))
        np.savez_compressed(directory / 'partial_trajectory.npz',
                           S=states[:past_steps+step+1], Npst=past_steps, Tstep=Tstep)
        raise

def main():
    print(f'[RNDDPC] scenario={scenario}  control=RNDDPC-HI  -> {output_dir}')
    subplatoon_numHDV = list(subplatoon_spec)
    available_models = WITH_HISTORY_MODEL_PATHS
    for nh in subplatoon_numHDV:
        if nh not in available_models:
            mode = 'history-informed'
            raise ValueError(f'RNDDPC has no {mode} encoder for num_HDV={nh}; available={sorted(available_models)}')
    num_CAV = 1
    sim_subplatoon_num = len(subplatoon_numHDV)
    sub_numveh = [num_CAV + nh for nh in subplatoon_numHDV]
    sub_start = [0] * sim_subplatoon_num
    for k in range(1, sim_subplatoon_num):
        sub_start[k] = sub_start[k - 1] + sub_numveh[k - 1]
    sim_num_veh = sum(sub_numveh)
    print(f"[RNDDPC] layout {['1C%dH' % h for h in subplatoon_numHDV]}, total veh={sim_num_veh}, start={sub_start}")
    if scenario == 'tracking':
        (time_list, PV_vel) = helper.generate_NEDC_velocity_profile(Tstep)
    elif scenario == 'braking':
        (time_list, PV_vel) = helper.generate_braking_velocity_profile(Tstep)
    else:
        raise ValueError(f'unknown scenario {scenario}')
    total_time_steps = len(time_list)
    param_data = f.load_vehicle_parameters(sim_num_veh)
    Tgap = param_data[:sim_num_veh, 0]
    v0_IDM = param_data[:sim_num_veh, 1]
    veh_len = param_data[:sim_num_veh, 2]
    a_IDM = param_data[:sim_num_veh, 5]
    b_IDM = param_data[:sim_num_veh, 6]
    s0_IDM = param_data[:sim_num_veh, 7]
    if time_varying:
        Tgap_rate = param_data[:sim_num_veh, 3]
        v0_rate = param_data[:sim_num_veh, 4]
    else:
        Tgap_rate = np.zeros(sim_num_veh)
        v0_rate = np.zeros(sim_num_veh)
    sub_bundle = []
    for (k, nh) in enumerate(subplatoon_numHDV):
        base = sub_start[k]
        nv = sub_numveh[k]
        idm_params = (Tgap[base:base + nv], v0_IDM[base:base + nv], veh_len[base:base + nv], a_IDM[base:base + nv], b_IDM[base:base + nv], s0_IDM[base:base + nv])
        sub_bundle.append(build_rnddpc_bundle(nh, idm_params=idm_params, bundle_label=f'sub{k}'))
    args = vars(helper.get_args_local(False))
    Npst = args['Npst']
    v_init = np.full(sim_num_veh, PV_vel[0])
    s_init = np.zeros(sim_num_veh)
    for i in range(sim_num_veh):
        s_init[i] = compute_IDM_steady_s(PV_vel[0], PV_vel[0], v0_IDM[i], Tgap[i], a_IDM[i], b_IDM[i], s0_IDM[i]) + veh_len[i]
    (S_history, x_history_seq_list, x_seq_list) = generate_history_seq(args, sim_subplatoon_num, sim_num_veh, sub_start, subplatoon_numHDV, PV_vel[0], v_init, s_init, Tgap, v0_IDM, Tgap_rate, v0_rate, veh_len, Tstep, a_IDM, b_IDM, s0_IDM)
    S = np.zeros([Npst + total_time_steps, sim_num_veh + 1, 3])
    S[0:Npst + 1, :, :] = S_history
    S[Npst:, 0, 1] = PV_vel
    for i in range(Npst + total_time_steps - 1):
        S[i, 0, 2] = (S[i + 1, 0, 1] - S[i, 0, 1]) / Tstep
    sub_solver = [RNDDPCSolver(b, Nfut, scenario) for b in sub_bundle]
    last_u = [np.zeros(1) for _ in range(sim_subplatoon_num)]
    run_settings = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'scenario': scenario,
        'control_method': 'RNDDPC-HI',
        'Nfut': Nfut,
        'SDES': sdes,
        'Spa[min,max]': [SpaMin, SpaMax],
        'Vel[min,max]': [VelMin, VelMax],
        'Acc[min,max]': [AccMin, AccMax],
        'subplatoon_config': subplatoon_numHDV,
        'vehicle_csv': vehicle_csv_path,
        'time_varying': time_varying,
        'weights': {'spacing': rnddpc_q_spa, 'velocity': rnddpc_q_vel, 'control': rnddpc_r, 'delta_u_penalty': rnddpc_rd},
        'koopman_lifting': {'with_history': True, 'checkpoints': [b['model_ref'] for b in sub_bundle] if sub_bundle else []},
        'constraints': {'spacing': 'hard CAV reachable-interval bounds', 'velocity': 'hard CAV reachable-interval bounds', 'terminal_velocity': 'hard equality on nominal terminal velocity; target projected when needed', 'acceleration': [AccMin, AccMax]},
        'rnddpc_uncertainty': {'EPS_MAX': f.EPS_MAX, 'Q_COVER': f.Q_COVER, 'residual_scale': 1.0, 'RCOND_PINV': f.RCOND_PINV, 'EP_EXTRAPOLATE': EP_EXTRAPOLATE, 'uncertainty_propagation': 'generator-preserving matrix zonotope', 'CONIC_SOLVER': CONIC_SOLVER, 'MZ_KEEP_GENERATORS': MZ_KEEP_GENERATORS, 'MZ_GENERATOR_COUNTS': [b['mz_generator_counts'] for b in sub_bundle] if sub_bundle else []},
        'data_collection': {'rollouts': f.DATA_ROLLOUTS, 'rollout_len': f.DATA_ROLLOUT_LEN, 'noise': f.DATA_NOISE, 'seed': f.DATA_SEED, 'vbase_range': [f.DATA_VBASE_MIN, f.DATA_VBASE_MAX], 'head_velocity_range': [f.DATA_HEAD_MIN, f.DATA_HEAD_MAX], 'head_random_walk_std': f.DATA_HEAD_STD, 'cav_pe_std': f.DATA_PE_STD, 'idm_params': 'vehicle_parameters.csv slices per subplatoon'},
        'output_dir': output_dir,
    }
    save_run_settings(os.path.join(output_dir, 'run_settings.txt'), run_settings)
    control_time = np.zeros((sim_subplatoon_num, total_time_steps - 1))
    solver_core_time = np.zeros((sim_subplatoon_num, total_time_steps - 1))
    total_control_time = np.zeros_like(control_time)
    spa_contraction_log = np.zeros((sim_subplatoon_num, total_time_steps - 1))
    vel_contraction_log = np.zeros((sim_subplatoon_num, total_time_steps - 1))
    for i in range(total_time_steps - 1):
        Tgap_t = Tgap * (1 + Tgap_rate / 100) ** (i + Npst)
        v0_t = v0_IDM * (1 + v0_rate / 100) ** (i + Npst)
        acel = IDM_dynamics(S[Npst + i, :, :], Tgap_t, v0_t, veh_len, a_IDM, b_IDM, s0_IDM)
        S[Npst + i, 1:, 2] = acel
        for k in range(sim_subplatoon_num):
            b = sub_bundle[k]
            base = sub_start[k]
            nv = sub_numveh[k]
            lead_idx = base
            cav_idx = base + 1
            preparation_started = time.perf_counter()
            x_t = x_seq_list[k][0].copy()
            z_t = b['model'].lift(x_t, x_history_seq_list[k]).squeeze()
            ep_t = float(S[Npst + i, lead_idx, 1])
            x_ref = np.zeros((b['s_dim'], Nfut + 1))
            x_ref[:, 0] = x_t
            x_ref[0::2, 1:] = sdes
            x_ref[1::2, 1:] = ep_t
            if EP_EXTRAPOLATE:
                aPV_t = float(S[Npst + i, lead_idx, 2])
                ep_seq = np.clip(ep_t + aPV_t * Tstep * np.arange(Nfut + 1), 0.0, VelMax)[None, :]
            else:
                ep_seq = np.full((1, Nfut + 1), ep_t)
            preparation_time = time.perf_counter() - preparation_started
            (x_pred, z_pred, u_pred, ct, st, tube_halfwidth) = solve_local(sub_solver[k], (z_t, x_t, x_ref, ep_seq, last_u[k]), S, Npst, i, k, preparation_time)
            control_time[k, i] = ct
            total_control_time[k, i] = ct + preparation_time
            solver_core_time[k, i] = st
            spa_contraction_log[k, i] = float(np.max(2.0 * tube_halfwidth[0, :]))
            vel_contraction_log[k, i] = float(np.max(2.0 * tube_halfwidth[1, :]))
            u_apply = float(u_pred[0, 0])
            last_u[k] = np.array([u_apply])
            S[Npst + i, cav_idx, 2] = u_apply
            if i % 50 == 0:
                print(f'  sub{k} step{i}: u={u_pred[0, 0]:+.3f}  tube(spa={spa_contraction_log[k, i]:.3f}m vel={vel_contraction_log[k, i]:.3f}) total={ct * 1000:.1f}ms solver={st * 1000:.1f}ms', flush=True)
        S[Npst + i + 1, :, 0] = S[Npst + i, :, 0] + Tstep * S[Npst + i, :, 1]
        S[Npst + i + 1, 1:, 1] = S[Npst + i, 1:, 1] + Tstep * S[Npst + i, 1:, 2]
        for k in range(sim_subplatoon_num):
            base = sub_start[k]
            nv = sub_numveh[k]
            numHDV_k = subplatoon_numHDV[k]
            slice_dim = args['HV_hist_slice_dim']
            x_new_slice = np.zeros((1, 1, slice_dim, numHDV_k))
            for j in range(numHDV_k):
                x_new_slice[0, 0, 0, j] = S[Npst + i, base + j + 1, 1]
                x_new_slice[0, 0, 1, j] = S[Npst + i, base + j + 1, 0] - S[Npst + i, base + j + 2, 0]
                x_new_slice[0, 0, 2, j] = S[Npst + i, base + j + 2, 1]
            x_history_seq_list[k] = np.concatenate((x_history_seq_list[k][:, 1:, :, :], x_new_slice), axis=1)
            x_seq_list[k][0, 0::2] = S[Npst + i + 1, base:base + nv, 0] - S[Npst + i + 1, base + 1:base + nv + 1, 0]
            x_seq_list[k][0, 1::2] = S[Npst + i + 1, base + 1:base + nv + 1, 1]
    print(f'[RNDDPC] writing results to {output_dir}')
    plot(args, scenario, S, Tstep, sim_subplatoon_num, subplatoon_numHDV, out_dir=output_dir)
    save_vehicle_data_to_csv(S, filename_prefix=os.path.join(output_dir, 'vehicle_data'))
    metrics = compute_tracking_metrics(S[Npst:], scenario, sdes=sdes)
    metrics['AvgControlTime_s'] = float(np.mean(control_time))
    metrics['AvgSolverCoreTime_s'] = float(np.nanmean(solver_core_time))
    metrics['AvgNonSolverTime_s'] = float(np.nanmean(control_time - solver_core_time))
    metrics['AvgTotalTime_s'] = float(np.mean(total_control_time))
    metrics['AvgStepControlTime_s'] = float(np.mean(np.sum(total_control_time, axis=0)))
    metrics['P95TotalTime_s'] = float(np.percentile(total_control_time, 95))
    metrics['MaxTotalTime_s'] = float(np.max(total_control_time))
    metrics['ControlUpdates'] = int(control_time.size)
    metrics['weight_s'] = rnddpc_q_spa
    metrics['weight_v'] = rnddpc_q_vel
    metrics['weight_u'] = rnddpc_r
    metrics['delta_u_penalty'] = rnddpc_rd
    metrics['velocity_constraints'] = True
    metrics['terminal_velocity_constraint'] = True
    metrics['SlackEnabled'] = 0
    metrics['ConicSolver'] = CONIC_SOLVER
    print('==================== metrics ====================')
    print(f"  RMSVE = {metrics['RMSVE']:.4f} m/s   RMSSE = {metrics['RMSSE']:.4f} m")
    solve_label = 'matrix-zonotope RPC'
    print(f"  {solve_label} = {metrics['AvgControlTime_s'] * 1000:.2f} ms | solver core = {metrics['AvgSolverCoreTime_s'] * 1000:.2f} ms | non-solver work = {metrics['AvgNonSolverTime_s'] * 1000:.2f} ms | total = {metrics['AvgTotalTime_s'] * 1000:.2f} ms")
    print('=================================================')
    save_metrics(metrics, filename=os.path.join(output_dir, 'metrics.csv'))
    save_computation_time(control_time, 'control', out_dir=output_dir)
    save_computation_time(solver_core_time, 'solver_core', out_dir=output_dir)
    save_computation_time(total_control_time, 'total_control', out_dir=output_dir)
    spa_tube_filename = os.path.join(output_dir, 'spacing_tube_contraction.csv')
    vel_tube_filename = os.path.join(output_dir, 'velocity_tube_contraction.csv')
    np.savetxt(spa_tube_filename, spa_contraction_log, delimiter=',')
    np.savetxt(vel_tube_filename, vel_contraction_log, delimiter=',')
    print(f"Tube contraction data saved to '{spa_tube_filename}' and '{vel_tube_filename}'")
if __name__ == '__main__':
    main()
