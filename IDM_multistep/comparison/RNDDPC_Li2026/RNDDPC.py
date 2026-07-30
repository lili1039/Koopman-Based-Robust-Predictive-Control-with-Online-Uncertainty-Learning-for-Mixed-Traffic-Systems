# =============================================================================
#  RNDDPC (Li et al., 2026) — closed-loop driver  [baseline for comparison]
#
#  Mirrors the parent `IDM/Koopman_MPC_IDM_learning_set.py` (decentralized
#  subplatoon decomposition, IDM HDVs, same scenarios / limits / cost), but the
#  CAV of every subplatoon is controlled by the RNDDPC optimisation (50):
#  data-driven Koopman model + matrix-zonotope reachable-set constraint
#  tightening. Each subplatoon is solved independently (no inter-CAV coupling),
#  matching how the paper decentralises a large platoon.
#
#  Run examples (PowerShell):
#    $env:KMPC_SCENARIO='braking';  python RNDDPC.py
#    $env:KMPC_SCENARIO='tracking'; python RNDDPC.py
#    $env:KMPC_CONTROL='All HDVs';  python RNDDPC.py
# =============================================================================
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import time
import numpy as np

import functions as f
from functions import (
    helper, scenario, control_method, output_dir, Tstep, Nfut,
    SpaMax, SpaMin, VelMax, VelMin, AccMax, AccMin, sdes, vdes_stab, EPS_MAX, EP_EXTRAPOLATE,
    HF_MODELS, WITH_HISTORY_MODEL_PATHS, USE_HISTORY_KOOPMAN,
    subplatoon_spec, vehicle_csv_path, cutin_time, cutin_drop,
    rnddpc_q_spa, rnddpc_q_vel, rnddpc_r, rnddpc_rd,
    W_SPACING_LOW, W_TERM_SPACING, W_SLACK, W_SLACK_L2,
    TUBE_MODE, CONIC_SOLVER, MZ_KEEP_GENERATORS,
    build_rnddpc_bundle, reachable_tube, nominal_lifted_rollout, RNDDPC_Solver, RNDDPC_PaperConicSolver,
    IDM_dynamics, compute_IDM_steady_s, generate_history_seq,
    plot, compute_tracking_metrics, save_vehicle_data_to_csv, save_metrics, save_computation_time,
)

time_varying = True   # heterogeneous time-varying IDM (matches parent default)


def save_run_settings(path, settings):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fobj:
        for key, value in settings.items():
            fobj.write(f"{key}: {value}\n")
    print(f"Run settings saved to '{path}'")


def main():
    print(f"[RNDDPC] scenario={scenario}  control={control_method}  -> {output_dir}")

    # ===================== subplatoon layout (== parent default) =====================
    # each segment = 1 CAV + num_HDV HDVs; order from the PV backwards
    subplatoon_numHDV = list(subplatoon_spec)
    available_models = WITH_HISTORY_MODEL_PATHS if USE_HISTORY_KOOPMAN else HF_MODELS
    for nh in subplatoon_numHDV:
        if nh not in available_models:
            mode = 'with-history' if USE_HISTORY_KOOPMAN else 'history-free'
            raise ValueError(f"RNDDPC has no {mode} encoder for num_HDV={nh}; available={sorted(available_models)}")
    num_CAV = 1
    sim_subplatoon_num = len(subplatoon_numHDV)
    sub_numveh = [num_CAV + nh for nh in subplatoon_numHDV]
    sub_sdim = [2 * nv for nv in sub_numveh]
    sub_start = [0] * sim_subplatoon_num
    for k in range(1, sim_subplatoon_num):
        sub_start[k] = sub_start[k - 1] + sub_numveh[k - 1]
    sim_num_veh = sum(sub_numveh)
    print(f"[RNDDPC] layout {['1C%dH' % h for h in subplatoon_numHDV]}, total veh={sim_num_veh}, start={sub_start}")

    # ===================== scenario velocity profile =====================
    if scenario == 'tracking':
        time_list, PV_vel = helper.generate_NEDC_velocity_profile(Tstep)
    elif scenario == 'braking':
        time_list, PV_vel = helper.generate_braking_velocity_profile(Tstep)
    elif scenario == 'stabilization':
        time_list, PV_vel = helper.generate_sine_velocity_profile(Tstep)
    elif scenario == 'cutin':
        time_list, PV_vel = helper.generate_cutin_velocity_profile(Tstep, cutin_time=cutin_time)
        cutin_step = int(round(cutin_time / Tstep))
    else:
        raise ValueError(f'unknown scenario {scenario}')
    total_time_steps = len(time_list)
    sim_maxtime = f._env_float('KMPC_SIM_MAXTIME', 0.0)
    if sim_maxtime > 0:
        cap = int(round(sim_maxtime / Tstep))
        if 0 < cap < total_time_steps:
            time_list = time_list[:cap]
            PV_vel = PV_vel[:cap]
            total_time_steps = cap
            print(f"[sim-cap] Truncated simulation to {sim_maxtime:.1f}s ({total_time_steps} steps)")

    # ===================== vehicle / IDM parameters =====================
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

    # ===================== offline bundles (one per actual subplatoon) =====================
    sub_bundle = None
    if control_method != 'All HDVs':
        sub_bundle = []
        for k, nh in enumerate(subplatoon_numHDV):
            base = sub_start[k]
            nv = sub_numveh[k]
            idm_params = (
                Tgap[base:base + nv],
                v0_IDM[base:base + nv],
                veh_len[base:base + nv],
                a_IDM[base:base + nv],
                b_IDM[base:base + nv],
                s0_IDM[base:base + nv],
            )
            sub_bundle.append(build_rnddpc_bundle(nh, idm_params=idm_params, bundle_label=f"sub{k}"))

    args = vars(helper.get_args_local(False))
    Npst = args['Npst']

    # initial equilibrium
    v_init = np.full(sim_num_veh, PV_vel[0])
    s_init = np.zeros(sim_num_veh)
    for i in range(sim_num_veh):
        s_init[i] = compute_IDM_steady_s(PV_vel[0], PV_vel[0], v0_IDM[i], Tgap[i], a_IDM[i], b_IDM[i], s0_IDM[i]) + veh_len[i]

    # warm-up history (reuse parent helper)
    S_history, x_history_seq_list, x_seq_list = generate_history_seq(
        args, sim_subplatoon_num, sim_num_veh, sub_start, subplatoon_numHDV,
        PV_vel[0], v_init, s_init, Tgap, v0_IDM, Tgap_rate, v0_rate, veh_len, Tstep, a_IDM, b_IDM, s0_IDM)

    # full state buffer
    S = np.zeros([Npst + total_time_steps, sim_num_veh + 1, 3])   # pos/vel/acc
    S[0:Npst + 1, :, :] = S_history
    S[Npst:, 0, 1] = PV_vel
    for i in range(Npst + total_time_steps - 1):
        S[i, 0, 2] = (S[i + 1, 0, 1] - S[i, 0, 1]) / Tstep

    # ===================== solvers =====================
    sub_solver = None
    if control_method == 'RNDDPC':
        if TUBE_MODE == 'paper_conic':
            sub_solver = [RNDDPC_PaperConicSolver(b, Nfut, scenario) for b in sub_bundle]
        else:
            sub_solver = [RNDDPC_Solver(b['A'], b['B'], b['H'], b['C'], b['n_z'], b['s_dim'], Nfut, scenario)
                          for b in sub_bundle]
        prev_u = [np.zeros(Nfut) for _ in range(sim_subplatoon_num)]   # nominal control for tube
        last_u = [np.zeros(1) for _ in range(sim_subplatoon_num)]

    run_settings = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'scenario': scenario,
        'control_method': control_method,
        'Nfut': Nfut,
        'SDES': sdes,
        'Spa[min,max]': [SpaMin, SpaMax],
        'Vel[min,max]': [VelMin, VelMax],
        'Acc[min,max]': [AccMin, AccMax],
        'subplatoon_config': subplatoon_numHDV,
        'vehicle_csv': vehicle_csv_path,
        'time_varying': time_varying,
        'weights': {
            'spacing': rnddpc_q_spa,
            'velocity': rnddpc_q_vel,
            'control': rnddpc_r,
            'delta_u_penalty': rnddpc_rd,
            'spacing_low_penalty': W_SPACING_LOW,
            'terminal_spacing_penalty': W_TERM_SPACING,
        },
        'koopman_lifting': {
            'with_history': USE_HISTORY_KOOPMAN,
            'checkpoints': [b['model_ref'] for b in sub_bundle] if sub_bundle else [],
        },
        'constraints': {
            'spacing': 'tightened with RNDDPC reachable tube and softened by slack',
            'velocity': 'tightened with RNDDPC reachable tube and softened by slack',
            'terminal_velocity': 'soft equality with slack',
            'acceleration': [AccMin, AccMax],
        },
        'slack_weights': {
            'W_SLACK': W_SLACK,
            'W_SLACK_L2': W_SLACK_L2,
        },
        'rnddpc_uncertainty': {
            'EPS_MAX': f.EPS_MAX,
            'Q_COVER': f.Q_COVER,
            'SIGMA_SCALE': f.SIGMA_SCALE,
            'RCOND_PINV': f.RCOND_PINV,
            'EP_EXTRAPOLATE': EP_EXTRAPOLATE,
            'TUBE_MODE': TUBE_MODE,
            'CONIC_SOLVER': CONIC_SOLVER if TUBE_MODE == 'paper_conic' else None,
            'MZ_KEEP_GENERATORS': MZ_KEEP_GENERATORS,
            'MZ_GENERATOR_COUNTS': [b['mz_generator_counts'] for b in sub_bundle] if sub_bundle else [],
        },
        'data_collection': {
            'rollouts': f.DATA_ROLLOUTS,
            'rollout_len': f.DATA_ROLLOUT_LEN,
            'noise': f.DATA_NOISE,
            'seed': f.DATA_SEED,
            'vbase_range': [f.DATA_VBASE_MIN, f.DATA_VBASE_MAX],
            'head_velocity_range': [f.DATA_HEAD_MIN, f.DATA_HEAD_MAX],
            'head_random_walk_std': f.DATA_HEAD_STD,
            'cav_pe_std': f.DATA_PE_STD,
            'idm_params': 'vehicle_parameters.csv slices per subplatoon',
        },
        'cutin_time': cutin_time,
        'cutin_drop': cutin_drop,
        'output_dir': output_dir,
        'sim_maxtime': sim_maxtime,
    }
    save_run_settings(os.path.join(output_dir, 'run_settings.txt'), run_settings)

    control_time = np.zeros((sim_subplatoon_num, total_time_steps - 1))
    solver_core_time = np.zeros((sim_subplatoon_num, total_time_steps - 1))
    tube_time = np.zeros((sim_subplatoon_num, total_time_steps - 1))
    spa_contraction_log = np.zeros((sim_subplatoon_num, total_time_steps - 1))
    vel_contraction_log = np.zeros((sim_subplatoon_num, total_time_steps - 1))
    slack_max_collection = np.zeros((sim_subplatoon_num, total_time_steps - 1))

    # =============================== simulation ===============================
    for i in range(total_time_steps - 1):
        Tgap_t = Tgap * (1 + Tgap_rate / 100) ** (i + Npst)
        v0_t = v0_IDM * (1 + v0_rate / 100) ** (i + Npst)

        # HDV accelerations (IDM) for the whole platoon
        acel = IDM_dynamics(S[Npst + i, :, :], Tgap_t, v0_t, veh_len, a_IDM, b_IDM, s0_IDM)
        S[Npst + i, 1:, 2] = acel

        if control_method == 'RNDDPC':
            for k in range(sim_subplatoon_num):
                b = sub_bundle[k]
                base = sub_start[k]; nv = sub_numveh[k]
                lead_idx = base; cav_idx = base + 1

                # current subplatoon state x and lifted z
                x_t = x_seq_list[k][0].copy()
                if b.get('use_history_koopman', False):
                    z_t = b['model'].lift(x_t, x_history_seq_list[k]).squeeze()
                else:
                    z_t = b['model'].lift(x_t).squeeze()
                ep_t = float(S[Npst + i, lead_idx, 1])      # velocity of vehicle ahead of this CAV

                # ---- reference: track leader velocity, desired spacing ----
                x_ref = np.zeros((b['s_dim'], Nfut + 1))
                x_ref[:, 0] = x_t
                if scenario == 'stabilization':
                    x_ref[0::2, :] = sdes; x_ref[1::2, :] = vdes_stab
                else:
                    x_ref[0::2, 1:] = sdes; x_ref[1::2, 1:] = ep_t
                # head-velocity sequence for the nominal dynamics: extrapolate with
                # the preceding vehicle's current acceleration (paper's constant-eps
                # if EP_EXTRAPOLATE is off). The tube still robustifies variation.
                if EP_EXTRAPOLATE and scenario != 'stabilization':
                    aPV_t = float(S[Npst + i, lead_idx, 2])
                    ep_seq = np.clip(ep_t + aPV_t * Tstep * np.arange(Nfut + 1), 0.0, VelMax)[None, :]
                else:
                    ep_seq = np.full((1, Nfut + 1), ep_t)

                # ---- solve RNDDPC QP. Slack handles softened constraints; no fallback input is used. ----
                if TUBE_MODE == 'paper_conic':
                    x_pred, z_pred, u_pred, ct, st, slack_vals, tube_halfwidth = sub_solver[k].solve(
                        z_t, x_t, x_ref, ep_seq, last_u[k])
                    control_time[k, i] = ct
                    solver_core_time[k, i] = st
                    tube_time[k, i] = 0.0
                    spa_contraction_log[k, i] = float(np.max(2.0 * tube_halfwidth[0, :]))
                    vel_contraction_log[k, i] = float(np.max(2.0 * tube_halfwidth[1, :]))
                else:
                    # ---- reachable-set tube on a nominal trajectory (decoupled from u) ----
                    t0 = time.time()
                    u_nom = np.concatenate([prev_u[k][1:], prev_u[k][-1:]])   # shifted previous solution
                    z_nom = nominal_lifted_rollout(b, z_t, ep_t, u_nom, Nfut)
                    Ex_interval = reachable_tube(b, ep_t, EPS_MAX, Nfut, z_nom, u_nom)
                    tube_time[k, i] = time.time() - t0
                    spa_contraction_log[k, i] = max(iv.right_limit[0] - iv.left_limit[0] for iv in Ex_interval)
                    vel_contraction_log[k, i] = max(iv.right_limit[1] - iv.left_limit[1] for iv in Ex_interval)
                    x_pred, z_pred, u_pred, ct, st, slack_vals = sub_solver[k].solve(
                        z_t, x_t, x_ref, ep_seq, Ex_interval, last_u[k])
                    control_time[k, i] = ct
                    solver_core_time[k, i] = st
                prev_u[k] = u_pred[0].copy()
                u_apply = float(u_pred[0, 0])
                last_u[k] = np.array([u_apply])
                slack_max_collection[k, i] = max(
                    float(np.max(v)) if isinstance(v, np.ndarray) else float(v)
                    for v in slack_vals.values()
                )
                S[Npst + i, cav_idx, 2] = u_apply
                if i % 50 == 0:
                    print(f"  sub{k} step{i}: u={u_pred[0,0]:+.3f}  tube(spa={spa_contraction_log[k,i]:.3f}m "
                          f"vel={vel_contraction_log[k,i]:.3f}) total={ct*1000:.1f}ms "
                          f"solver={st*1000:.1f}ms", flush=True)

        # propagate platoon one step
        S[Npst + i + 1, :, 0] = S[Npst + i, :, 0] + Tstep * S[Npst + i, :, 1]
        S[Npst + i + 1, 1:, 1] = S[Npst + i, 1:, 1] + Tstep * S[Npst + i, 1:, 2]
        if scenario == 'cutin' and (i + 1) == cutin_step:
            S[Npst + i + 1, 0, 0] -= cutin_drop
            print(f"[cut-in] Step {i+1}: leader position shifted backward by {cutin_drop:.1f} m; CAV1-PV spacing dropped sharply")

        # refresh current subplatoon states
        if control_method != 'All HDVs':
            for k in range(sim_subplatoon_num):
                base = sub_start[k]; nv = sub_numveh[k]
                numHDV_k = subplatoon_numHDV[k]
                slice_dim = args['HV_hist_slice_dim']
                x_new_slice = np.zeros((1, 1, slice_dim, numHDV_k))
                for j in range(numHDV_k):
                    x_new_slice[0, 0, 0, j] = S[Npst + i + 1, base + j + 1, 1]
                    x_new_slice[0, 0, 1, j] = S[Npst + i + 1, base + j + 1, 0] - S[Npst + i + 1, base + j + 2, 0]
                    x_new_slice[0, 0, 2, j] = S[Npst + i + 1, base + j + 2, 1]
                x_history_seq_list[k] = np.concatenate(
                    (x_history_seq_list[k][:, 1:, :, :], x_new_slice),
                    axis=1,
                )
                x_seq_list[k][0, 0::2] = S[Npst + i + 1, base:base + nv, 0] - S[Npst + i + 1, base + 1:base + nv + 1, 0]
                x_seq_list[k][0, 1::2] = S[Npst + i + 1, base + 1:base + nv + 1, 1]

    # =============================== outputs ===============================
    print(f"[RNDDPC] writing results to {output_dir}")
    plot(args, scenario, S, Tstep, sim_subplatoon_num, subplatoon_numHDV, out_dir=output_dir)
    save_vehicle_data_to_csv(S, filename_prefix=os.path.join(output_dir, 'vehicle_data'))

    metrics = compute_tracking_metrics(S[Npst:], scenario, sdes=sdes, vdes_stab=vdes_stab)
    if control_method != 'All HDVs':
        metrics['AvgControlTime_s'] = float(np.mean(control_time))
        metrics['AvgSolverCoreTime_s'] = float(np.nanmean(solver_core_time))
        metrics['AvgModelingInterfaceTime_s'] = float(np.nanmean(control_time - solver_core_time))
        metrics['AvgTubeTime_s'] = float(np.mean(tube_time))
        metrics['AvgTotalTime_s'] = float(np.mean(control_time + tube_time))
        metrics['weight_s'] = rnddpc_q_spa
        metrics['weight_v'] = rnddpc_q_vel
        metrics['weight_u'] = rnddpc_r
        metrics['delta_u_penalty'] = rnddpc_rd
        metrics['spacing_low_penalty'] = W_SPACING_LOW
        metrics['terminal_spacing_penalty'] = W_TERM_SPACING
        metrics['velocity_constraints'] = True
        metrics['terminal_velocity_constraint'] = True
        metrics['SlackEnabled'] = 1
        metrics['SlackMax'] = float(np.max(slack_max_collection))
        metrics['TubeMode'] = TUBE_MODE
        metrics['ConicSolver'] = CONIC_SOLVER if TUBE_MODE == 'paper_conic' else ''
    print("==================== metrics ====================")
    print(f"  RMSVE = {metrics['RMSVE']:.4f} m/s   RMSSE = {metrics['RMSSE']:.4f} m")
    if control_method != 'All HDVs':
        solve_label = 'ctrl conic' if TUBE_MODE == 'paper_conic' else 'ctrl QP'
        print(f"  {solve_label} = {metrics['AvgControlTime_s']*1000:.2f} ms | "
              f"solver core = {metrics['AvgSolverCoreTime_s']*1000:.2f} ms | "
              f"interface = {metrics['AvgModelingInterfaceTime_s']*1000:.2f} ms | "
              f"tube = {metrics['AvgTubeTime_s']*1000:.2f} ms | total = {metrics['AvgTotalTime_s']*1000:.2f} ms")
    print("=================================================")
    save_metrics(metrics, filename=os.path.join(output_dir, 'metrics.csv'))
    if control_method != 'All HDVs':
        save_computation_time(control_time, 'control', out_dir=output_dir)
        save_computation_time(solver_core_time, 'solver_core', out_dir=output_dir)
        save_computation_time(tube_time, 'tube', out_dir=output_dir)
        slack_filename = os.path.join(output_dir, 'slack_max.csv')
        np.savetxt(slack_filename, slack_max_collection, delimiter=',')
        print(f"Slack max data saved to '{slack_filename}'")
        spa_tube_filename = os.path.join(output_dir, 'spacing_tube_contraction.csv')
        vel_tube_filename = os.path.join(output_dir, 'velocity_tube_contraction.csv')
        np.savetxt(spa_tube_filename, spa_contraction_log, delimiter=',')
        np.savetxt(vel_tube_filename, vel_contraction_log, delimiter=',')
        print(f"Tube contraction data saved to '{spa_tube_filename}' and '{vel_tube_filename}'")


if __name__ == '__main__':
    main()
