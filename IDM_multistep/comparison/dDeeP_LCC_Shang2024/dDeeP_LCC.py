import numpy as np
import cvxpy as cp
import time
import os
from functions import (Tstep, pos_cav, n_vehicle, n_cav, Tini, N,
                        s_limit, u_limit, scenario, disturbance_method, Ts_down, T, kappa,
                        IDM_dynamics, Tgap_list, v0_IDM_list, Tgap_rate_list, v0_rate_list, veh_len_list,
                        a_IDM_list, b_IDM_list, s0_IDM_list,
                        init_sub, save_vehicle_data_to_csv, plot,
                        estimate_disturbance_bounds, disturbance_vertices,
                        compute_tracking_metrics, save_metrics, _env_float, _env_flag,
                        subplatoon_spec, vehicle_csv_path, data_file, time_varying,
                        weight_s, weight_v, weight_u, lambda_gi, lambda_yi, reg_pinv)

DDEEPC_WARM_START = _env_flag('DDEEPC_WARM_START', True)

# =============================================================================
#            Decentralized Robust DeeP-LCC (dDeeP-LCC), Shang 2024
#
#  Each CF-LCC subsystem solves its OWN robust min-max problem (16) at every
#  time step, with NO communication to its neighbours. The preceding vehicle's
#  future velocity (the coupling disturbance) is bounded by an online-estimated
#  uncertainty set W_i, and the safety/spacing constraint must hold for every
#  disturbance in W_i. The robust problem is solved with the vertex-based
#  strategy (Method I, Prop. 3 / eq. (24)).
#
#  Local problem (after constraint elimination, Section V-A), variables
#  x = (u, sigma_y); for each vertex w_j of the down-sampled set W_tilde:
#       g_j  = g_const + G_yini @ sigma_y + G_u @ u + G_eps @ w_j
#       y_j  = Yif @ g_j
#       Vi_j = ||u||_R^2 + ||y_j - yref||_Q^2 + lambda_g ||g_j||^2 + lambda_y ||sigma_y||^2
#  min_x  max_j Vi_j      s.t.   s_min <= Pi @ y_j <= s_max  (all j),  u in [umin,umax]
# =============================================================================


class RobustSubsystemSolver:
    """Pre-compiled robust local DeeP-LCC for one CF-LCC subsystem.

    The worst-case-over-vertices cost shares the same Hessian M in z = (u,sigma_y)
    for every vertex, so robust problem (24) reduces to the pure QP
        min_z  quad(z, M) + t
        s.t.   d_j^T z + c0_j <= t                        (worst-case cost, all j)
               s_min <= Pby z + soff_j <= s_max           (robust safety, all j)
               u_min <= u <= u_max
    which is solved with OSQP. Vertex-dependent data (d_j, c0_j, soff_j) are fed
    as Parameters and rebuilt cheaply in numpy each step.
    """

    def __init__(self, precomp, n_vertices):
        self.p_sub = precomp['p_sub']
        self.down_nodes = precomp['down_nodes']
        self.n_eps = len(precomp['down_nodes'])
        self.n_vertices = n_vertices
        self.nz = precomp['nz']

        # numeric maps kept for the per-step assembly
        self.G_uini = precomp['G_uini']; self.G_eini = precomp['G_eini']
        self.G_yini = precomp['G_yini']
        self.Yif = precomp['Yif']; self.By = precomp['By']; self.Ye = precomp['Ye']
        self.Bg = precomp['Bg']; self.G_eps = precomp['G_eps']
        self.Q = precomp['Q']; self.lg = precomp['lambda_gi']
        self.Pi = precomp['Pi']; self.Dw = precomp['Dw']; self.PiYe = precomp['PiYe']
        self.warm_start = DDEEPC_WARM_START

        Pby = precomp['Pby']
        Lc = precomp['Lc']
        smin, smax = s_limit[0], s_limit[1]
        umin, umax = u_limit[0], u_limit[1]

        # ---- cvxpy QP in z = (u, sigma_y), compiled once ----
        # worst-case cost = ||Lc^T z||^2 + max_j (d_j^T z + c0_j)  (epigraph t)
        z = cp.Variable(self.nz, name='z')
        t = cp.Variable(name='t')
        self.d_all = cp.Parameter((self.nz, n_vertices), name='d_all')
        self.c0_all = cp.Parameter(n_vertices, name='c0_all')
        self.soff_all = cp.Parameter((N, n_vertices), name='soff_all')

        constraints = [z[:N] >= umin, z[:N] <= umax]
        for j in range(n_vertices):
            constraints.append(self.d_all[:, j] @ z + self.c0_all[j] <= t)
            spacing_j = Pby @ z + self.soff_all[:, j]
            constraints.append(spacing_j >= smin)
            constraints.append(spacing_j <= smax)

        objective = cp.Minimize(cp.sum_squares(Lc.T @ z) + t)
        self.z = z
        self.prob = cp.Problem(objective, constraints)

        # cached per-step quantities (filled in solve)
        self.g_const = None
        self.ay = None

    def solve(self, uini, eini, yini_flat, yref, vertices):
        # constant (here-and-now) part of g and the nominal output ay = Yif g_const
        g_const = self.G_uini @ uini + self.G_eini @ eini + self.G_yini @ yini_flat
        ay = self.Yif @ g_const
        ay_ref = ay - yref
        Qay = self.Q @ ay_ref
        d_base = 2.0*(self.By.T @ Qay) + 2.0*self.lg*(self.Bg.T @ g_const)
        Pi_ay = self.Pi @ ay

        d_all = np.empty((self.nz, self.n_vertices))
        c0_all = np.empty(self.n_vertices)
        soff_all = np.empty((N, self.n_vertices))
        for j, w in enumerate(vertices):
            d_all[:, j] = d_base + self.Dw @ w
            resid = ay_ref + self.Ye @ w
            g_w = g_const + self.G_eps @ w
            c0_all[j] = resid @ (self.Q @ resid) + self.lg*float(g_w @ g_w)
            soff_all[:, j] = Pi_ay + self.PiYe @ w

        self.d_all.value = d_all
        self.c0_all.value = c0_all
        self.soff_all.value = soff_all
        # With a persistently exciting data set the reduced Hessian is well
        # conditioned (cond(M) ~ 1e3), so OSQP solves quickly (~0.1 s). Clarabel
        # is kept as a reliable interior-point fallback.
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=self.warm_start, verbose=False,
                            max_iter=8000, eps_abs=1e-4, eps_rel=1e-4)
            if self.prob.status not in ('optimal', 'optimal_inaccurate'):
                self.prob.solve(solver=cp.CLARABEL, verbose=False)
        except cp.error.SolverError:
            self.prob.solve(solver=cp.CLARABEL, verbose=False)

        self.g_const = g_const
        self.ay = ay
        if self.z.value is None:
            return np.zeros(N), np.zeros(self.By.shape[0])
        z = self.z.value
        u = z[:N]
        # nominal predicted output (at the midpoint disturbance) for reference use
        w_nom = np.mean(np.array(vertices), axis=0)
        y_nom = ay + self.By @ z + self.Ye @ w_nom
        return u, y_nom


if __name__ == "__main__":
    args = init_sub(scenario)
    SDES = _env_float('KMPC_SDES', 40.0)
    result_root = os.environ.get('KMPC_RESULT_ROOT', 'result')
    run_suffix = os.environ.get('KMPC_RUN_SUFFIX', '')
    if not run_suffix.strip():
        run_suffix = '_' + time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(result_root, scenario, 'dDeeP_LCC' + run_suffix)
    vehicle_data_dir = os.path.join(output_dir, 'vehicle_data')
    os.makedirs(output_dir, exist_ok=True)

    S_list, uini_list, eini_list, yini_list, precomp_list = [], [], [], [], []
    for i in range(n_cav):
        S, ui_ini, ei_ini, yi_ini, precomp = args[i]
        S_list.append(S)
        uini_list.append(ui_ini)
        eini_list.append(ei_ini)
        yini_list.append(yi_ini)
        precomp_list.append(precomp)

    n_vertices = 2 ** len(precomp_list[0]['down_nodes'])
    solvers = [RobustSubsystemSolver(precomp_list[i], n_vertices) for i in range(n_cav)]
    print(f'dDeeP-LCC: method={disturbance_method}, Ts_down={Ts_down}, '
          f'n_eps={solvers[0].n_eps}, vertices per subsystem={n_vertices}')

    run_settings = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'scenario': scenario,
        'control_method': 'dDeeP-LCC Shang2024',
        'time_varying': time_varying,
        'T': T,
        'Tini': Tini,
        'N': N,
        'kappa': kappa,
        'SDES(desired/equilibrium spacing)': SDES,
        'Spa[min,max]': s_limit,
        'Acc[min,max]': u_limit,
        'subplatoon_config': subplatoon_spec,
        'vehicle_csv': vehicle_csv_path,
        'data_file': data_file,
        'robust_settings': {
            'disturbance_method': disturbance_method,
            'Ts_down': Ts_down,
            'n_vertices': n_vertices,
            'reg_pinv': reg_pinv,
            'warm_start': DDEEPC_WARM_START,
        },
        'deepc_weights': {
            'weight_s': weight_s,
            'weight_v': weight_v,
            'weight_u': weight_u,
            'lambda_gi': lambda_gi,
            'lambda_yi': lambda_yi,
        },
        'result_root': result_root,
        'run_suffix': run_suffix,
        'output_dir': output_dir,
    }
    settings_path = os.path.join(output_dir, 'run_settings.txt')
    with open(settings_path, 'w', encoding='utf-8') as f:
        for key, value in run_settings.items():
            f.write(f"{key}: {value}\n")
    print(f"Run settings saved to '{settings_path}'")

    total_time_steps = S_list[0].shape[0]
    control_steps = total_time_steps - Tini - 1
    computation_time = np.zeros(control_steps)

    yref_list = [[] for _ in range(n_cav)]
    y_opt_list = [np.zeros(yini_list[i].shape[0] * N) for i in range(n_cav)]  # previous optimal output

    Tgap_this_step_list = [[] for i in range(n_cav)]
    v0_IDM_this_step_list = [[] for i in range(n_cav)]
    cutin_time = 8.0
    cutin_drop = 25.0
    cutin_step = int(round(cutin_time / Tstep))

    for k in range(Tini, total_time_steps - 1):
        step_idx = k - Tini
        for i in range(n_cav):
            Tgap_this_step_list[i] = Tgap_list[i] * (1 + Tgap_rate_list[i]/100)**(k+Tini)
            v0_IDM_this_step_list[i] = v0_IDM_list[i] * (1 + v0_rate_list[i]/100)**(k+Tini)

        # ---- build reference output (same logic as the distributed baseline) ----
        if scenario == 'stabilization':
            vdes = 25
            sdes = SDES
            for i in range(n_cav):
                p = yini_list[i].shape[0]
                yref = np.zeros((p, N))
                yref[0::2, :] = sdes
                yref[1::2, :] = vdes
                yref_list[i] = yref.reshape(p*N, order='F')
        elif scenario == 'tracking' or scenario == 'braking' or scenario == 'cutin':
            sdes = SDES
            for i in range(n_cav):
                p = yini_list[i].shape[0]
                v_pv = np.tile(S_list[0][k, 0, 1], N)
                if i == 0:
                    v_cav_prev = np.tile(S_list[0][k, 0, 1], N)
                else:
                    y_prev_reshaped = y_opt_list[i-1].reshape(yini_list[i-1].shape[0], N, order='F')
                    v_cav_prev_opt = y_prev_reshaped[1, :]
                    v_cav_prev = np.append(v_cav_prev_opt[1:], v_cav_prev_opt[-1])

                y_curr_reshaped = y_opt_list[i].reshape(yini_list[i].shape[0], N, order='F')
                v_cav_as_opt = y_curr_reshaped[1, :]
                v_cav_as = np.append(v_cav_as_opt[1:], v_cav_as_opt[-1])

                if i == 0:
                    v_ref = np.tile(S_list[0][k, 0, 1], N)
                else:
                    y_prev_reshaped = y_opt_list[i-1].reshape(yini_list[i-1].shape[0], N, order='F')
                    v_ref_opt = y_prev_reshaped[-1, :]
                    v_ref = np.append(v_ref_opt[1:], v_ref_opt[-1])

                vdes = (v_pv + v_cav_prev + 2*v_cav_as + 2*v_ref) / 6
                yref = np.zeros((p, N))
                yref[0::2, :] = sdes
                yref[1::2, :] = vdes
                yref_list[i] = yref.reshape(p*N, order='F')

        # ---- decentralized robust control: each subsystem solves independently ----
        start_time = time.time()
        u_opt = [None] * n_cav
        for i in range(n_cav):
            # estimate the disturbance set W_i from the past disturbance eps_i,ini
            eps_min, eps_max = estimate_disturbance_bounds(eini_list[i], N, Tstep, disturbance_method)
            vertices = disturbance_vertices(eps_min, eps_max, precomp_list[i]['down_nodes'])
            yini_flat = yini_list[i].reshape(yini_list[i].shape[0]*Tini, order='F')
            u_i, y_nom = solvers[i].solve(uini_list[i], eini_list[i], yini_flat, yref_list[i], vertices)
            u_opt[i] = u_i
            y_opt_list[i] = y_nom
        computation_time[step_idx] = (time.time() - start_time) / n_cav
        print(f'Time step {step_idx}/{control_steps}, Computation Time per Subsystem: {computation_time[step_idx]:.4f}s')

        for i in range(n_cav):
            # apply control input and propagate the sub-platoon with IDM dynamics
            acel = IDM_dynamics(
                S_list[i][k, :, :],
                Tgap_this_step_list[i],
                v0_IDM_this_step_list[i],
                veh_len_list[i],
                a_IDM_list[i],
                b_IDM_list[i],
                s0_IDM_list[i],
            )
            S_list[i][k, 1:, 2] = acel            # all following vehicles use HDV model
            S_list[i][k, 1, 2] = u_opt[i][0]      # CAV control input

            S_list[i][k+1, 1:, 1] = S_list[i][k, 1:, 1] + Tstep*S_list[i][k, 1:, 2]
            if i != 0:
                S_list[i][k+1, 0, 1] = S_list[i-1][k+1, -1, 1]
            S_list[i][k+1, :, 0] = S_list[i][k, :, 0] + Tstep*S_list[i][k, :, 1]
            if i != 0:
                S_list[i][k+1, 0, 0] = S_list[i-1][k+1, -1, 0]

            # record output and update past initial trajectories
            y = np.ravel(np.column_stack((S_list[i][k+1, :-1, 0] - S_list[i][k+1, 1:, 0], S_list[i][k+1, 1:, 1])))
            if i == 0:
                e = S_list[0][k+1, 0, 1]   # PV velocity
            else:
                e = S_list[i-1][k+1, -1, 1]  # velocity of the last car of the previous subsystem
            u = u_opt[i][0]

            eini_list[i] = np.append(eini_list[i][1:], e)
            uini_list[i] = np.append(uini_list[i][1:], u)
            yini_list[i] = np.hstack((yini_list[i][:, 1:], y.reshape(-1, 1)))

        if scenario == 'cutin' and (step_idx + 1) == cutin_step:
            S_list[0][k+1, 0, 0] -= cutin_drop
            print(f"[cut-in] Step {step_idx+1}: leader position shifted backward by {cutin_drop:.1f} m; CAV1-PV spacing dropped sharply")

    S_total = np.zeros((total_time_steps, n_vehicle+1, 3))
    for i in range(n_cav):
        if i == 0:
            S_total[:, 0:pos_cav[i+1]+1, :] = S_list[i]
        elif i != n_cav-1:
            S_total[:, pos_cav[i]+1:pos_cav[i+1]+1, :] = S_list[i][:, 1:, :]
        else:
            S_total[:, pos_cav[i]+1:, :] = S_list[i][:, 1:, :]

    print(f"Result output directory: {output_dir}")
    np.savetxt(os.path.join(output_dir, 'control_computation_time.csv'), computation_time, delimiter=',')
    save_vehicle_data_to_csv(S_total, vehicle_data_dir)
    plot(scenario, S_total, Tstep, n_cav, output_dir, subplatoon_spec)

    metrics = compute_tracking_metrics(S_total[Tini:], scenario, sdes=SDES)
    metrics.update({
        'AvgControlTime_s': float(np.mean(computation_time)),
        'weight_s': weight_s,
        'weight_v': weight_v,
        'weight_u': weight_u,
        'lambda_gi': lambda_gi,
        'lambda_yi': lambda_yi,
        'disturbance_method': disturbance_method,
        'Ts_down': Ts_down,
        'n_vertices': n_vertices,
        'warm_start': DDEEPC_WARM_START,
    })
    print("==================== Evaluation Metrics ====================")
    print(f"  RMSVE = {metrics['RMSVE']:.4f} m/s   RMSSE = {metrics['RMSSE']:.4f} m")
    print(f"  Avg control solve time = {metrics['AvgControlTime_s']*1000:.3f} ms")
    print("=================================================")
    save_metrics(metrics, filename=os.path.join(output_dir, 'metrics.csv'))
