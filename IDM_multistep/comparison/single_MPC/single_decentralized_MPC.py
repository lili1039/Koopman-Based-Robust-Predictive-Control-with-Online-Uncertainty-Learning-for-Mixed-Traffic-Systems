# Basic imports
import os
import time
import numpy as np
import cvxpy as cp
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math

from functions import plot, save_vehicle_data_to_csv, save_computation_time, compute_tracking_metrics, save_metrics, Tstep, Npst, IDM_dynamics, compute_IDM_steady_s, generate_history_seq, predict_future_ep_and_constraints, generate_NEDC_velocity_profile, generate_sine_velocity_profile, generate_braking_velocity_profile, generate_cutin_velocity_profile
# ======================================================
#               distributed single MPC
# ======================================================
# Get the project root directory, which is two levels up from the current file's directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(f'project_root: {project_root}')

def _env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value is not None and value.strip() else default


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value is not None and value.strip() else default


def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _env_int_list(name, default):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return [int(item.strip()) for item in value.split(',') if item.strip()]


scenario = os.environ.get('KMPC_SCENARIO', 'cutin')
num_CAV = 1
subplatoon_spec = _env_int_list('KMPC_SUBPLATOON', [2, 3, 1, 4, 2])
vehicle_csv = os.environ.get('KMPC_VEHICLE_CSV', 'vehicle_parameters.csv')
result_root = os.environ.get('KMPC_RESULT_ROOT', 'result')
run_suffix = os.environ.get('KMPC_RUN_SUFFIX', '')
SDES = _env_float('KMPC_SDES', 40.0)
SpaMin = _env_float('KMPC_SPA_MIN', 10.0)
SpaMax = _env_float('KMPC_SPA_MAX', 80.0)
VelMin = _env_float('KMPC_VEL_MIN', 0.0)
VelMax = _env_float('KMPC_VEL_MAX', 35.0)
AccMin = _env_float('KMPC_ACC_MIN', -6.0)
AccMax = _env_float('KMPC_ACC_MAX', 4.0)
Nfut_setting = _env_int('KMPC_NFUT', 15)
dec_q_spa = _env_float('KMPC_DEC_Q_SPA', 2.5)
dec_q_vel = _env_float('KMPC_DEC_Q_VEL', 1.5)
dec_r = _env_float('KMPC_DEC_R', 20.0)
dec_rd = _env_float('KMPC_DEC_RD', 30.0)
enable_slack = _env_flag('KMPC_ENABLE_SLACK', False)
W_SLACK = _env_float('KMPC_SLACK_W', 1.0e4)
W_SLACK_L2 = _env_float('KMPC_SLACK_W_L2', 1.0e2)
cutin_time = 8.0
cutin_drop = 25.0

if not run_suffix.strip():
    run_suffix = '_' + time.strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join(result_root, scenario, 'Decentralized_single_MPC' + run_suffix)

time_varying = True

# --------- Distributed Solver ---------
class Decentralized_VehicleSolver:
    def __init__(self, A, B, H, SpaMax, SpaMin, VelMax, VelMin, AccMin, AccMax, Nfut, scenario):

        # ===== Parameter definitions; values are updated at each solve. =====
        self.x_ref = cp.Parameter((2, Nfut+1), name="x_ref")
        self.x_t = cp.Parameter((2,), name="x_t")
        self.ep  = cp.Parameter((1,Nfut+1), name="ep")
        self.u_prev = cp.Parameter((1,), name="u_prev")
        
        # ===== Decision variables =====
        self.u = cp.Variable((1, Nfut))
        self.x = cp.Variable((2, Nfut+1))
        self.sl_spa_lb = cp.Variable((Nfut,), nonneg=True)
        self.sl_spa_ub = cp.Variable((Nfut,), nonneg=True)
        self.sl_vel_lb = cp.Variable((Nfut,), nonneg=True)
        self.sl_vel_ub = cp.Variable((Nfut,), nonneg=True)

        # ===== Weights =====
        if scenario == 'tracking' or scenario == 'braking' or scenario == 'cutin':
            Q4 = np.diag([dec_q_spa, dec_q_vel])       # ref
            R = dec_r*np.eye(1)
            Rd = dec_rd*np.eye(1)

        elif scenario == 'stabilization':
            Q4 = np.diag([1,2])
            R = 12.5*np.eye(1)      # Control input.
            Rd = 5*np.eye(1)

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

        for i in range(Nfut):
            constraints += [
                self.x[:, i+1] == A @ self.x[:,i] + B @ self.u[:,i] + H @ self.ep[:,i],  # System dynamics.
                # Control-input constraints.
                self.u[0, i] >= self.AccMin,
                self.u[0, i] <= self.AccMax,
            ]
            if enable_slack:
                constraints += [
                    # Constrain only CAV spacing, with soft constraints.
                    self.x[0, i+1] <= self.SpaMax + self.sl_spa_ub[i],
                    self.x[0, i+1] >= self.SpaMin - self.sl_spa_lb[i],
                    # Constrain CAV speed, with soft constraints.
                    self.x[1, i+1] <= self.VelMax + self.sl_vel_ub[i],
                    self.x[1, i+1] >= self.VelMin - self.sl_vel_lb[i],
                ]
            else:
                constraints += [
                    # Constrain only CAV spacing.
                    self.x[0, i+1] <= self.SpaMax,
                    self.x[0, i+1] >= self.SpaMin,
                    # Constrain CAV speed.
                    self.x[1, i+1] <= self.VelMax,
                    self.x[1, i+1] >= self.VelMin,
                ]
            objective += (
                         cp.quad_form((self.x[:,i+1]-self.x_ref[:,i+1]), Q4) + 
                         cp.quad_form((self.u[:,i]), R)
                         )

        objective += cp.quad_form(self.u[:, 0] - self.u_prev, Rd)
        for i in range(1, Nfut):
            objective += cp.quad_form(self.u[:, i] - self.u[:, i-1], Rd)

        if enable_slack:
            slack_l1 = (
                cp.sum(self.sl_spa_lb) + cp.sum(self.sl_spa_ub) +
                cp.sum(self.sl_vel_lb) + cp.sum(self.sl_vel_ub)
            )
            slack_l2 = (
                cp.sum_squares(self.sl_spa_lb) + cp.sum_squares(self.sl_spa_ub) +
                cp.sum_squares(self.sl_vel_lb) + cp.sum_squares(self.sl_vel_ub)
            )
            objective += W_SLACK * slack_l1 + W_SLACK_L2 * slack_l2

        # Initial state.
        constraints += [self.x[:, 0] == self.x_t,]
        # Terminal state.
        constraints += [self.x[1,Nfut] == self.x_ref[1,Nfut],]
        


        # ===== Build the optimization problem =====
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def _slack_report(self):
        if not enable_slack:
            return None
        return {
            'spa_lb': np.asarray(self.sl_spa_lb.value).ravel(),
            'spa_ub': np.asarray(self.sl_spa_ub.value).ravel(),
            'vel_lb': np.asarray(self.sl_vel_lb.value).ravel(),
            'vel_ub': np.asarray(self.sl_vel_ub.value).ravel(),
        }

    def solve(self, x_t_val, x_ref_val, ep_val, u_prev_val):
        # Update parameters.
        self.x_t.value = x_t_val
        self.x_ref.value = x_ref_val
        self.ep.value = ep_val
        self.u_prev.value = u_prev_val

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
                self.u.value,
                compute_time,
                self._slack_report())

if __name__ == '__main__':
    A = np.array([[1.0, -Tstep],
                  [0.0, 1.0]])
    print(f"A shape: {A.shape}")

    B = np.array([[0.0],
                  [Tstep]])
    print(f"B shape: {B.shape}")

    H = np.array([[Tstep],   
                  [0.0]])
    print(f"H shape: {H.shape}")

    # ============================ Initialize each vehicle ===================================================
    if scenario == 'tracking':
        time_list, PV_vel = generate_NEDC_velocity_profile(Tstep)
    elif scenario == 'stabilization':
        time_list, PV_vel = generate_sine_velocity_profile(Tstep)
    elif scenario == 'braking':
        time_list, PV_vel = generate_braking_velocity_profile(Tstep)
    elif scenario == 'cutin':
        time_list, PV_vel = generate_cutin_velocity_profile(Tstep, cutin_time=cutin_time)
        cutin_step = int(round(cutin_time / Tstep))

    total_time_steps = len(time_list)
    sim_maxtime = _env_float('KMPC_SIM_MAXTIME', 0.0)
    if sim_maxtime > 0:
        cap = int(round(sim_maxtime / Tstep))
        if 0 < cap < total_time_steps:
            time_list = time_list[:cap]
            PV_vel = PV_vel[:cap]
            total_time_steps = cap
            print(f"[sim-cap] Truncated simulation to {sim_maxtime:.1f}s ({total_time_steps} steps)")

    sim_subplatoon_num = len(subplatoon_spec)
    sub_numHDV = subplatoon_spec
    sub_numveh = [num_CAV + nh for nh in sub_numHDV]
    sub_start = [0] * sim_subplatoon_num
    for k in range(1, sim_subplatoon_num):
        sub_start[k] = sub_start[k-1] + sub_numveh[k-1]
    sim_num_veh = sum(sub_numveh)
    print(f"Platoon composition: {['1C%dH' % h for h in sub_numHDV]}, total following vehicles (excluding PV)={sim_num_veh}, sub_start={sub_start}")
    
    # Load parameters from CSV in global vehicle order; number of rows must be >= sim_num_veh.
    vehicle_csv_path = vehicle_csv if os.path.isabs(vehicle_csv) else os.path.join(project_root, vehicle_csv)
    print(f"Vehicle parameter file: {vehicle_csv_path}")
    param_data = np.genfromtxt(vehicle_csv_path, delimiter=',', skip_header=1)
    param_data = np.atleast_2d(param_data)
    if param_data.shape[0] < sim_num_veh:
        raise ValueError(f"{vehicle_csv_path}: row count {param_data.shape[0]} < required vehicle count {sim_num_veh}")
    if param_data.shape[1] < 8:
        raise ValueError(f"{vehicle_csv_path}: expected columns Tgap,v0_IDM,veh_len,Tgap_rate,v0_rate,a_IDM,b_IDM,s0_IDM")

    param_data = param_data[:sim_num_veh, :]
    Tgap = param_data[:, 0]
    v0_IDM = param_data[:, 1]
    veh_len = param_data[:, 2]
    a_IDM = param_data[:, 5]
    b_IDM = param_data[:, 6]
    s0_IDM = param_data[:, 7]
    if time_varying:
        Tgap_rate = param_data[:, 3]
        v0_rate = param_data[:, 4]
    else:
        Tgap_rate = np.zeros(sim_num_veh)
        v0_rate = np.zeros(sim_num_veh)


    Nfut = Nfut_setting
    print(f"Constraint bounds: Spa[{SpaMin},{SpaMax}] Vel[{VelMin},{VelMax}] Acc[{AccMin},{AccMax}] Nfut={Nfut} sdes={SDES}")
    print(f"MPC weights: Q=diag([{dec_q_spa},{dec_q_vel}]) R={dec_r} RD={dec_rd}")
    print(f"Soft constraints / slack in optimization: {'ON' if enable_slack else 'OFF'}")
    os.makedirs(output_dir, exist_ok=True)
    run_settings = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'scenario': scenario,
        'control_method': 'Decentralized single MPC',
        'time_varying': time_varying,
        'Nfut': Nfut,
        'Npst': Npst,
        'SDES(desired/equilibrium spacing)': SDES,
        'Spa[min,max]': [SpaMin, SpaMax],
        'Vel[min,max]': [VelMin, VelMax],
        'Acc[min,max]': [AccMin, AccMax],
        'vehicle_csv': vehicle_csv_path,
        'subplatoon_config': sub_numHDV,
        'enable_slack': enable_slack,
        'slack_weights': {
            'W_SLACK': W_SLACK,
            'W_SLACK_L2': W_SLACK_L2,
        },
        'mpc_weights': {
            'dec_q_spa': dec_q_spa,
            'dec_q_vel': dec_q_vel,
            'dec_r': dec_r,
            'dec_rd': dec_rd,
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
    
    v_init = np.zeros(sim_num_veh)
    s_init = np.zeros(sim_num_veh)
    for i in range(sim_num_veh):
        v_init[i] = PV_vel[0] 
        s_init[i] = compute_IDM_steady_s(PV_vel[0], PV_vel[0], v0_IDM[i], Tgap[i], a_IDM[i], b_IDM[i], s0_IDM[i]) + veh_len[i] # Equilibrium value.
    print(f"v_init: {v_init}")
    print(f"s_init: {s_init}")
    
    # Each element in x_seq_list has shape (1, s_dim), with sim_subplatoon_num elements in total.
    S_history = generate_history_seq(sim_subplatoon_num,sim_num_veh,PV_vel[0],v_init,s_init,Tgap,v0_IDM,Tgap_rate,v0_rate,veh_len,Tstep,a_IDM,b_IDM,s0_IDM)

    # Define S to store simulation states.
    S = np.zeros([Npst+total_time_steps, sim_num_veh+1, 3]) # pos/vel/acc
    S[0:Npst+1,:,:] = S_history
    S[Npst:,0,1] = PV_vel
    # Compute PV acceleration.
    for i in range(Npst+total_time_steps-1):
        S[i,0,2] = (S[i+1,0,1]-S[i,0,1])/Tstep

    # ============================= Initialize VehicleSolver ==============================================
    Decentralized_solver = Decentralized_VehicleSolver(A, B, H, SpaMax, SpaMin, VelMax, VelMin, AccMin, AccMax, Nfut, scenario)

    # ============================= Simulation =============================================================
    control_compute_time_collection = np.zeros((sim_subplatoon_num, total_time_steps-1))
    slack_max_collection = np.zeros((sim_subplatoon_num, total_time_steps-1))

    for i in range(total_time_steps-1):
        Tgap_this_step = Tgap * (1 + Tgap_rate/100)**(i+Npst)
        v0_IDM_this_step = v0_IDM * (1 + v0_rate/100)**(i+Npst)
        # Update HDV acceleration according to IDM.
        acel = IDM_dynamics(S[Npst+i,:,:],Tgap_this_step,v0_IDM_this_step,veh_len,a_IDM,b_IDM,s0_IDM)
        S[Npst+i,1:,2] = acel

        ep_t = S[Npst+i,0,1]  # Pass PV speed as ep_t to the first subplatoon.
        # Apply MPC control separately to the CAV in each subplatoon.
        for platoon_idx in range(sim_subplatoon_num):
            lead_idx = sub_start[platoon_idx]
            cav_idx = lead_idx + 1
            x_t = np.zeros((2,))
            x_t[0] = S[Npst+i,lead_idx,0] - S[Npst+i,cav_idx,0] # CAV spacing.
            x_t[1] = S[Npst+i,cav_idx,1]                       # CAV speed.
            print(f"Subplatoon {platoon_idx}, step: {i}, x_t: {x_t}")
            
            # ============ Compute future ep estimates.
            ep = predict_future_ep_and_constraints(scenario, Nfut, S[Npst+i,lead_idx,1])

            # =========== Desired state.
            if scenario == 'stabilization':
                # The desired state in the cost is constant: both speed and spacing track predefined constants for traffic-wave stabilization.
                vdes = 25
                sdes = SDES

                x_ref = np.zeros((2, Nfut+1))
                x_ref[0,:] = sdes
                x_ref[1,:] = vdes

            elif scenario == 'tracking' or scenario == 'braking' or scenario == 'cutin':
                # Four desired states in the cost.
                sdes = SDES
                x_ref = np.zeros((2, Nfut+1))

                # Set x_ref velocity to the speed sequence of the vehicle ahead of the current subplatoon CAV, taken from the previous optimal predicted sequence.
                x_ref[0,:] = sdes
                if platoon_idx == 0:
                    x_ref[1,:] = ep_t
                elif platoon_idx != 0:
                    x_ref[1,:] = S[Npst+i,lead_idx,1]
                
            
            # ============= Solve.
            u_prev = np.array([S[Npst+i-1, cav_idx, 2]])
            x_next, u_next, compute_time, slack_vals = Decentralized_solver.solve(x_t, x_ref, ep, u_prev)
            print(f"Subplatoon {platoon_idx}, step {i}: u_next: {u_next[0,0]}, compute_time: {compute_time}")
            control_compute_time_collection[platoon_idx,i] = compute_time
            if slack_vals is not None:
                slack_max_collection[platoon_idx, i] = max(
                    float(np.max(v)) for v in slack_vals.values()
                )
            S[Npst+i,cav_idx,2] = u_next[0,0]

        # Update position and speed.
        S[Npst+i+1,:,0] = S[Npst+i,:,0] + Tstep * S[Npst+i,:,1]
        S[Npst+i+1,1:,1] = S[Npst+i,1:,1] + Tstep * S[Npst+i,1:,2]

        if scenario == 'cutin' and (i + 1) == cutin_step:
            S[Npst+i+1, 0, 0] -= cutin_drop
            print(f"[cut-in] Step {i+1}: leader position shifted backward by {cutin_drop:.1f} m; CAV1-PV spacing dropped sharply")

    print(f"Result output directory: {output_dir}")
    plot(scenario, S, Tstep, sim_subplatoon_num, sub_numHDV, out_dir=output_dir)

    # Save the simulation results
    save_vehicle_data_to_csv(S, filename_prefix=os.path.join(output_dir, 'vehicle_data'))

    metrics = compute_tracking_metrics(S[Npst:], scenario, sdes=SDES)
    metrics.update({
        'AvgControlTime_s': float(np.mean(control_compute_time_collection)),
        'dec_q_spa': dec_q_spa,
        'dec_q_vel': dec_q_vel,
        'dec_r': dec_r,
        'dec_rd': dec_rd,
        'SlackEnabled': int(enable_slack),
        'SlackMax': float(np.max(slack_max_collection)) if enable_slack else 0.0,
    })
    print("==================== Evaluation Metrics ====================")
    print(f"  RMSVE = {metrics['RMSVE']:.4f} m/s   RMSSE = {metrics['RMSSE']:.4f} m")
    print(f"  Avg control solve time = {metrics['AvgControlTime_s']*1000:.3f} ms")
    print("=================================================")
    save_metrics(metrics, filename=os.path.join(output_dir, 'metrics.csv'))

    # Save the computation time collection
    save_computation_time(control_compute_time_collection, "control", out_dir=output_dir)
    if enable_slack:
        slack_filename = os.path.join(output_dir, 'slack_max.csv')
        np.savetxt(slack_filename, slack_max_collection, delimiter=',')
        print(f"Slack max data saved to '{slack_filename}'")
        
