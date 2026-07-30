import numpy as np
import math
import time
import os
import itertools
import pandas as pd
from matplotlib import pyplot as plt

# Get the project root directory, which is two levels up from the current file's directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(f'project_root: {project_root}')

# =============================================================================
#  Decentralized Robust DeeP-LCC (dDeeP-LCC), Shang, Wang & Zheng, 2024
#  "Decentralized Robust Data-driven Predictive Control for Smoothing Mixed
#   Traffic Flow", arXiv:2401.15826.
#
#  Difference w.r.t. the distributed DeeP-LCC (Wang 2023) baseline:
#  ----------------------------------------------------------------------------
#   * No ADMM / no inter-subsystem communication. Each CF-LCC subsystem solves
#     its OWN local optimisation independently (Algorithm 1).
#   * The coupling from the preceding vehicle (its future velocity error eps_i)
#     is NOT exchanged; instead it is bounded by an uncertainty SET W_i that is
#     re-estimated online from the past disturbance eps_i,ini (Section IV).
#   * The local problem is a robust min-max QP (eq. (16)) that guarantees the
#     safety (spacing) constraint for every disturbance in W_i. It is solved
#     with the vertex-based strategy (Method I, Prop. 3 / eq. (24)) after a
#     down-sampling approximation of the disturbance set (Section IV-B).
#
#  This file shares the IDM-based mixed-traffic simulation framework, the
#  scenarios and the pre-collected Hankel data with ../distributed_DeePC so the
#  two baselines can be compared on an equal footing.
# =============================================================================

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


scenario = os.environ.get('KMPC_SCENARIO', 'braking')
subplatoon_spec = _env_int_list('KMPC_SUBPLATOON', [2, 3, 1, 4, 2])
layout_slug = '-'.join(str(x) for x in subplatoon_spec)
vehicle_csv = os.environ.get('KMPC_VEHICLE_CSV', 'vehicle_parameters.csv')

# Disturbance-set estimation method (Section IV-A):
#   'zero'         -> singleton {eps_cur}; equivalent to nominal DeeP-LCC per subsystem
#   'constant'     -> constant-velocity model, constant bound over the horizon
#   'time-varying' -> constant-acceleration model, time-varying (growing) bound
disturbance_method = os.environ.get('DDEEPC_DISTURBANCE_METHOD', 'time-varying')

time_varying = _env_flag('KMPC_TIME_VARYING', True)

# Parameter in Simulation
Tstep       = 0.12 # Time Step
ID = np.array([item for nh in subplatoon_spec for item in ([1] + [0] * nh)])  # ID of vehicle types 1: CAV  0: HDV
pos_cav     = np.where(ID == 1)[0]         # position of CAVs from PV backward
n_vehicle   = len(ID)                      # number of vehicles
n_cav       = len(pos_cav)                 # number of CAVs
n_hdv       = n_vehicle-n_cav              # number of HDVs
n_vehicle_sub = [1 + nh for nh in subplatoon_spec] # number of vehicles in each subsystem

# DeePC Parameters
T = _env_int('DDEEPC_DATA_T', _env_int('DEEPC_DATA_T', 500))             # length of data samples
Tini = 50           # length of past data
N = _env_int('KMPC_NFUT', 15)              # length of predicted horizon
kappa = T-Tini-N+1  # number of columns of Hankel matrix
if kappa <= 0:
    raise ValueError(f"DDEEPC_DATA_T={T} must be larger than Tini+N-1={Tini+N-1}")

# constant parameter
s_limit = [_env_float('KMPC_SPA_MIN', 10), _env_float('KMPC_SPA_MAX', 80)]
u_limit =  [_env_float('KMPC_ACC_MIN', -6), _env_float('KMPC_ACC_MAX', 4)]

# ---- robust dDeeP-LCC parameters ----
Ts_down = _env_int('DDEEPC_TS_DOWN', 7)         # down-sampling step for the disturbance set (Section IV-B)
                    # nv = 2**n_eps vertices, so keep n_eps small (Ts=7 -> n_eps=3, 8 vertices)
reg_pinv = _env_float('DDEEPC_REG_PINV', 1e-4)     # small Tikhonov regularisation for the constraint-elimination
                    # pseudo-inverse g = H^T (H H^T + reg_pinv*I)^{-1} b.

# parameters
if scenario == 'tracking' or scenario == 'braking' or scenario == 'cutin':
    lambda_gi       = 100   # 100
    lambda_yi       = 1e5   # penalty on ||sigma_y||_2^2 in objective 1e5
elif scenario == 'stabilization':
    lambda_gi       = 200
    lambda_yi       = 2*1e5   # penalty on ||sigma_y||_2^2 in objective 1e4

weight_s = _env_float('KMPC_DEC_Q_SPA', 5.0)
weight_v = _env_float('KMPC_DEC_Q_VEL', 1.5)
weight_u = _env_float('KMPC_DEC_R', 20.0)
if scenario == 'stabilization':
    weight_s = _env_float('DDEEPC_STAB_Q_SPA', 1.0)
    weight_v = _env_float('DDEEPC_STAB_Q_VEL', 2.0)
    weight_u = _env_float('DDEEPC_STAB_R', 12.5)

data_file = os.environ.get('DDEEPC_DATA_DIR', f'precollected_data_{scenario}_{layout_slug}_idm_input_T{T}_N{N}')

# Load parameters from CSV
vehicle_csv_path = vehicle_csv if os.path.isabs(vehicle_csv) else os.path.join(project_root, vehicle_csv)
param_data = np.genfromtxt(vehicle_csv_path, delimiter=',', skip_header=1)
param_data = np.atleast_2d(param_data)
if param_data.shape[0] < n_vehicle:
    raise ValueError(f"{vehicle_csv_path}: row count {param_data.shape[0]} < required vehicle count {n_vehicle}")
if param_data.shape[1] < 8:
    raise ValueError(f"{vehicle_csv_path}: expected columns Tgap,v0_IDM,veh_len,Tgap_rate,v0_rate,a_IDM,b_IDM,s0_IDM")
param_data = param_data[:n_vehicle, :]
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
    Tgap_rate = np.zeros(n_vehicle)
    v0_rate = np.zeros(n_vehicle)

Tgap_list,v0_IDM_list,veh_len_list,Tgap_rate_list,v0_rate_list,a_IDM_list,b_IDM_list,s0_IDM_list = [],[],[],[],[],[],[],[]
for i in range(n_cav):
    if i != n_cav-1:
        Tgap_list.append(Tgap[pos_cav[i]:pos_cav[i+1]])      # the following vehicle is CAV
        v0_IDM_list.append(v0_IDM[pos_cav[i]:pos_cav[i+1]])
        veh_len_list.append(veh_len[pos_cav[i]:pos_cav[i+1]])
        Tgap_rate_list.append(Tgap_rate[pos_cav[i]:pos_cav[i+1]])
        v0_rate_list.append(v0_rate[pos_cav[i]:pos_cav[i+1]])
        a_IDM_list.append(a_IDM[pos_cav[i]:pos_cav[i+1]])
        b_IDM_list.append(b_IDM[pos_cav[i]:pos_cav[i+1]])
        s0_IDM_list.append(s0_IDM[pos_cav[i]:pos_cav[i+1]])
    else:
        Tgap_list.append(Tgap[pos_cav[i]:n_vehicle])         # the following vehicle is HDV
        v0_IDM_list.append(v0_IDM[pos_cav[i]:n_vehicle])
        veh_len_list.append(veh_len[pos_cav[i]:n_vehicle])
        Tgap_rate_list.append(Tgap_rate[pos_cav[i]:n_vehicle])
        v0_rate_list.append(v0_rate[pos_cav[i]:n_vehicle])
        a_IDM_list.append(a_IDM[pos_cav[i]:n_vehicle])
        b_IDM_list.append(b_IDM[pos_cav[i]:n_vehicle])
        s0_IDM_list.append(s0_IDM[pos_cav[i]:n_vehicle])

# Generate a Hankel matrix of order L
def Hankel_matrix(u,L):
    m = u.shape[0]      # the dimension of signal
    T = u.shape[1]      # the length of data
    U = np.zeros([m*L,T-L+1])

    for i in range(L):
        U[i*m:(i+1)*m,:] = u[:,i:(T-L+1+i)]

    return U

def IDM_dynamics(S, Tgap, v0, veh_len, a_idm, b_idm, s0_idm):
    '''
    Inputs
    S: all-vehicle state data at time k, with state entries for position s and velocity v.
    Tgap: car-following parameter vector for the current subsystem's following vehicles.
    veh_len: vehicle lengths for the current subsystem's following vehicles.
    a_idm/b_idm/s0_idm: heterogeneous IDM parameters for each vehicle.

    Output
    acel: acceleration vector.
    '''

    # limitation of actuators
    acel_max = 4
    dcel_max = -6

    V_diff = S[0:-1,1] - S[1:,1] # the velocity error with former car
    D_diff = S[0:-1,0] - S[1:,0] - veh_len # the pos error with former car

    s_star = s0_idm + np.maximum(0, Tgap*S[1:,1] - (S[1:,1]*V_diff)/(2*np.sqrt(a_idm*b_idm)))
    acel = a_idm*(1-(S[1:,1]/v0)**4 - (s_star/D_diff)**2)

    # acceleration saturation
    acel = np.where(acel > acel_max, acel_max, acel)
    acel = np.where(acel < dcel_max, dcel_max, acel)

    return acel

def compute_IDM_steady_s(v_PV, v, v0, Tgap, a_idm, b_idm, s0_idm):
    s_star = s0_idm + v*Tgap - (v*(v_PV-v))/(2*np.sqrt(a_idm*b_idm))
    s = s_star/(np.sqrt(1-(v/v0)**4))
    return s

def generate_NEDC_velocity_profile(Tstep=0.12):
    segments = [10, 8, 40, 13, 40, 25, 40, 10, 80]
    total_time = sum(segments)
    time = np.arange(1, total_time, Tstep)
    vel = np.zeros_like(time)

    thresholds = np.cumsum(segments)

    for idx, t in enumerate(time):
        if t <= thresholds[0]:
            vel[idx] = 70
        elif t <= thresholds[1]:
            vel[idx] = 70 - 20/8*(t - thresholds[0])
        elif t <= thresholds[2]:
            vel[idx] = 50
        elif t <= thresholds[3]:
            vel[idx] = 50 + 20/13*(t - thresholds[2])
        elif t <= thresholds[4]:
            vel[idx] = 70
        elif t <= thresholds[5]:
            vel[idx] = 70 + 30/25*(t - thresholds[4])
        elif t <= thresholds[6]:
            vel[idx] = 100
        elif t <= thresholds[7]:
            vel[idx] = 100 - 30/10*(t - thresholds[6])
        elif t <= thresholds[8]:
            vel[idx] = 70

    vel = vel/3.6

    return time, vel

def generate_sine_velocity_profile(Tstep=0.12):
    total_time = 180     # seconds
    time = np.arange(0, total_time, Tstep)
    vel = np.zeros_like(time)

    # Sine-wave parameters.
    center = 25         # Center speed: 25 m/s.
    amplitude = 3       # Oscillation range: +/-5 m/s.
    periods = 4         # Four complete sine waves.
    omega = 2 * np.pi * periods / total_time   # Angular frequency.

    # Generate the sine curve.
    for idx, t in enumerate(time):
        vel[idx] = center + amplitude * np.sin(omega * t)

    return time, vel

def generate_braking_velocity_profile(Tstep=0.12):
    """
    Emergency braking scenario (total 30 s):
      Phase 1 [0,      1):    constant 20 m/s
      Phase 2 [1,     13):    braking at -5 m/s², 20 → 5 m/s
      Phase 3 [13,     18):    constant 5 m/s
      Phase 4 [18,     21):    acceleration at +5 m/s², 5 → 20 m/s
      Phase 5 [21,     30):    constant 20 m/s
    """
    v_high  = 20.0   # m/s
    v_low   = 8.0    # m/s
    a_brake = -4.0   # m/s²
    a_accel =  4.0   # m/s²

    # segment durations (s)
    t_const1  = 1.0
    t_brake   = (v_low - v_high) / a_brake   # = 3 s
    t_const2  = 5.0
    t_accel   = (v_high - v_low) / a_accel   # = 3 s
    t_const3  = 100.0

    segments = [t_const1, t_brake, t_const2, t_accel, t_const3]
    total_time = sum(segments)   # = 30 s
    thresholds = np.cumsum(segments)

    time = np.arange(0, total_time, Tstep)
    vel  = np.zeros_like(time)

    for idx, t in enumerate(time):
        if t < thresholds[0]:
            vel[idx] = v_high
        elif t < thresholds[1]:
            vel[idx] = v_high + a_brake * (t - thresholds[0])
        elif t < thresholds[2]:
            vel[idx] = v_low
        elif t < thresholds[3]:
            vel[idx] = v_low + a_accel * (t - thresholds[2])
        else:
            vel[idx] = v_high

    return time, vel

def generate_cutin_velocity_profile(Tstep=0.12, cutin_time=8.0):
    """
    Cut-in scenario with the same velocity profile as Koopman_MPC_IDM_learning_set.py.
    The position jump is injected in the main loop.
    """
    v_const = 20.0
    total_time = 80.0
    accel_dur = 3.0
    a_accel = 2.0
    a_brake = -5.0
    v_low = 8.0

    t_accel = cutin_time - accel_dur
    v_peak = v_const + a_accel * accel_dur
    t_brake = (v_peak - v_low) / (-a_brake)
    t_brake_end = cutin_time + t_brake

    time = np.arange(0, total_time, Tstep)
    vel = np.empty_like(time)
    for idx, t in enumerate(time):
        if t < t_accel:
            vel[idx] = v_const
        elif t < cutin_time:
            vel[idx] = v_const + a_accel * (t - t_accel)
        elif t < t_brake_end:
            vel[idx] = v_peak + a_brake * (t - cutin_time)
        else:
            vel[idx] = v_low
    return time, vel


# =============================================================================
#               Robust disturbance-set estimation (Section IV)
# =============================================================================
def build_downsample_matrix(N, Ts):
    """Linear-interpolation (down-sampling) matrix E_eps in (19).

    The N-dimensional future disturbance trajectory eps is approximated by a
    low-dimensional vector eps_tilde in R^{n_eps}, sampled every Ts steps along
    the horizon (always including the last step), via eps ~= E_eps @ eps_tilde.
    Keeping n_eps small is essential because the vertex-based robust strategy
    enumerates 2**n_eps vertices of the disturbance polytope.

    Returns E_eps (N x n_eps) and the list of sample indices (length n_eps).
    """
    nodes = list(range(0, N - 1, Ts))
    if nodes[-1] != N - 1:
        nodes.append(N - 1)
    n_eps = len(nodes)
    E = np.zeros((N, n_eps))
    for k in range(N):
        # locate the bracketing sample nodes and linearly interpolate
        j = 0
        while j < n_eps - 1 and nodes[j + 1] <= k:
            j += 1
        if j == n_eps - 1:
            E[k, j] = 1.0
        else:
            left, right = nodes[j], nodes[j + 1]
            w = (k - left) / (right - left)
            E[k, j] = 1.0 - w
            E[k, j + 1] = w
    return E, nodes


def estimate_disturbance_bounds(eps_ini, N, Tstep, method):
    """Estimate the future disturbance bounds [eps_min, eps_max] in R^N from the
    past disturbance trajectory eps_ini (the preceding vehicle's velocity),
    following the three methods of Section IV-A.

    Note: here eps is the actual preceding-vehicle velocity rather than the
    velocity error, so the bounds are expressed around eps_cur = eps_ini[-1].
    """
    eps_cur = eps_ini[-1]
    if method == 'zero':
        # Maintain the current velocity (singleton set) -> nominal prediction.
        eps_min = np.full(N, eps_cur)
        eps_max = np.full(N, eps_cur)
    elif method == 'constant':
        # Constant-velocity model: constant bound over the horizon.
        d_low = np.min(eps_ini) - np.mean(eps_ini)
        d_up  = np.max(eps_ini) - np.mean(eps_ini)
        eps_min = np.full(N, eps_cur + d_low)
        eps_max = np.full(N, eps_cur + d_up)
    elif method == 'time-varying':
        # Constant-acceleration model: time-varying (growing) bound.
        a_ini = np.diff(eps_ini) / Tstep
        a_cur = a_ini[-1]
        da_low = np.min(a_ini) - np.mean(a_ini)
        da_up  = np.max(a_ini) - np.mean(a_ini)
        steps = np.arange(1, N + 1) * Tstep
        eps_min = eps_cur + (a_cur + da_low) * steps
        eps_max = eps_cur + (a_cur + da_up) * steps
    else:
        raise ValueError(f'unknown disturbance_method: {method}')
    return eps_min, eps_max


def disturbance_vertices(eps_min, eps_max, nodes):
    """Return the 2**n_eps vertices of the down-sampled disturbance polytope
    W_tilde, used by the vertex-based strategy (Prop. 3). Each vertex is an
    eps_tilde vector in R^{n_eps} (the low-dimensional set); the precomputed
    G_eps map lifts it back to the full horizon when forming g."""
    lo = eps_min[nodes]
    hi = eps_max[nodes]
    n_eps = len(nodes)
    verts = []
    for mask in itertools.product((0, 1), repeat=n_eps):
            verts.append(np.where(np.array(mask) == 0, lo, hi))
    return verts

def precollected_data_exists():
    if not os.path.isdir(data_file):
        return False
    for i in range(n_cav):
        expected_rows = {
            'Ui': Tini + N,
            'Ei': Tini + N,
            'Yi': (Tini + N) * 2 * n_vehicle_sub[i],
        }
        for prefix, rows in expected_rows.items():
            path = os.path.join(data_file, f'{prefix}_{i}_moderate.csv')
            if not os.path.exists(path):
                return False
            arr = np.genfromtxt(path, delimiter=',', skip_header=0)
            arr = np.atleast_2d(arr)
            if arr.shape != (rows, kappa):
                return False
    return True


# Initialize each subsystem led by a CAV.
def init_sub(scenario):
    # Generate the PV trajectory according to the simulation scenario.
    if scenario == 'tracking':
        time_list, PV_vel = generate_NEDC_velocity_profile(Tstep)
    elif scenario == 'braking':
        time_list, PV_vel = generate_braking_velocity_profile(Tstep)
    elif scenario == 'stabilization':
        time_list, PV_vel = generate_sine_velocity_profile(Tstep)
    elif scenario == 'cutin':
        time_list, PV_vel = generate_cutin_velocity_profile(Tstep, cutin_time=8.0)
    total_time_steps = len(time_list)
    sim_maxtime = _env_float('KMPC_SIM_MAXTIME', 0.0)
    if sim_maxtime > 0:
        cap = int(round(sim_maxtime / Tstep))
        if 0 < cap < total_time_steps:
            time_list = time_list[:cap]
            PV_vel = PV_vel[:cap]
            total_time_steps = cap
            print(f"[sim-cap] Truncated simulation to {sim_maxtime:.1f}s ({total_time_steps} steps)")

    S_total = np.zeros([Tini+total_time_steps, n_vehicle+1, 3])   # state matrix: [position, velocity, acceleration]
    S_total[0,0,0] = 0 # initial position of the platoon
    S_total[0:Tini,0,1] = PV_vel[0]  # initial velocity of the head vehicle
    S_total[Tini:,0,1] = PV_vel.squeeze()  # velocity profile of the head vehicle

    v_init = np.zeros(n_vehicle)
    s_init = np.zeros(n_vehicle)
    for i in range(n_vehicle):
        v_init[i] = PV_vel[0]
        s_init[i] = compute_IDM_steady_s(PV_vel[0], PV_vel[0], v0_IDM[i], Tgap[i], a_IDM[i], b_IDM[i], s0_IDM[i]) + veh_len[i] # Equilibrium value.
    for i in range(1,n_vehicle+1):
        S_total[0,i,0] = S_total[0,i-1,0] - s_init[i-1] # initial position
    S_total[0,1:,1] = v_init # initial velocity

    # Initial trajectory
    uini = np.zeros([n_cav,Tini])
    eini = S_total[0:Tini,0,1] # velocity of the head vehicles
    yini = np.zeros([2*n_vehicle,Tini])

    # Initial Tini-step trajectory.
    for k in range(Tini):
        # update acceleration
        Tgap_this_step = Tgap * (1 + Tgap_rate/100)**k
        v0_IDM_this_step = v0_IDM * (1 + v0_rate/100)**k
        acel = IDM_dynamics(S_total[k,:,:],Tgap_this_step,v0_IDM_this_step,veh_len,a_IDM,b_IDM,s0_IDM)
        S_total[k,0,2] = 0                # the head vehicle has 0 acc
        S_total[k,1:,2] = acel            # all the vehicles using HDV model

        # The initial Tini-step trajectory is generated by IDM.
        uini[:,k] = S_total[k,pos_cav+1,2]

        S_total[k+1,1:,1] = S_total[k,1:,1] + Tstep*S_total[k,1:,2]
        S_total[k+1,:,0] = S_total[k,:,0] + Tstep*S_total[k,:,1]    # update position

        yini[:, k] = np.ravel(np.column_stack((S_total[k+1, :-1, 0] - S_total[k+1, 1:, 0], S_total[k+1, 1:, 1])))

    # down-sampling matrix (shared by all subsystems)
    E_down, down_nodes = build_downsample_matrix(N, Ts_down)

    # Per-subsystem precomputation for the local robust DeeP-LCC
    Qi_stack = [[] for _ in range(n_cav)]
    Ri_stack = [[] for _ in range(n_cav)]
    Pi = [[] for _ in range(n_cav)]            # spacing selector G1 in (14c)
    args_list = [[] for _ in range(n_cav)]

    if _env_flag('DDEEPC_REGENERATE_DATA', False) or not precollected_data_exists():
        print(f"[dDeeP-LCC] Pre-collected data is missing or regeneration was requested; generating into '{data_file}'")
        data_collection()

    for i in range(n_cav):
        Ui_temp = np.genfromtxt(os.path.join(data_file, 'Ui_'+ str(i) + '_moderate.csv'), delimiter=",", skip_header=0)
        Uip = Ui_temp[0:Tini,:]
        Uif = Ui_temp[Tini:,:]

        Ei_temp = np.genfromtxt(os.path.join(data_file, 'Ei_'+ str(i) + '_moderate.csv'), delimiter=",", skip_header=0)
        Eip = Ei_temp[0:Tini,:]
        Eif = Ei_temp[Tini:,:]

        Yi_temp = np.genfromtxt(os.path.join(data_file, 'Yi_'+ str(i) + '_moderate.csv'), delimiter=",", skip_header=0)
        linenum = int(Tini*2*n_vehicle_sub[i])
        Yip = Yi_temp[0:linenum,:]
        Yif = Yi_temp[linenum:,:]

        # cost weights: Q penalises spacing/velocity tracking, R penalises input
        pattern = np.array([weight_s, weight_v])
        weights = np.tile(pattern, n_vehicle_sub[i])
        Qi = np.diagflat(weights)
        Qi_stack[i] = np.kron(np.eye(N),Qi)
        Ri_stack[i] = weight_u * np.eye(N)
        Pi[i] = np.kron(np.eye(N),np.append([1],np.zeros(2*n_vehicle_sub[i]-1)))  # select CAV spacing

        # ----- constraint elimination (Section V-A), z = 0 (least-norm g) -----
        # H_i = col(Uip, Eip, Yip, Uif, Eif); g = H_i^dagger b,
        # b = col(uini, eini, yini+sigma_y, u, E_eps eps_tilde), y = Yif g.
        # Split the columns of H_i^dagger to express g affinely in the decision
        # variables (sigma_y, u) and the disturbance eps_tilde.
        Hi = np.vstack((Uip, Eip, Yip, Uif, Eif))
        # Tikhonov-regularised right pseudo-inverse (see reg_pinv note above)
        nrow = Hi.shape[0]
        Hi_pinv = Hi.T @ np.linalg.inv(Hi @ Hi.T + reg_pinv*np.eye(nrow))
        p_sub = 2*n_vehicle_sub[i]
        c_uini = Tini
        c_eini = Tini
        c_yini = p_sub*Tini
        c_u = N
        idx = 0
        G_uini = Hi_pinv[:, idx:idx+c_uini]; idx += c_uini
        G_eini = Hi_pinv[:, idx:idx+c_eini]; idx += c_eini
        G_yini = Hi_pinv[:, idx:idx+c_yini]; idx += c_yini
        G_u    = Hi_pinv[:, idx:idx+c_u];    idx += c_u
        G_eif  = Hi_pinv[:, idx:]                       # future-disturbance block (N)
        G_eps  = G_eif @ E_down                         # map eps_tilde (n_eps) -> g

        # ----- reduce to a small dense QP in z = (u, sigma_y) (Appendix A) -----
        # With g = g_const + Bg z + G_eps w and y = Yif g, the cost
        #   V_j(z) = ||u||_R^2 + ||y_j - yref||_Q^2 + lg||g_j||^2 + ly||sigma_y||^2
        # has a vertex-INDEPENDENT Hessian M in z, so the worst case over the
        # vertices is  quad(z, M) + max_j ( d_j^T z + c0_j ).  This turns the
        # robust problem (24) into a pure QP (no quadratic constraints).
        nz = N + p_sub*Tini
        Su   = np.hstack((np.eye(N), np.zeros((N, p_sub*Tini))))     # picks u
        Ssig = np.hstack((np.zeros((p_sub*Tini, N)), np.eye(p_sub*Tini)))  # picks sigma_y
        Bg = np.hstack((G_u, G_yini))      # g = g_const + Bg z + G_eps w
        By = Yif @ Bg                      # y = ay + By z + Ye w
        Ye = Yif @ G_eps
        Q = Qi_stack[i]; R = Ri_stack[i]
        M = (Su.T @ R @ Su + By.T @ Q @ By
             + lambda_gi * (Bg.T @ Bg) + lambda_yi * (Ssig.T @ Ssig))
        M = 0.5*(M + M.T)
        Dw  = 2.0*(By.T @ Q @ Ye + lambda_gi * (Bg.T @ G_eps))   # nz x n_eps
        Pby = Pi[i] @ By                                          # N x nz
        PiYe = Pi[i] @ Ye                                         # N x n_eps
        # Cholesky factor of the (shared) Hessian: quad(z, M) = ||Lc^T z||^2.
        # The Hankel pseudo-inverse makes M ill-conditioned (cond ~ 1e10); the
        # robust QP is therefore solved with the interior-point solver Clarabel.
        Lc = np.linalg.cholesky(M + 1e-9*np.eye(nz))

        # init data for this subsystem
        ui_ini = uini[i,:]
        if i == 0:
            ei_ini = eini.copy()
        else:
            ei_ini = S_total[0:Tini,pos_cav[i],1].copy()
        if i != n_cav-1:
            yi_ini = yini[2*pos_cav[i]:2*pos_cav[i+1],:]
        else:
            yi_ini = yini[2*pos_cav[i]:,:]

        # local sub-platoon state slice S (CAV head + following HDVs + leader)
        if i == 0:
            S = S_total[:,0:n_vehicle_sub[i]+1,:]
        else:
            S = S_total[:,pos_cav[i]:pos_cav[i]+n_vehicle_sub[i]+1,:]

        precomp = {
            'Yif': Yif, 'Pi': Pi[i], 'Qi_stack': Qi_stack[i],
            'G_uini': G_uini, 'G_eini': G_eini, 'G_yini': G_yini,
            'G_u': G_u, 'G_eps': G_eps,
            'down_nodes': down_nodes, 'p_sub': p_sub, 'nz': nz,
            # reduced QP data in z = (u, sigma_y) coordinates
            'Dw': Dw, 'Pby': Pby, 'PiYe': PiYe, 'Lc': Lc,
            'By': By, 'Ye': Ye, 'Bg': Bg, 'Q': Qi_stack[i], 'lambda_gi': lambda_gi,
        }
        args_list[i] = [S, ui_ini, ei_ini, yi_ini, precomp]

    return args_list

def data_collection():
    # Full-state measurement, including speed and position errors of all vehicles.
    m_ctr = n_cav          # number of input variables u(t)
    p_ctr = 2*n_vehicle    # number of output variables

    if scenario == 'tracking':
        v_init = 70/3.6
    elif scenario == 'braking':
        v_init = 20
    elif scenario == 'stabilization':
        v_init = 25
    elif scenario == 'cutin':
        v_init = 20

    S = np.zeros([T,n_vehicle+1,3])
    S[0,0,0] = 0
    S[0,:,1] = v_init * np.ones(n_vehicle+1) # initial velocity = v_init
    for i in range(1,n_vehicle+1):
        S[0,i,0] = S[0,i-1,0] - (compute_IDM_steady_s(S[0,i-1,1],S[0,i,1],v0_IDM[i-1],Tgap[i-1],a_IDM[i-1],b_IDM[i-1],s0_IDM[i-1]) + veh_len[i-1] + np.random.uniform(-10, 10)) # initial position

    # Match distributed_DeePC data generation: all vehicles evolve under IDM,
    # and the CAV input signal is the resulting IDM acceleration at pos_cav.
    ud = np.zeros([m_ctr, T])   # CAV acceleration [m/s^2]

    # eps^d: random persistently-exciting head velocity around v_init
    ed = v_init + 5 * np.random.uniform(-1, 1, size=T)

    yd = np.zeros([p_ctr,T])
    S[:,0,1] = ed

    # generate output data
    for k in range(T-1):
        acel = IDM_dynamics(S[k,:,:], Tgap, v0_IDM, veh_len, a_IDM, b_IDM, s0_IDM) + np.random.uniform(-0.2, 0.2, size=n_vehicle) # add some noise to the acceleration
        S[k,0,2] = 0                # the head vehicle has ignored acc
        ud[:,k] = acel[pos_cav]     # record IDM-generated CAV acceleration
        S[k,1:,2] = acel

        S[k+1,1:,1] = S[k,1:,1] + Tstep*S[k,1:,2]   # update velocity
        S[k+1,:,0] = S[k,:,0] + Tstep*S[k,:,1]      # update position
        yd[:, k] = np.ravel(np.column_stack((S[k, :-1, 0] - S[k, 1:, 0], S[k, 1:, 1])))

    k = k+1
    yd[:,k] = np.ravel(np.column_stack((S[k, :-1, 0] - S[k, 1:, 0], S[k, 1:, 1])))
    plot(scenario,S,Tstep,n_cav,data_file,subplatoon_spec)

    # construct distributed data
    ui_d, yi_d, ei_d = [], [], []

    for i in range(n_cav):
        ui_d.append(np.array([ud[i,:]]))

        if i != n_cav-1:
            yi_d.append(yd[2*pos_cav[i]:2*pos_cav[i+1],:])
        else:
            yi_d.append(yd[2*pos_cav[i]:2*n_vehicle,:])

        if i == 0:                          # the first subsystem
            ei_d.append(np.array([ed]))     # velocity error of the head
        else:
            ei_d.append(np.array([yd[2*pos_cav[i]-1,:]])) # velocity of the veh before the subsystem

    # data Hankel matrices for the decentralized DeeP-LCC
    Ui, Ei, Yi = [], [], []
    save_dir = data_file
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n_cav):
        Ui.append(Hankel_matrix(ui_d[i], Tini + N))
        Ei.append(Hankel_matrix(ei_d[i], Tini + N))
        Yi.append(Hankel_matrix(yi_d[i], Tini + N))

        np.savetxt(os.path.join(save_dir, f'Ui_{i}_moderate.csv'), Ui[i], fmt='%.6f', delimiter=',')
        np.savetxt(os.path.join(save_dir, f'Ei_{i}_moderate.csv'), Ei[i], fmt='%.6f', delimiter=',')
        np.savetxt(os.path.join(save_dir, f'Yi_{i}_moderate.csv'), Yi[i], fmt='%.6f', delimiter=',')

def save_vehicle_data_to_csv(S, filename_prefix):
    """
    Save vehicle trajectory data to separate CSV files for each vehicle.
    """
    num_vehicles = S.shape[1] - 1
    os.makedirs(filename_prefix, exist_ok=True)

    for veh_idx in range(0, num_vehicles + 1):
        vehicle_data = S[:, veh_idx, :]  # Shape: (total_time_steps, 3)
        df = pd.DataFrame(vehicle_data, columns=['position', 'velocity', 'acceleration'])
        filename = f'{filename_prefix}/veh_{veh_idx}.csv'
        df.to_csv(filename, index=False)

    print(f"Saved {num_vehicles + 1} vehicle data files to '{filename_prefix}' directory")

def compute_spacing_from_position(S):
    """Compute spacing s_i(k)=p_{i-1}(k)-p_i(k) from positions; vehicle 0 is the PV."""
    pos = S[:, :, 0]
    spacing = np.full_like(pos, np.nan, dtype=float)
    spacing[:, 1:] = pos[:, :-1] - pos[:, 1:]
    return spacing

def compute_tracking_metrics(S, scenario, sdes=40.0, vdes_stab=25.0):
    """Match the Koopman baseline: exclude the PV and compute RMSVE/RMSSE over the supplied time window."""
    vel = S[:, :, 1]
    spacing = compute_spacing_from_position(S)
    vel_eval = vel[:, 1:]
    spacing_eval = spacing[:, 1:]

    if scenario == 'stabilization':
        vel_err = vel_eval - vdes_stab
    else:
        vdes = vel[:, 0]
        vel_err = vel_eval - vdes[:, None]

    rmsve = float(np.sqrt(np.mean(vel_err ** 2)))
    rmsse = float(np.sqrt(np.mean((spacing_eval - sdes) ** 2)))
    return {'RMSVE': rmsve, 'RMSSE': rmsse}

def save_metrics(metrics, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pd.DataFrame(list(metrics.items()), columns=['metric', 'value']).to_csv(filename, index=False)
    print(f"Evaluation metrics saved to '{filename}'")

def plot(scenario, S, Tstep, sim_subplatoon_num, file, sub_numHDV_list=None):
    # Time sequence.
    os.makedirs(file, exist_ok=True)
    time_list = np.arange(S.shape[0]) * Tstep

    # Generate vehicle labels based on subplatoon structure
    veh_labels = ['Leader']  # Start with the leader vehicle
    veh_color = ['black']
    HDV_idx = 1  # Start counting from 1 (after leader)

    my_colors_cav = [
    '#0066CC',  # Bright blue.
    '#E35A58',  # Slightly bright dark red.
    '#009999',  # Teal green.
    '#7A42F4',  # Purple.
    '#FF8000'   # Orange.
    ]
    my_colors_veh = plt.cm.Greys(np.linspace(0.6,0.3,S.shape[1]-1))

    if sub_numHDV_list is None:
        sub_numHDV_list = subplatoon_spec
    if len(sub_numHDV_list) != sim_subplatoon_num:
        raise ValueError("sub_numHDV_list length must match sim_subplatoon_num")

    for subplatoon_idx in range(sim_subplatoon_num):
        # Add CAV for this subplatoon
        veh_labels.append(f'CAV_{subplatoon_idx + 1}')
        veh_color.append(my_colors_cav[subplatoon_idx % len(my_colors_cav)])  # Example color for CAV

        # Add HDVs for this subplatoon
        for hdv_idx in range(sub_numHDV_list[subplatoon_idx]):
            veh_labels.append(f'HDV_{HDV_idx}')
            veh_color.append(my_colors_veh[HDV_idx])  # Example color for HDV
            HDV_idx += 1

    # Each state type: position, speed, and acceleration.
    state_names = ['Position', 'Velocity', 'Acceleration']

    for state_idx, state_name in enumerate(state_names):
        print("Generating figure:", state_name)
        plt.figure(figsize=(10, 6))
        for veh_idx in range(S.shape[1]):
            plt.plot(time_list, S[:, veh_idx, state_idx], label=veh_labels[veh_idx], color=veh_color[veh_idx])
        plt.xlabel('Time (s)')
        plt.ylabel(state_name)
        plt.title(f'{state_name} over Time')
        plt.legend()
        if state_name == 'Velocity':
            if scenario == 'tracking':
                plt.ylim(5,35)
            elif scenario == 'braking':
                plt.ylim(0,40)
            elif scenario == 'stabilization':
                plt.ylim(15,35)
            elif scenario == 'cutin':
                plt.ylim(0,30)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{file}/plot_{state_name}.png')
        plt.close()

    # Compute vehicle spacing as leader position minus follower position.
    print("Generating figure: Vehicle Spacing")
    plt.figure(figsize=(10, 6))
    for i in range(1, S.shape[1]):  # Start from vehicle 1 and compute spacing with its leader.
        spacing = S[:, i-1, 0] - S[:, i, 0]  # Position difference.
        label = f'{veh_labels[i-1]} - {veh_labels[i]}'
        plt.plot(time_list, spacing, label=label, color=veh_color[i])

    plt.xlabel('Time (s)')
    plt.ylabel('Inter-vehicle Distance (m)')
    plt.title('Vehicle Spacing over Time')
    plt.legend()
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{file}/plot_Spacing.png')
    plt.close()

if __name__ == "__main__":
    data_collection()
