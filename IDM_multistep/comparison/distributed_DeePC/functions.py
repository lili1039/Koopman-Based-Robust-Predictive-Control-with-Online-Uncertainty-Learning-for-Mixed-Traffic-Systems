import numpy as np
import math
import time
import os
import pandas as pd
from matplotlib import pyplot as plt

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


scenario = os.environ.get('KMPC_SCENARIO', 'braking')
subplatoon_spec = _env_int_list('KMPC_SUBPLATOON', [2, 3, 1, 4, 2])
layout_slug = '-'.join(str(x) for x in subplatoon_spec)
vehicle_csv = os.environ.get('KMPC_VEHICLE_CSV', 'vehicle_parameters.csv')
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
T = _env_int('DEEPC_DATA_T', 500)             # length of data samples
Tini = 50           # length of past data
N = _env_int('KMPC_NFUT', 15)              # length of predicted horizon
kappa = T-Tini-N+1  # number of columns of Hankel matrix
if kappa <= 0:
    raise ValueError(f"DEEPC_DATA_T={T} must be larger than Tini+N-1={Tini+N-1}")

# constant parameter
s_limit = [_env_float('KMPC_SPA_MIN', 10), _env_float('KMPC_SPA_MAX', 80)]
u_limit =  [_env_float('KMPC_ACC_MIN', -6), _env_float('KMPC_ACC_MAX', 4)]
max_iteration = _env_int('DEEPC_MAX_ITERATION', 1000)

# parameters
if scenario == 'tracking' or scenario == 'braking' or scenario == 'cutin':
    lambda_gi       = 100   # 100
    lambda_yi       = 1e5   # penalty on ||sigma_y||_2^2 in objective 1e5
elif scenario == 'stabilization':
    lambda_gi       = 200
    lambda_yi       = 2*1e5   # penalty on ||sigma_y||_2^2 in objective 1e4
rho             = 1     # penality parameter in ADMM
error_absolute = 0.1  
if scenario == 'braking':
    error_relative = 0.01
else:
    error_relative = 0.001

weight_s = _env_float('KMPC_DEC_Q_SPA', 5)      # 2.5
weight_v = _env_float('KMPC_DEC_Q_VEL', 1.5)    # 1.5
weight_u = _env_float('KMPC_DEC_R', 20.0)       # 20
if scenario == 'stabilization':
    weight_s = _env_float('DEEPC_STAB_Q_SPA', 1.0)
    weight_v = _env_float('DEEPC_STAB_Q_VEL', 2.0)
    weight_u = _env_float('DEEPC_STAB_R', 12.5)

data_file = os.environ.get('DEEPC_DATA_DIR', f'precollected_data_{scenario}_{layout_slug}_T{T}_N{N}')

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

    # Initial Data's Hankel Matrix
    Hzi_vert = [[] for _ in range(n_cav)]
    KKT_vert_matrices = [[] for _ in range(n_cav)]
    Qi = [[] for _ in range(n_cav)]
    Ri = [[] for _ in range(n_cav)]
    Qi_stack = [[] for _ in range(n_cav)]
    Ri_stack = [[] for _ in range(n_cav)]
    Pi = [[] for _ in range(n_cav)]
    Ki = [[] for _ in range(n_cav)]
    args_list = [[] for _ in range(n_cav)]

    if _env_flag('DEEPC_REGENERATE_DATA', False) or not precollected_data_exists():
        print(f"[DeePC] Pre-collected data is missing or regeneration was requested; generating into '{data_file}'")
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

        # Create alternating pattern [weight_s, weight_v] repeated n_vehicle_sub[i] times
        pattern = np.array([weight_s, weight_v])
        weights = np.tile(pattern, n_vehicle_sub[i])
        Qi[i] = np.diagflat(weights)
        Qi_stack[i] = np.kron(np.eye(N),Qi[i])
        Ri[i] = weight_u
        Ri_stack[i] = np.kron(np.eye(N),Ri[i])
        Pi[i] = np.kron(np.eye(N),np.append([1],np.zeros(2*n_vehicle_sub[i]-1)))
        Ki[i] = np.kron(np.eye(N),np.append(np.zeros(2*n_vehicle_sub[i]-1),[1]))

        if i == 0:
            Hgi = Yif.T @ Qi_stack[i] @ Yif + Uif.T @ Ri_stack[i] @ Uif + (lambda_gi)*np.eye(kappa) + (lambda_yi)*(Yip.T @ Yip) + (rho/2)*(np.eye(kappa) + Yif.T @ Pi[i].T @ Pi[i] @ Yif + Uif.T @ Uif)
            Agi = np.vstack((Uip,Eip,Eif))
        else:
            Hgi = Yif.T @ Qi_stack[i] @ Yif + Uif.T @ Ri_stack[i] @ Uif + (lambda_gi)*np.eye(kappa) + (lambda_yi)*(Yip.T @ Yip) + (rho/2)*(np.eye(kappa) + Eif.T @ Eif + Yif.T @ Pi[i].T @ Pi[i] @ Yif + Uif.T @ Uif)
            Agi = np.vstack((Uip,Eip))
        KKT_matrix = np.block([
            [Hgi, Agi.T],
            [Agi, np.zeros((Agi.shape[0], Agi.shape[0]))]
        ])
        KKT_vert_matrices[i] = np.linalg.inv(KKT_matrix)

        if i != n_cav-1:
            Hzi_vert[i] = np.linalg.inv((rho/2)*(np.eye(kappa) + Yif.T @ Ki[i].T @ Ki[i] @ Yif))
        else:
            Hzi_vert[i] = np.linalg.inv((rho/2)*(np.eye(kappa)))

        # Initial dual variables
        g_initial = np.zeros(T-Tini-N+1)
        mu_initial = np.zeros(T-Tini-N+1)
        eta_initial = np.zeros(N)
        phi_initial = np.zeros(N)
        theta_initial = np.zeros(N)

        # init data
        ui_ini = uini[i,:]
        if i == 0:
            ei_ini = eini
        else:
            ei_ini = S_total[0:Tini,pos_cav[i],1]
        yi_ini = np.zeros([2*n_vehicle_sub[i],Tini])
        if i != n_cav-1:
            yi_ini = yini[2*pos_cav[i]:2*pos_cav[i+1],:]
        else:
            yi_ini = yini[2*pos_cav[i]:,:]
            
        # S
        S = np.zeros([Tini+total_time_steps,n_vehicle_sub[i]+1,3])
        if i == 0:
            S = S_total[:,0:n_vehicle_sub[i]+1,:]
        else:
            S = S_total[:,pos_cav[i]:pos_cav[i]+n_vehicle_sub[i]+1,:]

        args_list[i] = [S,ui_ini,ei_ini,yi_ini,Uip,Uif,Eip,Eif,Yip,Yif,g_initial,mu_initial,eta_initial,phi_initial,theta_initial,Qi_stack[i],Ri_stack[i],KKT_vert_matrices[i],Hzi_vert[i],Pi[i],Ki[i]]

    return args_list

def data_collection():
    # Full-state measurement, including speed and position errors of all vehicles.
    m_ctr = n_cav          # number of input variables u(t)
    p_ctr = 2*n_vehicle    # number of output variables

    # There is one head vehicle at the very beginning
    # S: all times all cars(including the head), distance error/ velocity error/ accelaration
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

    # Data collection
    # persistently exciting input data
    # ud = 0.5*(-1+2*np.random.random([m_ctr,T])) # np.random.random generates floats in [0.0, 1.0), so ud is in [-1.0, 1.0)
    ud = np.zeros([m_ctr,T])

    # ================ Replace ed with an arbitrary multi-step function; each speed value is within v_init +/- 10 m/s and lasts randomly for 20-50 steps.
    # ed = np.zeros(T)
    # k_idx = 0
    # while k_idx < T:
    #     # Randomly generate the duration of the current step, 20-50 samples.
    #     duration = np.random.randint(20, 51)
    #     # Randomly generate the speed value of the current step within v_init +/- 10 m/s.
    #     step_velocity = v_init + np.random.uniform(-10, 10)
    #     # Fill this time interval.
    #     end_idx = min(k_idx + duration, T)
    #     ed[k_idx:end_idx] = step_velocity
    #     k_idx = end_idx
    
    # =============== Replace ed with a sine function: period 200 steps, amplitude 10 m/s, center value v_init.
    # time_array = np.arange(T)
    # ed = v_init + 10 * np.sin(2 * np.pi * time_array / 200)

    # =============== ed is piecewise constant: 50/3.6, then 70/3.6, then 100/3.6, then 70/3.6, and finally 50/3.6.
    # ed = np.zeros(T)
    # seg_len = T // 5
    # # Add linear ramp transitions between speed steps: 50 -> 70 -> 100 -> 70 -> 50, in m/s.
    # v_levels = np.array([50, 70, 100, 70, 50]) / 3.6
    # # Use part of each segment as a ramp and the rest as a plateau.
    # ramp_len = max(1, seg_len // 5)   # Ramp length in steps, adjustable as needed.
    # plateau_len = seg_len - ramp_len
    # idx = 0
    # # First plateau segment.
    # ed[idx:idx + plateau_len] = v_levels[0]
    # idx += plateau_len
    # for i in range(1, len(v_levels)):
    #     v_start = v_levels[i - 1]
    #     v_end = v_levels[i]
    #     # Ramp segment: linearly transition from the previous segment speed to the current one.
    #     ed[idx:idx + ramp_len] = np.linspace(v_start, v_end, ramp_len, endpoint=False)
    #     idx += ramp_len
    #     if i < len(v_levels) - 1:
    #         # Middle plateau segment.
    #         ed[idx:idx + plateau_len] = v_end
    #         idx += plateau_len
    #     else:
    #         # Last segment: set all remaining time to the final speed.
    #         ed[idx:] = v_end

    # =============== ed is piecewise constant: 50/3.6, then 70/3.6, then 100/3.6, then 70/3.6, and finally 50/3.6.
    # ed = np.zeros(T)
    # seg_len = T // 5
    # # Add linear ramp transitions between speed steps: 50 -> 70 -> 100 -> 70 -> 50, in m/s.
    # v_levels = np.array([20, 8, 20, 8, 20])
    # # Use part of each segment as a ramp and the rest as a plateau.
    # ramp_len = max(1, seg_len // 3)   # Ramp length in steps, adjustable as needed.
    # plateau_len = seg_len - ramp_len
    # idx = 0
    # # First plateau segment.
    # ed[idx:idx + plateau_len] = v_levels[0]
    # idx += plateau_len
    # for i in range(1, len(v_levels)):
    #     v_start = v_levels[i - 1]
    #     v_end = v_levels[i]
    #     # Ramp segment: linearly transition from the previous segment speed to the current one.
    #     ed[idx:idx + ramp_len] = np.linspace(v_start, v_end, ramp_len, endpoint=False)
    #     idx += ramp_len
    #     if i < len(v_levels) - 1:
    #         # Middle plateau segment.
    #         ed[idx:idx + plateau_len] = v_end
    #         idx += plateau_len
    #     else:
    #         # Last segment: set all remaining time to the final speed.
    #         ed[idx:] = v_end
    # ed += np.random.uniform(-1, 1, size=T)  # Add random perturbations to the step function for richer data.

    # =============== ed can be random perturbation; the following line can be used instead:
    ed = v_init + 5 * np.random.uniform(-1, 1, size=T)
    
    yd = np.zeros([p_ctr,T])
    S[:,0,1] = ed

    # generate output data
    for k in range(T-1):
        acel = IDM_dynamics(S[k,:,:], Tgap, v0_IDM, veh_len, a_IDM, b_IDM, s0_IDM) + np.random.uniform(-0.2, 0.2, size=n_vehicle) # add some noise to the acceleration
        S[k,0,2] = 0                # the head vehicle has ignored acc
        # acel[pos_cav] = ud[:,k]     # CAV control input is the randomly assigned ud signal.
        ud[:,k] = acel[pos_cav]      # CAV control input is assigned from the acceleration computed by IDM dynamics.
        S[k,1:,2] = acel            

        S[k+1,1:,1] = S[k,1:,1] + Tstep*S[k,1:,2]   # update velocity
        S[k+1,:,0] = S[k,:,0] + Tstep*S[k,:,1]      # update position
        # The output yd is an array of [spacing1, velocity1, spacing2, velocity2, ...].
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

    # data Hankel matrices
    # for distributed DeePLCC
    Ui, Ei, Yi = [], [], []

    # Build the save path.
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
    
    Args:
        S: numpy array with shape (total_time_steps, num_vehicles+1, 3)
            Last dimension: [position, velocity, acceleration]
        filename_prefix: prefix for output CSV files
    """
    num_vehicles = S.shape[1] - 1
    
    # Create directory if it doesn't exist
    os.makedirs(filename_prefix, exist_ok=True)
    
    for veh_idx in range(0, num_vehicles + 1):
        # Extract data for this vehicle (skip PV at index 0)
        vehicle_data = S[:, veh_idx, :]  # Shape: (total_time_steps, 3)
        
        # Create DataFrame with column headers
        df = pd.DataFrame(vehicle_data, columns=['position', 'velocity', 'acceleration'])
        
        # Save to CSV
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
