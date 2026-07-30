# ========== Standard imports ==========
import argparse
import numpy as np
import pandas as pd
import time
import os, sys
import pickle as pkl
import re
from matplotlib import pyplot as plt
import cvxpy as cp

# Project directory.
np.set_printoptions(threshold=sys.maxsize)
cur_path = os.getcwd()
project_path = os.path.dirname(os.path.dirname(cur_path)) # Directory of the full project.

Tstep = 0.12
Npst = 50


def IDM_dynamics(S, Tgap, v0, veh_len, a_idm, b_idm, s0_idm):
    '''
    Inputs
    S: all-vehicle state data at time k, with state entries for position s and velocity v.
    Tgap: car-following parameter vector for all following vehicles.
    veh_len: vehicle lengths for all following vehicles.
    a_idm/b_idm/s0_idm: heterogeneous IDM parameters for each vehicle.

    Output
    acel: acceleration vector.
    '''

    # limitation of actuators
    acel_max = 4
    dcel_max = -6

    V_diff = S[0:-1,1] - S[1:,1] # the velocity error with former car
    D_diff = S[0:-1,0] - S[1:,0] - veh_len # the pos error with former car
    # D_diff = S[0:-1,0] - S[1:,0]

    s_star = s0_idm + np.maximum(0, Tgap*S[1:,1] - (S[1:,1]*V_diff)/(2*np.sqrt(a_idm*b_idm)))
    acel = a_idm*(1-(S[1:,1]/v0)**4 - (s_star/D_diff)**2)
    
    # acceleration saturation
    acel = np.where(acel > acel_max, acel_max, acel)
    acel = np.where(acel < dcel_max, dcel_max, acel)
    
    return acel

def compute_IDM_steady_s(v_PV, v, v0, Tgap, a_idm, b_idm, s0_idm):
    s_star = s0_idm + v*Tgap - (v*(v_PV-v))/(2*np.sqrt(a_idm*b_idm))
    ratio = np.minimum(v / v0, 0.9999)  # Clamp to prevent division by zero
    s = s_star / np.sqrt(1 - ratio**4)
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

    # plt.figure(figsize=(10,4))
    # plt.plot(time, vel)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Velocity")
    # plt.title("Velocity Profile")
    # plt.grid(True)
    # plt.show()

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

def generate_history_seq(sim_subplatoon_num,sim_num_veh,vPV_init,v_init,s_init,Tgap,v0_IDM,Tgap_rate,v0_rate,veh_len,Tstep,a_idm,b_idm,s0_idm):
    """
    Generate historical state sequences with IDM for any number of controlled vehicles.

    Returns:
    S shape=(Npst+1, 1+sim_num_veh, 3)  pos/vel/acc
    """

    S = np.zeros([Npst+1,sim_num_veh+1,3]) #pos/vel/acc

    S[:,0,1] = vPV_init
    for i in range(sim_num_veh):
        S[0,i+1,0] = S[0,i,0] - s_init[i]
        S[0,i+1,1] = v_init[i] 

    # Time-step simulation.
    for i in range(Npst):
        Tgap_this_step = Tgap * (1 + Tgap_rate/100)**i
        v0_IDM_this_step = v0_IDM * (1 + v0_rate/100)**i
        # print(f"Step {i}: Tgap_this_step: {Tgap_this_step}, v0_IDM_this_step: {v0_IDM_this_step}")
        acel = IDM_dynamics(S[i, :, :], Tgap_this_step, v0_IDM_this_step, veh_len, a_idm, b_idm, s0_idm)
        S[i,1:,2] = acel
        # Update speed and position.
        S[i+1,1:,1] = S[i,1:,1] + acel * Tstep
        S[i+1,:,0] = S[i,:,0] + S[i+1,:,1] * Tstep
    
    return S

def plot(scenario, S, Tstep, sim_subplatoon_num, sub_numHDV_list, out_dir='result'):
    # Time sequence.
    os.makedirs(out_dir, exist_ok=True)
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
    
    if len(sub_numHDV_list) != sim_subplatoon_num:
        raise ValueError("sub_numHDV_list length must match sim_subplatoon_num")

    for subplatoon_idx in range(sim_subplatoon_num):
        # Add CAV for this subplatoon
        veh_labels.append(f'CAV_{subplatoon_idx + 1}')
        veh_color.append(my_colors_cav[subplatoon_idx])  # Example color for CAV
        
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
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'plot_{state_name}.png'))
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
    plt.ylim(10, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot_Spacing.png'))
    plt.close()

def predict_future_ep_and_constraints(scenario, Nfut, v_preceed_t):
    ep = np.zeros([1,Nfut+1])
    ep_aver_window = 20     # Past window length for estimating ep min/max.

    if scenario == 'stabilization':
        # Future ep is estimated as the constant equilibrium value 25.
        ep = np.tile(25, (1, Nfut+1))
        
    elif scenario == 'tracking' or scenario == 'braking' or scenario == 'cutin':
        # Future ep is estimated as constant ep_t.
        ep = np.tile(v_preceed_t, (1, Nfut+1))
        
    return ep

def save_vehicle_data_to_csv(S, filename_prefix='result/vehicle_data'):
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
    """
    Match the Koopman baseline: exclude the PV and compute RMSVE/RMSSE over the supplied time window.
    Pass S[Npst:] from the caller to evaluate only the control segment.
    """
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

def save_metrics(metrics, filename='result/metrics.csv'):
    """Save the metrics dictionary as a two-column CSV (metric,value)."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pd.DataFrame(list(metrics.items()), columns=['metric', 'value']).to_csv(filename, index=False)
    print(f"Evaluation metrics saved to '{filename}'")

def save_computation_time(compute_time_collection, type, out_dir='result'):
    """
    Save computation time collection to CSV file.
    
    Args:
        compute_time_collection: numpy array of shape (sim_subplatoon_num, total_time_steps-1)
        filename: output CSV filename
    """
    os.makedirs(out_dir, exist_ok=True)
    filename=os.path.join(out_dir, f'{type}_computation_time.csv')
    np.savetxt(filename, compute_time_collection, delimiter=',')
    print(f"Computation time data saved to '{filename}'")
