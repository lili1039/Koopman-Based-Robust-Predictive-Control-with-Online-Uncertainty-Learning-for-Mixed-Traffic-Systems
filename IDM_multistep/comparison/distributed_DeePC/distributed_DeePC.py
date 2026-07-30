import numpy as np
import pandas as pd
import math
import time
import os
from functions import (Tstep, ID, pos_cav, n_vehicle, n_cav, n_hdv, T, Tini, N, kappa, rho, lambda_gi, lambda_yi, error_absolute, error_relative,
                        Hankel_matrix, IDM_dynamics, Tgap_list, v0_IDM_list, Tgap_rate_list, v0_rate_list, veh_len_list, a_IDM_list, b_IDM_list, s0_IDM_list,
                        s_limit, u_limit, max_iteration, scenario, subplatoon_spec, vehicle_csv_path, data_file, time_varying, weight_s, weight_v, weight_u,
                        init_sub, save_vehicle_data_to_csv, plot, compute_tracking_metrics, save_metrics, _env_float)

# =========================================================================
#                   Distributed DeeP-LCC Formulation
#
# Uip --> Eif:                  data hankel matrices
# ui_ini --> ei_ini:            past trajectory before time t
# Lambda_gi & lambda_yi:        penalty for the regularization in the cost function
# u_limit & s_limit:            box constraint for control and spacing
# rho:                          penality value for ADMM (Augmented Lagrangian)
# mu_initial --> theta_initial: initial value for variables in ADMM
# KKT_vert:                     pre-calculated inverse matrix for the KKT system
# Hz_vert:                      pre-calculated value of Hz^-1
# ====``=====================================================================

def RunSimulation(Tini,N,kappa,n_cav,uini_list,eini_list,yini_list,yi_ref,Uip,Uif,Eip,Eif,Yip,Yif,g_initial,mu_initial,eta_initial,phi_initial,theta_initial,Qi_stack,Ri_stack,KKT_vert_matrices,Hzi_vert,Pi,Ki,ep_future_0):
    # problem size
    m = 1                   # the size of control input of each subsystem

    # init data
    ui_ini = [uini_list[i].reshape(m*Tini,order='F') for i in range(n_cav)]
    yi_ini = [yini_list[i].reshape(yini_list[i].shape[0]*Tini,order='F') for i in range(n_cav)]
    ei_ini = [eini_list[i].reshape(Tini,order='F') for i in range(n_cav)]
    beqg = [[] for _ in range(n_cav)]
    for i in range(n_cav):        
        if i == 0:
            beqg[i] = np.hstack((ui_ini[i],ei_ini[i],ep_future_0))  # One-dimensional array.
        else:
            beqg[i] = np.hstack((ui_ini[i],ei_ini[i]))              # One-dimensional array.

    # temp variables
    qgi = [[] for _ in range(n_cav)]
    qzi = [[] for _ in range(n_cav)]

    g = [g_initial[i] for i in range(n_cav)] # shape:(kappa,)
    g_plus = [g_initial[i] for i in range(n_cav)] # shape:(kappa,)
    z = [g_initial[i] for i in range(n_cav)] # shape:(kappa,)
    z_plus = [g_initial[i] for i in range(n_cav)] # shape:(kappa,)
    s = [Pi[i]@Yif[i]@g[i] for i in range(n_cav)]  # shape:(N,)
    s_plus = [Pi[i]@Yif[i]@g[i] for i in range(n_cav)]  # shape:(N,)
    u = [Uif[i]@g[i] for i in range(n_cav)]  # shape:(N,)
    u_plus = [Uif[i]@g[i] for i in range(n_cav)]  # shape:(N,)
    mu = [mu_initial[i] for i in range(n_cav)] # shape:(kappa,)
    mu_plus = [mu_initial[i] for i in range(n_cav)] # shape:(kappa,)
    eta = [eta_initial[i] for i in range(n_cav)] # shape:(N,)
    eta_plus = [eta_initial[i] for i in range(n_cav)] # shape:(N,)
    phi = [phi_initial[i] for i in range(n_cav)] # shape:(N,)
    phi_plus = [phi_initial[i] for i in range(n_cav)] # shape:(N,)
    theta = [theta_initial[i] for i in range(n_cav)] # shape:(N,)
    theta_plus = [theta_initial[i] for i in range(n_cav)] # shape:(N,)

    bar_eta = [[] for _ in range(n_cav)] # shape:(N,)
    bar_epsilon = [[] for _ in range(n_cav)] # shape:(N,)

    for k in range(max_iteration):
        # =============== Step1: update gi ================
        # compute bar_eta & beqg & qgi
        for i in range(n_cav):
            if i != n_cav-1:
                bar_eta[i] = eta[i] - rho*Ki[i]@Yif[i]@z[i]

        for i in range(n_cav):        
            if i == 0:
                qgi[i] = -lambda_yi*Yip[i].T@yi_ini[i] - Yif[i].T@Qi_stack[i]@yi_ref[i] + 1/2*(mu[i] - rho*z[i] - Yif[i].T@Pi[i].T@phi[i] - Uif[i].T@theta[i] - rho*Yif[i].T@Pi[i].T@s[i] - rho*Uif[i].T@u[i])
            else:
                qgi[i] = -lambda_yi*Yip[i].T@yi_ini[i] - Yif[i].T@Qi_stack[i]@yi_ref[i] + 1/2*(mu[i] - rho*z[i] - Yif[i].T@Pi[i].T@phi[i] - Uif[i].T@theta[i] - rho*Yif[i].T@Pi[i].T@s[i] - rho*Uif[i].T@u[i]) + 1/2*Eif[i].T@bar_eta[i-1]
            
            g_plus[i] = (-KKT_vert_matrices[i]@np.hstack((qgi[i],-beqg[i])))[0:kappa]

        # =============== Step2: update z/s/u ================
        for i in range(n_cav):
            if i != 0:
                bar_epsilon[i] = Eif[i]@g_plus[i]

        for i in range(n_cav):
            if i != n_cav-1:
                qzi[i] = -mu[i]/2 - rho/2*g_plus[i] - Yif[i].T@Ki[i].T@(eta[i]/2 + rho/2*bar_epsilon[i+1])
            else:
                qzi[i] = -mu[i]/2 - rho/2*g_plus[i]

            z_plus[i] = -Hzi_vert[i]@qzi[i]
            s_temp = Pi[i]@Yif[i]@g_plus[i] - phi[i]/rho
            s_plus[i] = s_temp.clip(min=s_limit[0],max=s_limit[1])
            u_temp = Uif[i]@g_plus[i] - theta[i]/rho
            u_plus[i] = u_temp.clip(min=u_limit[0],max=u_limit[1])

        # =============== Step3: update dual variables ================
        for i in range(n_cav):
            if i != n_cav-1:
                eta_plus[i] = eta[i] + rho*(bar_epsilon[i+1]-Ki[i]@Yif[i]@z_plus[i])
            mu_plus[i] = mu[i] + rho*(g_plus[i]-z_plus[i])
            phi_plus[i] = phi[i] + rho*(s_plus[i] - Pi[i]@Yif[i]@g_plus[i])
            theta_plus[i] = theta[i] + rho*(u_plus[i] - Uif[i]@g_plus[i])

        # =============== Check iteration stopping condition ================
        error_pri1, error_pri2, error_pri3, error_pri4 = 0,0,0,0
        error_dual1, error_dual2, error_dual3, error_dual4 = 0,0,0,0
        tolerance_pri1, tolerance_pri2, tolerance_pri3, tolerance_pri4 = 0,0,0,0
        tolerance_dual1, tolerance_dual2, tolerance_dual3, tolerance_dual4 = 0,0,0,0
        for i in range(n_cav):        
            error_pri1 += np.linalg.norm(g_plus[i]-z_plus[i])
            error_dual1 += rho*np.linalg.norm(z_plus[i]-z[i])

            if i != n_cav-1:
                error_pri2 += np.linalg.norm(bar_epsilon[i+1]-Ki[i]@Yif[i]@z_plus[i])
                error_dual2 += rho*np.linalg.norm(Eif[i+1].T@Ki[i]@Yif[i]@(z_plus[i]-z[i]))

            error_pri3 += np.linalg.norm(s_plus[i] - Pi[i]@Yif[i]@g_plus[i])
            error_dual3 += rho*np.linalg.norm(Yif[i].T@Pi[i].T@(s_plus[i]-s[i]))
            
            error_pri4 +=  np.linalg.norm(u_plus[i] - Uif[i]@g_plus[i])
            error_dual4 += rho*np.linalg.norm(Uif[i].T@(u_plus[i]-u[i]))
            
            tolerance_pri1 += math.sqrt(kappa)*error_absolute + \
                error_relative*max(np.linalg.norm(g_plus[i]),np.linalg.norm(z_plus[i]))
            tolerance_dual1 += math.sqrt(kappa)*error_absolute + error_relative*np.linalg.norm(mu_plus[i])

            if i != n_cav-1:
                tolerance_pri2 += math.sqrt(N)*error_absolute + \
                    error_relative*max(np.linalg.norm(bar_epsilon[i+1]),np.linalg.norm(Ki[i]@Yif[i]@z_plus[i]))
                tolerance_dual2 += math.sqrt(N)*error_absolute + error_relative*np.linalg.norm(Eif[i+1].T@eta_plus[i])  

            tolerance_pri3 += math.sqrt(N)*error_absolute + error_relative*max(np.linalg.norm(s_plus[i]),np.linalg.norm(Pi[i]@Yif[i]@g_plus[i]))
            tolerance_dual3 += math.sqrt(kappa)*error_absolute + error_relative*np.linalg.norm(Yif[i].T@Pi[i].T@phi_plus[i])    

            tolerance_pri4 += math.sqrt(N)*error_absolute + error_relative*max(np.linalg.norm(Uif[i]@g_plus[i]),np.linalg.norm(u_plus[i]))
            tolerance_dual4 += math.sqrt(kappa)*error_absolute + error_relative*np.linalg.norm(Uif[i].T@theta_plus[i])
            
        for i in range(n_cav):
            if i != n_cav-1:
                eta[i] = eta_plus[i]

            g[i] = g_plus[i]
            z[i] = z_plus[i]
            u[i] = u_plus[i]
            s[i] = s_plus[i]
            mu[i] = mu_plus[i]
            phi[i] = phi_plus[i]
            theta[i] = theta_plus[i]

        if error_pri1 <= tolerance_pri1 and error_dual1 <= tolerance_dual1 and \
        error_pri2 <= tolerance_pri2 and error_dual2 <= tolerance_dual2 and \
        error_pri3 <= tolerance_pri3 and error_dual3 <= tolerance_dual3 and \
        error_pri4 <= tolerance_pri4 and error_dual4 <= tolerance_dual4:
            break
        

    # Record optimal value
    real_iter_num = k+1
    g_opt = g
    mu_opt = mu
    eta_opt = eta
    phi_opt = phi
    theta_opt = theta
    u_opt = u
    y_opt = [Yif[i]@g_opt[i] for i in range(n_cav)]

    return real_iter_num, g_opt, mu_opt, eta_opt, phi_opt, theta_opt, u_opt, y_opt

if __name__ == "__main__":
    # Initialization.
    args = init_sub(scenario)
    SDES = _env_float('KMPC_SDES', 40.0)
    result_root = os.environ.get('KMPC_RESULT_ROOT', 'result')
    run_suffix = os.environ.get('KMPC_RUN_SUFFIX', '')
    if not run_suffix.strip():
        run_suffix = '_' + time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(result_root, scenario, 'Distributed_DeePC' + run_suffix)
    vehicle_data_dir = os.path.join(output_dir, 'vehicle_data')
    os.makedirs(output_dir, exist_ok=True)

    run_settings = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'scenario': scenario,
        'control_method': 'Distributed DeePC',
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
        'deepc_weights': {
            'weight_s': weight_s,
            'weight_v': weight_v,
            'weight_u': weight_u,
            'lambda_gi': lambda_gi,
            'lambda_yi': lambda_yi,
            'rho': rho,
        },
        'max_iteration': max_iteration,
        'result_root': result_root,
        'run_suffix': run_suffix,
        'output_dir': output_dir,
    }
    settings_path = os.path.join(output_dir, 'run_settings.txt')
    with open(settings_path, 'w', encoding='utf-8') as f:
        for key, value in run_settings.items():
            f.write(f"{key}: {value}\n")
    print(f"Run settings saved to '{settings_path}'")

    # Save data and parameters for all subsystems as lists.
    S_list,uini_list,eini_list,yini_list,Uip_list,Uif_list,Eip_list,Eif_list,Yip_list,Yif_list,g_initial_list,mu_initial_list,eta_initial_list,phi_initial_list,theta_initial_list,Qi_stack_list,Ri_stack_list,KKT_vert_matrices_list,Hzi_vert_list,Pi_list,Ki_list = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    for i in range(n_cav):
        S, uini, eini, yini, Uip, Uif, Eip, Eif, Yip, Yif, g_initial,mu_initial,eta_initial,phi_initial,theta_initial, Qi_stack,Ri_stack,KKT_vert_matrices,Hzi_vert,Pi,Ki = args[i]
        S_list.append(S)
        uini_list.append(uini)
        eini_list.append(eini)
        yini_list.append(yini)
        Uip_list.append(Uip)
        Uif_list.append(Uif)
        Eip_list.append(Eip)
        Eif_list.append(Eif)
        Yip_list.append(Yip)
        Yif_list.append(Yif)
        g_initial_list.append(g_initial)
        mu_initial_list.append(mu_initial)
        eta_initial_list.append(eta_initial)    
        phi_initial_list.append(phi_initial)
        theta_initial_list.append(theta_initial)
        Qi_stack_list.append(Qi_stack)
        Ri_stack_list.append(Ri_stack)
        KKT_vert_matrices_list.append(KKT_vert_matrices)
        Hzi_vert_list.append(Hzi_vert)
        Pi_list.append(Pi)
        Ki_list.append(Ki)

    # operation data for analyse
    total_time_steps = S_list[0].shape[0]
    control_steps = total_time_steps - Tini - 1
    computation_time = np.zeros(control_steps)
    iteration_num    = np.zeros(control_steps)

    yref_list = [[] for _ in range(n_cav)]
    y_opt_list = [np.zeros(yini_list[i].shape[0]*N) for i in range(n_cav)]  # Previous optimal output trajectory.

    Tgap_this_step_list = [[] for i in range(n_cav)]
    v0_IDM_this_step_list = [[] for i in range(n_cav)]
    cutin_time = 8.0
    cutin_drop = 25.0
    cutin_step = int(round(cutin_time / Tstep))
    # simulate
    for k in range(Tini, total_time_steps-1):
        step_idx = k - Tini
        for i in range(n_cav):
            Tgap_this_step_list[i] = Tgap_list[i] * (1 + Tgap_rate_list[i]/100)**(k+Tini)
            v0_IDM_this_step_list[i] = v0_IDM_list[i] * (1 + v0_rate_list[i]/100)**(k+Tini)
            # print(f"Step {i}: Tgap_this_step: {Tgap_this_step}, v0_IDM_this_step: {v0_IDM_this_step}")

        # Construct reference output.
        if scenario == 'stabilization':
            vdes = 25
            sdes = SDES
            for i in range(n_cav):
                p = yini_list[i].shape[0]
                yref = np.zeros((p,N))
                yref[0::2,:] = sdes
                yref[1::2,:] = vdes
                yref_list[i] = yref.reshape(p*N, order='F')
        elif scenario == 'tracking' or scenario == 'braking' or scenario == 'cutin':
            sdes = SDES
            # Reference Koopman MPC: compute reference speed separately for each subsystem.
            # v_des_i[j] = 1/6*(PV speed[j] + CAV i-1 speed[j] + 2*CAV i current speed[j] + 2*CAV i direct-leader speed[j]).
            # Extract speed sequence from the previous optimized trajectory: start from the second value and append a repeated final value.
            for i in range(n_cav):
                p = yini_list[i].shape[0]

                # PV speed sequence for the future N steps.
                v_pv = np.tile(S_list[0][k, 0, 1], N)  # PV speed for the future N steps.

                # Previous CAV speed sequence extracted from the previous optimized output.
                if i == 0:
                    v_cav_prev = np.tile(S_list[0][k, 0, 1], N)   # PV speed.
                else:
                    # Extract the speed sequence from the previous subsystem optimized output at odd indices.
                    y_prev_reshaped = y_opt_list[i-1].reshape(yini_list[i-1].shape[0], N, order='F')
                    # Assuming output structure [spacing, speed, spacing, speed, ...], take the first CAV speed at index 1.
                    v_cav_prev_opt = y_prev_reshaped[1, :]  # Speed sequence optimized at the previous step.
                    v_cav_prev = np.append(v_cav_prev_opt[1:], v_cav_prev_opt[-1])  # Start from the second value and append one final value.

                # Current CAV speed sequence extracted from the previous optimized output.
                y_curr_reshaped = y_opt_list[i].reshape(yini_list[i].shape[0], N, order='F')
                v_cav_as_opt = y_curr_reshaped[1, :]  # Current CAV speed sequence.
                v_cav_as = np.append(v_cav_as_opt[1:], v_cav_as_opt[-1])  # Start from the second value and append one final value.

                # Leader speed sequence from the last vehicle of the previous subsystem.
                if i == 0:
                    v_ref = np.tile(S_list[0][k, 0, 1], N)  # PV speed.
                else:
                    # Speed of the last vehicle in the previous subsystem.
                    y_prev_reshaped = y_opt_list[i-1].reshape(yini_list[i-1].shape[0], N, order='F')
                    v_ref_opt = y_prev_reshaped[-1, :]  # Speed of the last vehicle in the previous subsystem, i.e., the last speed row.
                    v_ref = np.append(v_ref_opt[1:], v_ref_opt[-1])  # Start from the second value and append one final value.

                # Compute reference speed sequence.
                vdes = (v_pv + v_cav_prev + 2*v_cav_as + 2*v_ref) / 6

                yref = np.zeros((p,N))
                yref[0::2,:] = sdes
                yref[1::2,:] = vdes
                yref_list[i] = yref.reshape(p*N, order='F')


        ep_future_0 = np.tile(S_list[0][k, 0, 1], N)  # PV speed for the future N steps, unknown to the CAV.
        start_time = time.time()
        real_iter_num, g_opt, mu_opt, eta_opt, phi_opt, theta_opt, u_opt, y_opt = RunSimulation(Tini,N,kappa,n_cav,uini_list,eini_list,yini_list,yref_list,Uip_list,Uif_list,Eip_list,Eif_list,Yip_list,Yif_list,g_initial_list,mu_initial_list,eta_initial_list,phi_initial_list,theta_initial_list,Qi_stack_list,Ri_stack_list,KKT_vert_matrices_list,Hzi_vert_list,Pi_list,Ki_list,ep_future_0)
        computation_time[step_idx] = (time.time() - start_time)/n_cav
        iteration_num[step_idx] = real_iter_num
        print(f'Time step {step_idx}/{control_steps}, Computation Time per Subsystem: {computation_time[step_idx]:.4f}s, Iteration Number: {real_iter_num}')

        # Save the current optimized output trajectory for the next step.
        for i in range(n_cav):
            y_opt_list[i] = y_opt[i]

        for i in range(n_cav):
            # apply control input and update system state for each subsystem
            acel = IDM_dynamics(
                S_list[i][k,:,:],
                Tgap_this_step_list[i],
                v0_IDM_this_step_list[i],
                veh_len_list[i],
                a_IDM_list[i],
                b_IDM_list[i],
                s0_IDM_list[i],
            )
            S_list[i][k,1:,2] = acel            # all the vehicles using HDV model 
            # apply control input
            S_list[i][k,1,2] = u_opt[i][0] 

            # update velocity
            S_list[i][k+1,1:,1] = S_list[i][k,1:,1] + Tstep*S_list[i][k,1:,2]

            # Update S[~,0,1]; do not update S[~,0,2].
            if i !=0 :
                S_list[i][k+1,0,1] = S_list[i-1][k+1,-1,1]
                    
            # update position
            S_list[i][k+1,:,0] = S_list[i][k,:,0] + Tstep*S_list[i][k,:,1]
            if i != 0:
                S_list[i][k+1,0,0] = S_list[i-1][k+1,-1,0]
            
            # record output
            y = np.ravel(np.column_stack((S_list[i][k+1, :-1, 0] - S_list[i][k+1, 1:, 0], S_list[i][k+1, 1:, 1])))
            if i == 0:
                e = S_list[0][k+1, 0, 1]   # PV speed.
            else:
                e = S_list[i-1][k+1,-1,1]  # Speed of the last vehicle in the previous subsystem.
            u = u_opt[i][0] 

            # update yini & eini & uini
            eini_list[i] = np.append(eini_list[i][1:],e)
            uini_list[i] = np.append(uini_list[i][1:],u)
            yini_list[i] = np.hstack((yini_list[i][:,1:],y.reshape(-1,1)))

            # update initial dual variables for next time step
            g_initial_list[i] = g_opt[i]
            mu_initial_list[i] = mu_opt[i]
            eta_initial_list[i] = eta_opt[i]
            phi_initial_list[i] = phi_opt[i]
            theta_initial_list[i] = theta_opt[i]

        if scenario == 'cutin' and (step_idx + 1) == cutin_step:
            S_list[0][k+1, 0, 0] -= cutin_drop
            print(f"[cut-in] Step {step_idx+1}: leader position shifted backward by {cutin_drop:.1f} m; CAV1-PV spacing dropped sharply")
    
    S_total = np.zeros((total_time_steps,n_vehicle+1,3))
    for i in range(n_cav):
        if i == 0:
            S_total[:,0:pos_cav[i+1]+1,:] = S_list[i]
        elif i != n_cav-1:
            S_total[:,pos_cav[i]+1:pos_cav[i+1]+1,:] = S_list[i][:,1:,:]
        else:
            S_total[:,pos_cav[i]+1:,:] = S_list[i][:,1:,:]

    # save operation data
    print(f"Result output directory: {output_dir}")
    np.savetxt(os.path.join(output_dir, 'control_computation_time.csv'), computation_time, delimiter=',')
    np.savetxt(os.path.join(output_dir, 'iteration_num.csv'), iteration_num, delimiter=',')
    save_vehicle_data_to_csv(S_total, vehicle_data_dir)
    plot(scenario, S_total, Tstep, n_cav, output_dir, subplatoon_spec)

    metrics = compute_tracking_metrics(S_total[Tini:], scenario, sdes=SDES)
    metrics.update({
        'AvgControlTime_s': float(np.mean(computation_time)),
        'AvgIteration': float(np.mean(iteration_num)),
        'weight_s': weight_s,
        'weight_v': weight_v,
        'weight_u': weight_u,
        'lambda_gi': lambda_gi,
        'lambda_yi': lambda_yi,
        'rho': rho,
    })
    print("==================== Evaluation Metrics ====================")
    print(f"  RMSVE = {metrics['RMSVE']:.4f} m/s   RMSSE = {metrics['RMSSE']:.4f} m")
    print(f"  Avg control solve time = {metrics['AvgControlTime_s']*1000:.3f} ms | Avg iteration = {metrics['AvgIteration']:.2f}")
    print("=================================================")
    save_metrics(metrics, filename=os.path.join(output_dir, 'metrics.csv'))

        

