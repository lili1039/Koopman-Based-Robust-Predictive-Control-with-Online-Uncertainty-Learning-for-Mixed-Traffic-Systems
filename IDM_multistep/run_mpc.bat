@echo off
REM ============================================================================
REM  Koopman multi-step predictor MPC simulation - one-click run script
REM  Usage: edit the settings below, save, then run this .bat.
REM  All resolved settings are written to <result_dir>/run_settings.txt.
REM ============================================================================
setlocal

REM ---- Python interpreter (python310 conda env) ----
set "PYTHON=C:\Users\13373\.conda\envs\python310\python.exe"

REM ====================== core ======================
REM scenario: tracking / braking / stabilization / cutin
set "KMPC_SCENARIO=tracking"
REM control method: All HDVs / Decentralized MPC / Distributed MPC
set "KMPC_CONTROL=Distributed MPC"
REM build disturbance set and tighten constraints: 1=on, 0=off
set "KMPC_DISTURBANCE_ESTIMATE=1"
REM MPC prediction horizon (must be <= model export steps = 15)
set "KMPC_NFUT=15"
REM platoon composition: comma-separated, each number = HDV count of that
REM subplatoon (from PV backward) and also the HDV model id used (valid: 1-4)
set "KMPC_SUBPLATOON=2,3,1,4,2"

REM ====================== spacing / constraint bounds ======================
REM desired (equilibrium) spacing [m]
set "KMPC_SDES=40"
REM spacing bounds [min,max] [m]
set "KMPC_SPA_MIN=10"
set "KMPC_SPA_MAX=80"
REM velocity bounds [min,max] [m/s]
set "KMPC_VEL_MIN=0"
set "KMPC_VEL_MAX=35"
REM acceleration bounds [min,max] [m/s^2]
set "KMPC_ACC_MIN=-6"
set "KMPC_ACC_MAX=4"

REM ====================== MPC weights (tracking / braking / cutin only) ======================
REM Decentralized MPC: Q=diag([spacing, velocity] repeated), Qa likewise.
REM Tuned recommendation 202607 sweep: keep spa:vel=2.0:0.8, raise R from 20 to 30.
REM Cuts startup shock and accel oscillation vs R=20 while keeping spacing regulation.
REM See 202607_wtune/recommendation.md. Alternatives: R=20 tightest spacing, R=40 smoothest.
set "KMPC_DEC_Q_SPA=2.5"
set "KMPC_DEC_Q_VEL=1.5"
set "KMPC_DEC_QA_SPA=0.0"
set "KMPC_DEC_QA_VEL=0.0"
set "KMPC_DEC_R=20.0"
set "KMPC_DEC_RD=30.0"

REM Distributed MPC: Q1 tracks PV, Q2 tracks preceding CAV, Qa/Q3 tracks assumed own last-step traj, Q4 tracks local leader.
REM Recommended 202607 tuning: velocity ratio PV:CAVi-1:assume:LVi = 0.05:0.05:0.5:1.0, spacing 0.125 each,
REM together with KMPC_DIST_TERM_REFS=weighted, i.e. terminal = velocity-weighted avg of the 4 refs, consistent with cost.
REM PV-weight Q1,Q2 is the tradeoff knob: more PV = stronger braking string-stability but looser NEDC spacing;
REM less PV = tighter NEDC spacing but weaker braking damping. Alternatives: 0.15:0.15:0.2:1.0 for max braking robustness,
REM or 0:0:0.5:1.0 for tightest NEDC spacing. KMPC_DIST_Q_SPA / KMPC_DIST_Q_VEL still override Q1,Q2,Q4 together.
set "KMPC_DIST_Q1_SPA=0.125"
set "KMPC_DIST_Q1_VEL=0.1"
set "KMPC_DIST_Q2_SPA=0.125"
set "KMPC_DIST_Q2_VEL=0.1"
set "KMPC_DIST_QA_SPA=0.125"
set "KMPC_DIST_QA_VEL=0.5"
set "KMPC_DIST_Q4_SPA=0.125"
set "KMPC_DIST_Q4_VEL=1.0"
set "KMPC_DIST_R=20.0"
set "KMPC_DIST_RD=30.0"
REM Distributed terminal velocity target: weighted = velocity-weighted avg of the refs, recommended.
REM Set to pv,preceed,ref,assume or leave unset for the original equal 4-average; a comma subset e.g. ref also works.
set "KMPC_DIST_TERM_REFS=weighted"

REM ====================== online correction / error set ======================
REM alpha exponential smoothing factor (larger = more inertia)
set "KMPC_LAM=0.7"
REM refit sliding-window history point count
set "KMPC_NPOINT=20"
REM past window length to estimate PV acceleration range
set "KMPC_EP_WINDOW=20"
REM online-correction worker threads (1 = serial; ~8 recommended for real-time)
set "KMPC_CORR_THREADS=8"

REM ====================== output / vehicle params ======================
REM result root dir
set "KMPC_RESULT_ROOT=202606_result"
REM run suffix: empty => auto timestamp (to the second). custom e.g. _myrun
set "KMPC_RUN_SUFFIX=_delayed"
REM vehicle parameter csv
set "KMPC_VEHICLE_CSV=vehicle_parameters.csv"

REM ====================== optional ======================
REM --- signal delay (tracking + Distributed MPC) ---
set "KMPC_SIGNAL_DELAY=1"
set "KMPC_DELAY_STEP_WINDOWS=120-124,320-324,520-524,720-724,920-924,1120-1124,1320-1324,1520-1524,1720-1724,1920-1924"
set "KMPC_DELAY_HDVS=2:1,4:2"

REM --- soft-constraint (slack): enable controls optimization; save controls recording only ---
set "KMPC_ENABLE_SLACK=0"
set "KMPC_SAVE_SLACK=0"
REM slack trigger threshold (a per-step slack above this counts as triggered)
set "KMPC_SLACK_TOL=1e-3"

REM ====================== run ======================
cd /d "%~dp0"
"%PYTHON%" Koopman_MPC_IDM_learning_set.py
echo.
echo ==== done, exit code %ERRORLEVEL% ====
pause
endlocal
