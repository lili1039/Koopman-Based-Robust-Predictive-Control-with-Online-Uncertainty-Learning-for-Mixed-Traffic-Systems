@echo off
REM ============================================================================
REM  Koopman multi-step predictor MPC simulation - one-click run script
REM  Usage: edit the settings below, save, then run this .bat.
REM  All resolved settings are written to <result_dir>/run_settings.txt.
REM ============================================================================
setlocal
REM Prevent settings from an earlier batch experiment leaking into this single run.
for /f "tokens=1 delims==" %%V in ('set KMPC_ 2^>nul') do set "%%V="

REM Use python from the activated environment.
set "PYTHON=python"
"%PYTHON%" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo Python could not be started. Activate your project environment.
    exit /b 1
)

REM ====================== core ======================
REM scenario: tracking / braking
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

REM ====================== MPC weights (tracking / braking) ======================
REM Decentralized MPC: Q=diag([spacing, velocity] repeated), Qa likewise.
set "KMPC_DEC_Q_SPA=2.5"
set "KMPC_DEC_Q_VEL=1.5"
set "KMPC_DEC_QA_SPA=0.0"
set "KMPC_DEC_QA_VEL=0.0"
set "KMPC_DEC_R=20.0"
set "KMPC_DEC_RD=30.0"

REM Distributed MPC: Q1 tracks PV, Q2 tracks preceding CAV, Qa/Q3 tracks assumed own last-step traj, Q4 tracks local leader.
REM Terminal target uses the same velocity weights as these references.
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
set "KMPC_RESULT_ROOT=result"
REM Empty suffix uses the method name; existing names receive _2, _3, etc.
set "KMPC_RUN_SUFFIX="
REM vehicle parameter csv
set "KMPC_VEHICLE_CSV=vehicle_parameters.csv"

REM ====================== optional ======================
REM --- signal delay (tracking + Distributed MPC) ---
set "KMPC_SIGNAL_DELAY=0"
set "KMPC_DELAY_SEED=202606"
set "KMPC_DELAY_COUNT=10"
set "KMPC_DELAY_MIN_DURATION=3"
set "KMPC_DELAY_MAX_DURATION=8"
REM Empty => randomly choose from all HDVs; optional candidates use subplatoon:HDV, e.g. 2:1,4:2.
set "KMPC_DELAY_HDVS="

REM --- random CAV optimal-prediction packet loss (Distributed MPC) ---
REM Receiver subplatoon k loses the optimal prediction sent by CAV(k-1).
REM Empty receivers => all downstream subplatoons 2..N are candidates.
set "KMPC_CAV_PRED_LOSS=0"
set "KMPC_CAV_PRED_LOSS_SEED=202607"
set "KMPC_CAV_PRED_LOSS_COUNT=10"
set "KMPC_CAV_PRED_LOSS_MIN_DURATION=3"
set "KMPC_CAV_PRED_LOSS_MAX_DURATION=8"
set "KMPC_CAV_PRED_LOSS_RECEIVERS=2,3,4,5"

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
