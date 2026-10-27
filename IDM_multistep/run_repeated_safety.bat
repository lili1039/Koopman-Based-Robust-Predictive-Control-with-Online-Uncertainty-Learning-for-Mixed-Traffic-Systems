@echo off
setlocal
REM One launch: 15 random IDM profiles x 4 braking intensities x 6 times x 3 methods.
REM Run all 720 controlled hard cases first; rerun only confirmed infeasible cases with slack.
REM All 360 All-HDVs baselines are simulated afresh. No previous results are required.
REM Default: fixed result name; existing names receive _2, _3, etc.
REM --resume FULL_RESULT_PATH continues that batch.
REM --check-config validates settings and model inputs only; --workers N sets 1..8 workers.
REM Activate the Python environment before running this script.
for /f "tokens=1 delims==" %%V in ('set KMPC_ 2^>nul') do set "%%V="
set "PYTHON=python"
"%PYTHON%" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo Python could not be started. Activate your project environment.
    exit /b 1
)
set "KMPC_REPEAT_WORKERS=4"
REM Shared evaluation ends at step 333 = 39.96 s; history is excluded.
set "KMPC_REPEAT_EVALUATION_END_STEP=333"
set "KMPC_REPEAT_PARAMETER_SEEDS=2026102001,2026102002,2026102301,2026102302,2026102303,2026102304,2026102305,2026102306,2026102307,2026102308,2026102309,2026102310,2026102311,2026102312,2026102313"
set "KMPC_REPEAT_BRAKES=3,3.5,4,4.5"
set "KMPC_REPEAT_START_STEP=84"
set "KMPC_REPEAT_INTERVAL_STEPS=10"
set "KMPC_REPEAT_TIMING_COUNT=6"
set "KMPC_REPEAT_CI_SEED=20261023"
set "KMPC_REPEAT_CI_RESAMPLES=10000"
set "KMPC_SIM_SEED=20261020"
set "KMPC_SCENARIO=cutin"
set "KMPC_CUTIN_GAP=25.5"
set "KMPC_CUTIN_FINAL_SPEED=4"
REM Duration is (26-4)/braking_magnitude; 5.5 s for the original -4 m/s^2 case.
set "KMPC_TIME_VARYING=1"
set "KMPC_SUBPLATOON=2,3,1,4,2"
set "KMPC_NFUT=15"
set "KMPC_SDES=40"
set "KMPC_SPA_MIN=10"
set "KMPC_SPA_MAX=80"
set "KMPC_VEL_MIN=0"
set "KMPC_VEL_MAX=35"
set "KMPC_ACC_MIN=-6"
set "KMPC_ACC_MAX=4"
set "KMPC_DISTURBANCE_ESTIMATE=1"
set "KMPC_LAM=0.7"
set "KMPC_NPOINT=20"
set "KMPC_EP_WINDOW=20"
set "KMPC_ABLATION_FREEZE_STEP=50"
REM Serial learning within each run avoids thread scheduling differences; four runs in parallel.
set "KMPC_CORR_THREADS=1"
set "KMPC_GRB_THREADS=1"
REM The runner enables identical slack penalties only for the separate soft-rerun stage.
set "KMPC_ENABLE_SLACK=0"
set "KMPC_SAVE_SLACK=0"
set "KMPC_HARD_FIRST_SLACK_FALLBACK=0"
set "KMPC_SLACK_W=10000"
set "KMPC_SLACK_W_L2=100"
set "KMPC_SLACK_TOL=0.001"
set "KMPC_DEC_Q_SPA=2.5"
set "KMPC_DEC_Q_VEL=1.5"
set "KMPC_DEC_QA_SPA=0"
set "KMPC_DEC_QA_VEL=0"
set "KMPC_DEC_R=20"
set "KMPC_DEC_RD=30"
set "KMPC_DIST_Q1_SPA=0.125"
set "KMPC_DIST_Q1_VEL=0.1"
set "KMPC_DIST_Q2_SPA=0.125"
set "KMPC_DIST_Q2_VEL=0.1"
set "KMPC_DIST_QA_SPA=0.125"
set "KMPC_DIST_QA_VEL=0.5"
set "KMPC_DIST_Q4_SPA=0.125"
set "KMPC_DIST_Q4_VEL=1"
set "KMPC_DIST_R=20"
set "KMPC_DIST_RD=30"
set "KMPC_DIST_TERM_REFS=weighted"
set "KMPC_SIGNAL_DELAY=0"
set "KMPC_CAV_PRED_LOSS=0"
set "KMPC_SIM_MAXTIME=0"
set "KMPC_SAVE_PLOTS=0"
set "KMPC_VERBOSE_STEP=0"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "PYTHONHASHSEED=20261020"
set "PYTHONUNBUFFERED=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "CUDA_VISIBLE_DEVICES=-1"
set "MPLBACKEND=Agg"
pushd "%~dp0"
"%PYTHON%" -B -u run_repeated_safety.py %*
set "RUN_EXIT=%ERRORLEVEL%"
popd
echo Repeated safety evaluation finished, exit code %RUN_EXIT%
if "%~1"=="" pause
exit /b %RUN_EXIT%
