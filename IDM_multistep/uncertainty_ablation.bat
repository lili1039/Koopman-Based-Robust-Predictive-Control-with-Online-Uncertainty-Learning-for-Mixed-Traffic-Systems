@echo off
setlocal
REM Uncertainty ablation. Edit these settings, then run from an activated Python environment.
REM Four methods: nominal / fixed global / frozen learned local / online local.
REM Use a fixed result name; existing names receive _2, _3, etc.
REM Root run_settings.txt saves this BAT and resolved settings; each case also saves its settings.

REM Clear inherited simulation overrides: this BAT defines the complete experiment.
for /f "tokens=1 delims==" %%V in ('set KMPC_ 2^>nul') do set "%%V="
REM Use python from the activated environment.
set "PYTHON=python"
"%PYTHON%" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo Python could not be started. Activate your project environment.
    exit /b 1
)

REM ==== Batch and arithmetic cut-in times ====
REM (84 + 10*i)*0.12 s, i=0..5 -> 10.08,11.28,12.48,13.68,14.88,16.08.
REM Use 1..4 workers. Runtime measurements remain excluded from this comparison.
set "KMPC_BATCH_WORKERS=4"
set "KMPC_BATCH_START_STEP=84"
set "KMPC_BATCH_INTERVAL_STEPS=10"
set "KMPC_BATCH_COUNT=6"
set "KMPC_BATCH_RESULT_PREFIX=result_uncertainty_ablation"
set "KMPC_BATCH_LAUNCHER=%~f0"
REM ==== Cut-in scene ====
REM GAP is ACTUAL PV-CAV1 spacing immediately after cut-in, NOT spacing loss.
REM PV cruises at 20 m/s, then accelerates at +2 m/s^2 for 3 s before cut-in to 26 m/s.
REM At cut-in, brake to FINAL_SPEED over BRAKE_DURATION, then hold speed.
REM Current deceleration: (4-26)/5.5 = -4 m/s^2. No additional learning prefix.
set "KMPC_CUTIN_GAP=25.5"
set "KMPC_CUTIN_FINAL_SPEED=4"
set "KMPC_CUTIN_BRAKE_DURATION=5.5"

REM ==== Prediction, platoon and physical bounds ====
REM NFUT <= 15. Sampling interval 0.12 s and history length 50 belong to the learned model.
REM Each subplatoon entry is its HDV count/model id (1..4); it also contains one CAV.
set "KMPC_NFUT=15"
set "KMPC_SUBPLATOON=2,3,1,4,2"
set "KMPC_SDES=40"
set "KMPC_SPA_MIN=10"
set "KMPC_SPA_MAX=80"
set "KMPC_VEL_MIN=0"
set "KMPC_VEL_MAX=35"
set "KMPC_ACC_MIN=-6"
set "KMPC_ACC_MAX=4"
set "KMPC_VEHICLE_CSV=vehicle_parameters.csv"
set "KMPC_TIME_VARYING=1"

REM ==== Distributed MPC cost ====
REM Four spacing weights: 0.125 each, total 0.5, matching the retained results.
REM Q1: PV; Q2: preceding CAV; QA: assumed own trajectory; Q4: local leader.
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
REM weighted, or a comma-separated subset of pv,preceed,ref,assume.
set "KMPC_DIST_TERM_REFS=weighted"

REM ==== Uncertainty learning ====
REM Freeze after 50 control steps (6 s); online continues updating. No extra warm-up.
set "KMPC_LAM=0.7"
set "KMPC_NPOINT=20"
set "KMPC_EP_WINDOW=20"
set "KMPC_ABLATION_FREEZE_STEP=50"
set "KMPC_CORR_THREADS=8"

REM ==== Output and computation ====
set "KMPC_SAVE_PLOTS=1"
set "KMPC_VERBOSE_STEP=0"
set "KMPC_GRB_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "PYTHONHASHSEED=202610"
set "PYTHONUNBUFFERED=1"
set "MPLBACKEND=Agg"

REM ==== Per-method slack ====
REM Compare separate hard and soft stages; do not mix hard-first fallback into them.
set "KMPC_HARD_FIRST_SLACK_FALLBACK=0"
REM 1 softens spacing, velocity and terminal velocity; acceleration bounds remain hard.
REM The hard stage disables slack; the soft stage enables it for all four methods.
set "KMPC_BATCH_SLACK_NOMINAL=1"
set "KMPC_BATCH_SLACK_GLOBAL=1"
set "KMPC_BATCH_SLACK_FROZEN=1"
set "KMPC_BATCH_SLACK_ONLINE=1"
REM Use the original solver's common L1/L2 slack penalties for every soft method.
set "KMPC_SLACK_W=10000"
set "KMPC_SLACK_W_L2=100"
set "KMPC_SLACK_TOL=0.001"

REM ==== Fixed comparison protocol ====
REM Zero delay/loss, complete 80 s profile (last sample 79.92 s).
REM Nominal disables tightening only; all methods share the same forecast path.
set "KMPC_SCENARIO=cutin"
set "KMPC_CONTROL=Distributed MPC"
set "KMPC_DISTURBANCE_ESTIMATE=1"
set "KMPC_SIGNAL_DELAY=0"
set "KMPC_CAV_PRED_LOSS=0"
set "KMPC_SIM_MAXTIME=0"

pushd "%~dp0"
REM --check-config validates settings only, without running simulations.
REM --review-evidence runs fresh hard/soft comparisons plus dedicated serial timing.
REM --review-hard runs only 24 hard-constraint cases, without soft runs or timing.
REM --review-performance runs the hard/soft stages only.
REM --serial-timing RESULT_ROOT adds dedicated timing and the final summary later.
REM Default: all six timings, hard/soft comparisons, then a dedicated serial timing stage.
if "%~1"=="" (
    "%PYTHON%" run_uncertainty_ablation.py --review-evidence
) else (
    "%PYTHON%" run_uncertainty_ablation.py %*
)
set "RUN_EXIT=%ERRORLEVEL%"
popd
echo.
echo ==== comparison finished, exit code %RUN_EXIT% ====
if /i not "%~1"=="--check-config" pause
exit /b %RUN_EXIT%
