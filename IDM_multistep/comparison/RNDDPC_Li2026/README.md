# RNDDPC (Li et al., 2026) — baseline

Implementation of the baseline from

> S. Li, J. Wang, K. Yang, Q. Xu, J. Wang, K. Li, *"Robust nonlinear data-driven
> predictive control for mixed vehicle platoons via Koopman operator and
> reachability analysis"*, Transportation Research Part C 182 (2026) 105410.

Built on the same IDM mixed-traffic framework as the parent
`IDM/Koopman_MPC_IDM_learning_set.py` (same scenarios, IDM HDVs, limits, cost,
plotting/IO), so it can be compared on an equal footing with our method and the
other baselines under `IDM/comparison/`.

## Key idea (RNDDPC vs. our method)

| | our Koopman tube-MPC | RNDDPC (this) |
|---|---|---|
| Lifting | Koopman **with** LSTM history | Koopman **without** history (paper's deep EDMD) |
| Robustness source | single additive error zonotope `W` (online-adapted) on `omega` | **matrix zonotope** sets `M_ABH`, `M_C` (uncertainty in the *model matrices*) + additive `Z_sigma`, `Z_rho` |
| Reachable set | propagate `W` through nominal `A` | recursive data-driven reachable set `R^z(i+1)=M_ABH(R^z x Z_u x Z_eps)+Z_sigma`, `R(i+1)=M_C R^z+Z_rho` (eq. 45) |
| Constraint handling | tighten by `Ex_interval` | over-approximate `R(i+1)` to an interval, tighten (eq. 49–50) |

## Result (legacy braking, old config: `N=5`, `sigma_scale=0.1`, `rcond=1e-2`)

The historical numbers below used `[1C1H, 1C3H, 1C2H, 1C3H, 1C1H]` and
`vehicle_parameters_1.csv`. The current fair-comparison default follows
`run_mpc.bat`: `KMPC_SUBPLATOON=2,3,1,4,2`, `vehicle_parameters.csv`,
`KMPC_NFUT=15`, `KMPC_SDES=40`, spacing bounds `[10,80]`, acceleration bounds
`[-6,4]`, and RNDDPC Q/R weights from `KMPC_DEC_Q_SPA/KMPC_DEC_Q_VEL/KMPC_DEC_R`.
RNDDPC sits between the uncontrolled platoon and our methods —
exactly where a *decentralized robust* baseline should (the paper's RNDDPC is
also decentralized per subplatoon, so it cannot exploit inter-CAV coupling like
our Distributed MPC):

| method (braking) | RMSVE [m/s] | RMSSE [m] | | method (tracking/NEDC) | RMSVE | RMSSE |
|---|---|---|---|---|---|---|
| All HDVs | 5.87 | 18.27 | | All HDVs | 2.95 | 16.92 |
| **RNDDPC (this)** | **4.44** | **15.88** | | **RNDDPC (this)** | **2.34** | **19.05** |
| our Decentralized MPC | 4.68 | 14.36 | | our Decentralized MPC | 2.45 | 14.87 |
| our Distributed MPC | 3.60 | 13.30 | | our Distributed MPC | 1.88 | 14.97 |

(Our `All HDVs` reproduces the parent's number to the digit, confirming the
shared framework.) Per-step cost ≈ 4 ms QP + 37 ms reachable tube.

**Reading the numbers.** On both scenarios RNDDPC tracks *velocity* well — on
NEDC it even edges out our Decentralized MPC (2.34 vs 2.45). Its *spacing* on
NEDC is larger (19.05), driven by the short `N=5` horizon (the paper's choice for
reachable-set methods): with only 0.6 s of preview the CAV reacts late to long
acceleration ramps and a couple of subplatoons let their gap grow. We verified
this is **not** a model-quality issue (5-step prediction RMSE is 2–3 m / ~1 m/s
across all sizes) nor a head-velocity-prediction issue (extrapolating `eps`
changes RMSSE by <0.1 %). It is the genuine horizon/robustness trade-off of the
method — exactly the kind of gap a longer-horizon, communication-coupled
controller (our Distributed MPC) is expected to close.

## Pipeline (mapping to the paper)

Per `1 CAV + N HDV` subplatoon model (history-free checkpoints
`2026_06_02_10_41_46` (1H), `2026_06_01_14_54_33` (2H), `2026_06_02_10_43_15`
(3H), `2026_06_02_10_44_27` (4H)):

1. **Lifting** [Sec 3.3 / 4.2.1] — use the trained encoder only:
   `z(k) = [x(k); encoder(x(k))]`, `n_z = 2 + 10·N`.
2. **Data collection** [Sec 4.1] — `simulate_PE_data`: excite the platoon
   (IDM HDVs; CAV = IDM + persistently-exciting perturbation; head-vehicle
   velocity = random walk; small measurement noise), lift to `z`, build the
   sequences `Z-,Z+,U-,E-,X-,X+` of eq. (25).
3. **LS Koopman fit** [eq. 20,21 / Lemma 1 center] — `fit_koopman_ls`:
   `[A B H] = Z+ [Z-;U-;E-]^+`, `C = X+ [Z+]^+`. These are the **centers** of the
   matrix zonotopes `M_ABH`, `M_C`.
4. **Error bounds** [eq. 37 / App. E] — `error_bounds`: `sigma_max`, `rho_max`
   from a per-dimension quantile of the residuals `sigma = Z+-[A B H]D`,
   `rho = X+-C Z+` (covers `Q_COVER` of the data, like the paper's 99.2 % / 98.4 %).
5. **Reachable tube** [Lemma 2 / eq. 45] — `reachable_tube`: recursively
   propagates the compact matrix-zonotope models `M_ABH` and `M_C`, then converts
   the resulting reachable set to interval bounds for constraint tightening.
6. **RNDDPC QP** [eq. 50] — `RNDDPC_Solver`: convex QP with Koopman-aligned
   Q/R/RD weights, acceleration limits, tightened spacing/velocity constraints,
   and a softened terminal velocity equality.

## Adaptations to this codebase (documented departures from the paper)

- **Data is simulated** (persistently-exciting roll-outs) rather than collected
  in PreScan; the encoder itself was trained on HighD.
- **No attack channel.** The paper carries an attack input `theta`/`J`; the
  parent framework has none, so we keep only the disturbance `eps` (head-vehicle
  velocity) and the model error. The machinery is otherwise the paper's.
- **QP execution.** The online problem is solved with OSQP only. Constraint
  infeasibility is handled inside the QP by slack variables on spacing, velocity,
  and terminal velocity; no shifted-input fallback policy is applied.
- **Numerical conditioning.** IDM-generated data is strongly rank-deficient
  (`cond ~1e19`); the default `pinv` keeps near-null directions and the
  matrix-zonotope tube blows up to ~`1e8` m. We use a **truncated-SVD
  pseudo-inverse** (`RCOND_PINV`), the same role the `dDeeP_LCC_Shang2024`
  baseline's Tikhonov regularisation plays.
- **Head-velocity prediction.** The default follows the paper: the nominal head
  velocity is held constant over the horizon (eq. 47f), with its variation
  robustified through the reachable set `Z_eps`. (`RNDDPC_EP_EXTRAP=1` extrapolates
  it with the head's current acceleration like the parent MPC; over the short
  `N=5` horizon this changes the result negligibly — see the note on spacing below.)
- **Horizon.** The paper deliberately uses a short `N=5` for the reachable-set
  methods, because the data-driven reachable tube grows over the horizon. The
  current fair-comparison default follows `run_mpc.bat` and uses `KMPC_NFUT=15`;
  set `KMPC_NFUT=5` to reproduce the older RNDDPC setting.
- **Tunable uncertainty.** `SIGMA_SCALE` (and `Q_COVER`) scale `sigma_max`/
  `rho_max`; lower them to shrink the tube if (50) becomes infeasible (the
  "how many sigma" knob). Defaults give a horizon-end tube of ~5 m / ~6.5 m/s.

## Files

- `functions.py` — config + offline pipeline (`DeepEDMD_NoLSTM`,
  `simulate_PE_data`, `fit_koopman_ls`, `error_bounds`, `build_rnddpc_bundle`),
  reachable set (`reachable_tube`, `nominal_lifted_rollout`, `Interval`) and the
  `RNDDPC_Solver` QP. Run directly (`python functions.py`) to smoke-test the
  offline build for all subplatoon sizes.
- `RNDDPC.py` — closed-loop simulation (subplatoon decomposition identical to the
  parent). Run to reproduce an experiment.
- `result/<scenario>/<method>_<timestamp>/` — trajectories, plots, metrics, per-step
  computation time.

## Configuration (env vars)

- `KMPC_SCENARIO` : `braking` | `tracking` | `stabilization` | `cutin`
- `KMPC_CONTROL`  : `RNDDPC` | `All HDVs`
- `KMPC_SUBPLATOON`: comma-separated HDV counts per subplatoon (default `2,3,1,4,2`)
- `KMPC_NFUT`     : prediction horizon (default 15)
- `KMPC_VEHICLE_CSV`: vehicle parameter CSV (default `vehicle_parameters.csv`)
- `RNDDPC_SIGMA_SCALE` : scale on `sigma_max`/`rho_max` (default 0.01)
- `RNDDPC_QCOVER` : residual coverage quantile (default 0.90)
- `RNDDPC_RCOND`  : truncated-SVD cutoff for the data pseudo-inverse (default 1e-2)
- `RNDDPC_EPS_MAX`: head-vehicle velocity uncertainty bound (default 0.2)
- `RNDDPC_MZ_KEEP`: retained SVD directions for offline matrix-zonotope generator
  reduction (default 999, which keeps the full low-dimensional input basis; this
  is still tens of generators after reduction). The residual generator mass is
  wrapped in a coordinate box, so online reachable-set constraints avoid the full
  PE data length.
- `RNDDPC_DATA_VBASE_MIN/MAX`, `RNDDPC_DATA_HEAD_MIN/MAX`,
  `RNDDPC_DATA_HEAD_STD`, `RNDDPC_DATA_PE_STD`: PE data operating envelope.
  Defaults cover the NEDC/braking operating range used in the comparison.
- `RNDDPC_TUBE_MODE`: `nominal_qp` for the fast shifted-input tube, or
  `paper_conic` for the stricter `Z_u=<u(i|k),0>` conic formulation
  (default `paper_conic`).
- `RNDDPC_CONIC_SOLVER`: conic solver for `paper_conic` (default `CLARABEL`).
- `RNDDPC_W_SPACING_LOW`: one-sided penalty for spacing below the reference spacing (default `200`).
- `RNDDPC_W_TERM_SPACING`: soft terminal penalty for spacing below the reference spacing (default `1000`).

## Run

```powershell
# offline smoke test (all subplatoon sizes)
python functions.py
# closed loop
$env:KMPC_SCENARIO='braking';  python RNDDPC.py
$env:KMPC_SCENARIO='tracking'; python RNDDPC.py
# reference (no CAV control)
$env:KMPC_CONTROL='All HDVs';  python RNDDPC.py
```

Use the `python310` conda env (`C:\Users\13373\.conda\envs\python310`).
