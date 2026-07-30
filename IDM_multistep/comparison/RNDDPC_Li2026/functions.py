# =============================================================================
#  RNDDPC (Li et al., 2026, TR-C) — baseline for comparison
#
#  Robust Nonlinear Data-Driven Predictive Control via Koopman operator and
#  reachability analysis.  This file implements the *offline* pipeline and the
#  reachable-set / QP machinery; the closed-loop driver is in `RNDDPC.py`.
#
#  Faithful pieces (paper section in brackets):
#    - Koopman lifting z(k)=[x(k); encoder(x(k), history)]                 [3.3/4.2.1]
#    - LS-refit of [A B H] and C on collected data  (Lemma 1 center)       [eq 20,21,39,40]
#    - model-error bounds sigma_max / rho_max from residual statistics     [eq 37, App.E]
#    - matrix-zonotope reachable set  R^z(i+1)=M_ABH(R^z x Zu x Zeps)+Zsig  [Lemma 2 / eq 45]
#      over-approximated to an interval each step for constraint tightening [eq 49,50d]
#
#  Documented adaptations to *this* codebase (see README):
#    - data is simulated by exciting the platoon (PE) instead of PreScan    [data collection]
#    - the attack channel J/theta is dropped (the parent framework has no
#      attack); only disturbance eps (head-vehicle velocity) + model error
#    - two online tube modes are available: the fast nominal-QP mode decouples
#      the tube from the current decision, while paper_conic keeps Zu=<u,0>
#      inside the optimization and solves the resulting conic problem.
#    - sigma_max/rho_max scaled by a tunable factor (SIGMA_SCALE, Q_COVER) so
#      the HighD-trained model's larger residuals do not make (50) infeasible
# =============================================================================
import os
# Avoid the Windows libiomp5md.dll duplicate-init hard crash (torch+MKL).
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import math
import time
import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
from pyzonotope import Zonotope

# ---- reuse the parent IDM framework (simulation / profiles / IO / plotting) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_IDM_DIR = os.path.dirname(os.path.dirname(_HERE))            # .../IDM
PROJECT_ROOT = os.path.dirname(_IDM_DIR)                       # .../from_server
if _IDM_DIR not in sys.path:
    sys.path.insert(0, _IDM_DIR)
import _heter_helperfuns_torch_IDM_learning_set as helper      # noqa: E402
helper.project_path = PROJECT_ROOT
from _heter_helperfuns_torch_IDM_learning_set import (         # noqa: E402
    IDM_dynamics,
    compute_IDM_steady_s,
    generate_NEDC_velocity_profile,
    generate_braking_velocity_profile,
    generate_sine_velocity_profile,
    generate_history_seq,
    plot,
    compute_tracking_metrics,
    save_vehicle_data_to_csv,
    save_metrics,
    save_computation_time,
    _env_flag,
    _env_float,
    _env_int,
    _env_int_list,
)

# =============================================================================
#                               Configuration
# =============================================================================
# scenario ∈ {'braking','tracking','stabilization','cutin'}; env KMPC_SCENARIO overrides
scenario = os.environ.get('KMPC_SCENARIO', 'braking')

# control_method ∈ {'All HDVs','RNDDPC'}; env KMPC_CONTROL overrides
control_method = os.environ.get('KMPC_CONTROL', 'RNDDPC')

result_root = os.environ.get('KMPC_RESULT_ROOT', 'result')
run_suffix = os.environ.get('KMPC_RUN_SUFFIX', '')
if not run_suffix.strip():
    run_suffix = '_' + time.strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join(result_root, scenario, control_method.replace(' ', '_') + run_suffix)

# ---- horizon / limits ----
Nfut = _env_int('KMPC_NFUT', 5)
subplatoon_spec = _env_int_list('KMPC_SUBPLATOON', [2, 3, 1, 4, 2])
SpaMin = _env_float('KMPC_SPA_MIN', 10.0)
SpaMax = _env_float('KMPC_SPA_MAX', 80.0)
VelMin = _env_float('KMPC_VEL_MIN', 0.0)
VelMax = _env_float('KMPC_VEL_MAX', 35.0)
AccMin = _env_float('KMPC_ACC_MIN', -6.0)
AccMax = _env_float('KMPC_ACC_MAX', 4.0)
sdes = _env_float('KMPC_SDES', 40.0)
vdes_stab = _env_float('KMPC_VDES_STAB', 25.0)
vehicle_csv = os.environ.get('KMPC_VEHICLE_CSV', os.environ.get('KMPC_VEH_PARAM', 'vehicle_parameters.csv'))
vehicle_csv_path = vehicle_csv if os.path.isabs(vehicle_csv) else os.path.join(_IDM_DIR, vehicle_csv)

cutin_time = _env_float('KMPC_CUTIN_TIME', 8.0)
cutin_drop = _env_float('KMPC_CUTIN_DROP', 25.0)

def _env_float_first(names, default):
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return float(value)
    return default

rnddpc_q_spa = _env_float_first(['KMPC_DEC_Q_SPA', 'KMPC_Q_SPA'], 2.5)
rnddpc_q_vel = _env_float_first(['KMPC_DEC_Q_VEL', 'KMPC_Q_VEL'], 1.5)
rnddpc_r = _env_float_first(['KMPC_DEC_R', 'KMPC_R'], 20.0)
rnddpc_rd = _env_float_first(['KMPC_DEC_RD', 'KMPC_RD'], 30.0)

# Slack follows the Koopman decentralized MPC convention: it is part of the QP,
# heavily penalized, and remains exactly zero when hard constraints are feasible.
W_SLACK = _env_float('KMPC_SLACK_W', 1.0e4)
W_SLACK_L2 = _env_float('KMPC_SLACK_W_L2', 1.0e2)
W_SPACING_LOW = _env_float('RNDDPC_W_SPACING_LOW', 200.0)
W_TERM_SPACING = _env_float('RNDDPC_W_TERM_SPACING', 1000.0)

# =============================================================================
#  RNDDPC uncertainty knobs  (the "how many sigma" tuning the user asked for)
# =============================================================================
# sigma_max / rho_max are taken as a per-dimension quantile of |residual| (covers
# Q_COVER of the data like the paper's 99.2%/98.4%) and then scaled by SIGMA_SCALE.
# Lower either to shrink the reachable tube if (50) is infeasible.
Q_COVER     = float(os.environ.get('RNDDPC_QCOVER', 0.90))
SIGMA_SCALE = float(os.environ.get('RNDDPC_SIGMA_SCALE', 0.01))
EPS_MAX     = float(os.environ.get('RNDDPC_EPS_MAX', 0.2))   # head-vehicle velocity uncertainty bound
# Nominal head-velocity prediction over the horizon. The paper assumes it
# constant (eq 47f) and robustifies variation via the reachable set Z_eps; on a
# varying cycle (NEDC) that lags during sustained ramps, so by default we
# extrapolate with the current head acceleration (as the parent MPC does) and
# keep the matrix-zonotope tube as the only RNDDPC-specific element. Set
# RNDDPC_EP_EXTRAP=0 to recover the paper's constant-velocity assumption.
EP_EXTRAPOLATE = os.environ.get('RNDDPC_EP_EXTRAP', '1') not in ('0', 'false', 'False')
# Truncated-SVD pseudo-inverse cutoff. IDM-generated data is strongly
# rank-deficient (cond ~1e19); the default rcond≈1e-15 keeps near-null
# directions, which blows up ||D^+ p||_1 and hence the matrix-zonotope tube.
# Truncating them tames the tube (same role as the dDeeP-LCC Tikhonov reg).
RCOND_PINV  = float(os.environ.get('RNDDPC_RCOND', 1e-2))

# Online handling of Z_u=<u(i|k),0> in eq. (45).
#   nominal_qp:  fast baseline; propagate tube with shifted previous control.
#   paper_conic: executable RNDDPC (50); the interval reachable set is built
#                inside the optimization and depends on the decision sequence u.
TUBE_MODE = os.environ.get('RNDDPC_TUBE_MODE', 'paper_conic').strip().lower()
if TUBE_MODE not in ('nominal_qp', 'paper_conic'):
    raise ValueError("RNDDPC_TUBE_MODE must be 'nominal_qp' or 'paper_conic'")
CONIC_SOLVER = os.environ.get('RNDDPC_CONIC_SOLVER', 'CLARABEL').strip().upper()

# ---- data collection (persistently exciting roll-outs) ----
DATA_ROLLOUTS   = int(os.environ.get('RNDDPC_DATA_ROLLOUTS', 30))
DATA_ROLLOUT_LEN = int(os.environ.get('RNDDPC_DATA_LEN', 120))
DATA_NOISE      = float(os.environ.get('RNDDPC_DATA_NOISE', 0.03))  # measurement noise ~ U[-n,n]
DATA_SEED       = int(os.environ.get('RNDDPC_DATA_SEED', 0))
DATA_VBASE_MIN  = float(os.environ.get('RNDDPC_DATA_VBASE_MIN', 10.0))
DATA_VBASE_MAX  = float(os.environ.get('RNDDPC_DATA_VBASE_MAX', 30.0))
DATA_HEAD_MIN   = float(os.environ.get('RNDDPC_DATA_HEAD_MIN', 10.0))
DATA_HEAD_MAX   = float(os.environ.get('RNDDPC_DATA_HEAD_MAX', 30.0))
DATA_HEAD_STD   = float(os.environ.get('RNDDPC_DATA_HEAD_STD', 0.8))
DATA_PE_STD     = float(os.environ.get('RNDDPC_DATA_PE_STD', 1.5))
# Matrix-zonotope generator reduction. The paper does not specify the retained
# generator count; CORA-style workflows typically reduce generator order before
# online reachability. We keep an SVD subspace of the data-driven generator rows
# and aggregate the residual into an axis-aligned box, yielding an outer
# approximation with O(keep + dimension) generators instead of O(T).
MZ_KEEP_GENERATORS = int(os.environ.get('RNDDPC_MZ_KEEP', 999))

# Koopman checkpoint selection.  By default RNDDPC now uses the same with-history
# multistep checkpoints as `Koopman_MPC_IDM_learning_set.py`; set
# RNDDPC_USE_HISTORY=0 only for the old history-free ablation.
USE_HISTORY_KOOPMAN = _env_flag('RNDDPC_USE_HISTORY', True)
WITH_HISTORY_MODEL_PATHS = {
    1: helper.HDV_1_MODEL_PATH,
    2: helper.HDV_2_MODEL_PATH,
    3: helper.HDV_3_MODEL_PATH,
    4: helper.HDV_4_MODEL_PATH,
}

# history-free (NoLSTM) Koopman checkpoints: num_HDV -> (date_folder, best_epoch)
HF_MODELS = {
    1: ('2026_06_02_10_41_46', 104),
    2: ('2026_06_01_14_54_33', 49),
    3: ('2026_06_02_10_43_15', 186),
    4: ('2026_06_02_10_44_27', 188),
}

Tstep = 0.12


# =============================================================================
#  History-free Koopman model (matches training_code/.../_3_..._NoLSTM.py)
# =============================================================================
class DeepEDMD_NoLSTM(nn.Module):
    """z(k) = [x(k); encoder(x(k))];  evolution g@A + u@B + ep@H;  x = decoder(g).

    Architecture replicated verbatim from `_3_heter_mainfuns_highd_NoLSTM.py`
    so the saved state_dict loads exactly. Only the lifting (encoder) is used by
    RNDDPC; A/B/H/decoder are re-identified from data (Lemma 1).
    """

    def __init__(self, num_HDV, keep_prob=0.8):
        super().__init__()
        self.num_HDV = num_HDV
        self.s_dim = 2 * (1 + num_HDV)
        self.lift_dim = 8 * num_HDV
        encoder_widths = [self.s_dim, 112, 96, self.lift_dim]
        eact = ['relu', 'relu', 'relu']

        layers = []
        prev = encoder_widths[0]
        for width, act in zip(encoder_widths[1:], eact):
            layers.append(nn.Linear(prev, width))
            if act == 'relu':
                layers.append(nn.ReLU())
            if keep_prob < 1.0:
                layers.append(nn.Dropout(p=1 - keep_prob))
            prev = width
        self.encoder = nn.Sequential(*layers)

        total_state_dim = self.s_dim + self.lift_dim
        self.decoder = nn.Linear(total_state_dim, self.s_dim, bias=False)
        # Koopman params (loaded but unused by RNDDPC; kept so state_dict matches)
        self.A = nn.Parameter(torch.zeros(total_state_dim, total_state_dim))
        self.B = nn.Parameter(torch.zeros(1, total_state_dim))
        self.H = nn.Parameter(torch.zeros(1, total_state_dim))

    @torch.no_grad()
    def lift(self, x_np):
        """x_np: (..., s_dim) numpy -> z: (..., s_dim+lift_dim) numpy."""
        x = torch.as_tensor(np.atleast_2d(x_np), dtype=torch.float32)
        encoded = self.encoder(x)
        g = torch.cat([x, encoded], dim=-1)
        return g.cpu().numpy()


def load_history_free_model(num_HDV):
    date_folder, epoch = HF_MODELS[num_HDV]
    path = os.path.join(PROJECT_ROOT, 'Deep_Koop_v1', 'model_highd', date_folder,
                        'past_models', f'model_epoch_{epoch}.weights.h5')
    model = DeepEDMD_NoLSTM(num_HDV)
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    print(f"[RNDDPC] loaded history-free encoder num_HDV={num_HDV}: {date_folder} (epoch {epoch}), "
          f"s_dim={model.s_dim}, n_z={model.s_dim + model.lift_dim}")
    return model


class KoopmanLiftWrapper:
    """Small numpy-facing wrapper around either RNDDPC lifting encoder."""

    def __init__(self, model, args, num_HDV, use_history, model_ref, device='cpu'):
        self.model = model
        self.args = args
        self.num_HDV = num_HDV
        self.use_history = use_history
        self.model_ref = model_ref
        self.device = device
        self.s_dim = args['s_dim'] if args is not None else model.s_dim
        self.lift_dim = args['lift_dim'] if args is not None else model.lift_dim
        self.n_z = self.s_dim + self.lift_dim
        self.Npst = args['Npst'] if args is not None and 'Npst' in args else None
        self.slice_dim = args['HV_hist_slice_dim'] if args is not None and 'HV_hist_slice_dim' in args else None

    @torch.no_grad()
    def lift(self, x_np, x_history_np=None):
        if not self.use_history:
            return self.model.lift(x_np)
        if x_history_np is None:
            raise ValueError('with-history Koopman lifting requires x_history_np')
        x = torch.as_tensor(np.atleast_2d(x_np), dtype=torch.float32, device=self.device)
        hist = torch.as_tensor(x_history_np, dtype=torch.float32, device=self.device)
        if hist.ndim == 4:
            pass
        elif hist.ndim == 5 and hist.shape[1] == 1:
            hist = hist[:, 0]
        else:
            raise ValueError(f'x_history_np shape {tuple(hist.shape)} is not compatible')
        if hist.shape[0] == 1 and x.shape[0] > 1:
            hist = hist.repeat(x.shape[0], 1, 1, 1)
        g = self.model.encode(x, hist)
        return g.detach().cpu().numpy()


def load_rnddpc_lift_model(num_HDV):
    if USE_HISTORY_KOOPMAN:
        args = vars(helper.get_args_local(False, num_HDV_override=num_HDV))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rel_path = WITH_HISTORY_MODEL_PATHS[num_HDV]
        model_path = os.path.join(PROJECT_ROOT, 'Deep_Koop_v1', 'model_highd_multistep', rel_path)
        model = helper.load_trained_model(args, model_path, device)
        print(f"[RNDDPC] loaded with-history Koopman encoder num_HDV={num_HDV}: {rel_path} "
              f"s_dim={args['s_dim']}, n_z={args['s_dim'] + args['lift_dim']}")
        return KoopmanLiftWrapper(model, args, num_HDV, True, rel_path, device)
    model = load_history_free_model(num_HDV)
    args = {
        's_dim': model.s_dim,
        'lift_dim': model.lift_dim,
        'Npst': None,
        'HV_hist_slice_dim': None,
    }
    date_folder, epoch = HF_MODELS[num_HDV]
    return KoopmanLiftWrapper(model, args, num_HDV, False,
                              f"{date_folder}/past_models/model_epoch_{epoch}.weights.h5")


# =============================================================================
#  Data collection phase  — persistently-exciting platoon roll-outs   [Sec 4.1]
# =============================================================================
def load_vehicle_parameters(required_rows=None):
    param_data = np.genfromtxt(vehicle_csv_path, delimiter=',', skip_header=1)
    param_data = np.atleast_2d(param_data)
    if required_rows is not None and param_data.shape[0] < required_rows:
        raise ValueError(f'{vehicle_csv_path}: {param_data.shape[0]} rows < {required_rows} vehicles')
    if param_data.shape[1] < 8:
        raise ValueError(f'{vehicle_csv_path}: need columns Tgap,v0_IDM,veh_len,Tgap_rate,v0_rate,a_IDM,b_IDM,s0_IDM')
    return param_data


def simulate_PE_data(model, num_HDV, idm_params, n_rollouts=DATA_ROLLOUTS,
                     rollout_len=DATA_ROLLOUT_LEN, noise=DATA_NOISE, seed=DATA_SEED):
    """Excite a (1 CAV + num_HDV HDV) platoon and collect lifted data sequences.

    Returns dict with X-,X+,U-,E-,Z-,Z+  (columns = time steps), matching eq (25).
    HDVs follow IDM; the CAV follows IDM + a persistently-exciting perturbation;
    the head vehicle velocity (disturbance eps) is a random walk; small
    measurement noise is injected before lifting (as in the paper).
    """
    rng = np.random.default_rng(seed)
    num_veh = 1 + num_HDV
    s_dim = 2 * num_veh
    Npst_local = model.Npst if model.use_history else 0
    slice_dim = model.slice_dim if model.use_history else 3
    sim_len = Npst_local + rollout_len

    X_, Xp, U_, E_, Z_, Zp = [], [], [], [], [], []

    for _ in range(n_rollouts):
        Tgap, v0, vlen, a_idm, b_idm, s0_idm = [np.asarray(x, dtype=float).copy() for x in idm_params]
        if len(Tgap) != num_veh:
            raise ValueError(f'idm_params length {len(Tgap)} does not match 1+num_HDV={num_veh}')
        v_base = rng.uniform(DATA_VBASE_MIN, DATA_VBASE_MAX)

        # head-vehicle velocity: random walk inside a band
        head_v = np.zeros(sim_len + 1)
        head_v[0] = v_base
        for k in range(sim_len):
            head_v[k + 1] = np.clip(head_v[k] + rng.normal(0, DATA_HEAD_STD), DATA_HEAD_MIN, DATA_HEAD_MAX)

        # PE perturbation added to the CAV's IDM acceleration (filtered noise)
        du = np.zeros(sim_len)
        for k in range(1, sim_len):
            du[k] = 0.7 * du[k - 1] + rng.normal(0, DATA_PE_STD)
        du = np.clip(du, AccMin, AccMax)

        # init at IDM steady state for v_base
        S = np.zeros((sim_len + 1, num_veh + 1, 3))   # pos/vel/acc; index0=head
        S[:, 0, 1] = head_v
        S[0, 0, 0] = 0.0
        for j in range(num_veh):
            s_eq = compute_IDM_steady_s(v_base, v_base, v0[j], Tgap[j], a_idm[j], b_idm[j], s0_idm[j]) + vlen[j]
            S[0, j + 1, 0] = S[0, j, 0] - s_eq
            S[0, j + 1, 1] = v_base

        for k in range(sim_len):
            acel = IDM_dynamics(S[k, :, :], Tgap, v0, vlen, a_idm, b_idm, s0_idm)
            S[k, 1:, 2] = acel
            S[k, 1, 2] = float(np.clip(acel[0] + du[k], AccMin, AccMax))   # CAV: IDM + PE
            S[k + 1, :, 0] = S[k, :, 0] + Tstep * S[k, :, 1]
            S[k + 1, 1:, 1] = S[k, 1:, 1] + Tstep * S[k, 1:, 2]

        # build x(k), u(k), eps(k) and lift to z(k) (+ measurement noise)
        xk = np.zeros((rollout_len + 1, s_dim))
        xhist = (np.zeros((rollout_len + 1, Npst_local, slice_dim, num_HDV))
                 if model.use_history else None)
        for j in range(num_veh):
            src = slice(Npst_local, Npst_local + rollout_len + 1)
            xk[:, 2 * j] = S[src, j, 0] - S[src, j + 1, 0]
            xk[:, 2 * j + 1] = S[src, j + 1, 1]
        if model.use_history:
            for k in range(rollout_len + 1):
                t = Npst_local + k
                for j in range(num_HDV):
                    hist_slice = slice(t - Npst_local, t)
                    xhist[k, :, 0, j] = S[hist_slice, j + 1, 1]
                    xhist[k, :, 1, j] = S[hist_slice, j + 1, 0] - S[hist_slice, j + 2, 0]
                    xhist[k, :, 2, j] = S[hist_slice, j + 2, 1]
        xk_noisy = xk + rng.uniform(-noise, noise, xk.shape)
        if model.use_history:
            xhist_noisy = xhist + rng.uniform(-noise, noise, xhist.shape)
            zk = model.lift(xk_noisy, xhist_noisy)      # (rollout_len+1, n_z)
        else:
            zk = model.lift(xk_noisy)                   # (rollout_len+1, n_z)
        uk = S[Npst_local:Npst_local + rollout_len, 1, 2]   # CAV acceleration
        epk = S[Npst_local:Npst_local + rollout_len, 0, 1]  # head-vehicle velocity

        X_.append(xk_noisy[:-1].T);  Xp.append(xk_noisy[1:].T)
        Z_.append(zk[:-1].T);        Zp.append(zk[1:].T)
        U_.append(uk[None, :]);      E_.append(epk[None, :])

    data = {
        'X_': np.hstack(X_), 'X+': np.hstack(Xp),
        'Z_': np.hstack(Z_), 'Z+': np.hstack(Zp),
        'U_': np.hstack(U_), 'E_': np.hstack(E_),
    }
    print(f"[RNDDPC] collected PE data num_HDV={num_HDV}: T={data['Z_'].shape[1]}, "
          f"n_z={data['Z_'].shape[0]}")
    return data


# =============================================================================
#  Offline learning phase — LS Koopman fit + matrix-zonotope bundle  [Lemma 1]
# =============================================================================
def fit_koopman_ls(data):
    """LS solution = centers of the matrix zonotopes (eq 39/40):
       [A B H] = Z+ [Z-;U-;E-]^†,   C = X+ [Z+]^†.
       Also returns the pseudo-inverses needed online for the matrix-zono tube.
    """
    Z_, Zp = data['Z_'], data['Z+']
    U_, E_ = data['U_'], data['E_']
    Xp = data['X+']
    n_z = Z_.shape[0]

    D = np.vstack([Z_, U_, E_])                 # (n_z+2, T)
    D_pinv = np.linalg.pinv(D, rcond=RCOND_PINV)  # (T, n_z+2), truncated SVD
    M = Zp @ D_pinv                             # (n_z, n_z+2) = [A B H]
    A = M[:, :n_z]
    B = M[:, n_z:n_z + 1]                        # (n_z,1)
    H = M[:, n_z + 1:n_z + 2]                    # (n_z,1)

    Zp_pinv = np.linalg.pinv(Zp, rcond=RCOND_PINV)  # (T, n_z)
    C = Xp @ Zp_pinv                            # (s_dim, n_z)  decoder/projection

    # residuals (App. E): sigma = Z+ - [A B H]D ,  rho = X+ - C Z+
    sigma_res = Zp - M @ D                       # (n_z, T)
    rho_res = Xp - C @ Zp                        # (s_dim, T)
    return {
        'A': A, 'B': B, 'H': H, 'C': C,
        'D_pinv': D_pinv, 'Zp_pinv': Zp_pinv,
        'sigma_res': sigma_res, 'rho_res': rho_res,
        'n_z': n_z, 's_dim': C.shape[0],
    }


def error_bounds(fit, q_cover=None, scale=None):
    """sigma_max / rho_max: per-dim quantile of |residual| (covers q_cover), scaled.
       Matches the paper's 'bounds obtained from prediction-error statistics'.
       Reads the module-level Q_COVER/SIGMA_SCALE at call time so env/runtime
       overrides take effect (don't bind them as default args)."""
    q_cover = Q_COVER if q_cover is None else q_cover
    scale = SIGMA_SCALE if scale is None else scale
    sigma_max = scale * np.quantile(np.abs(fit['sigma_res']), q_cover, axis=1)
    rho_max = scale * np.quantile(np.abs(fit['rho_res']), q_cover, axis=1)
    # guard against exact zeros (keeps the box non-degenerate)
    sigma_max = np.maximum(sigma_max, 1e-6)
    rho_max = np.maximum(rho_max, 1e-6)
    return sigma_max, rho_max


def reduce_pinv_generators(data_pinv, keep=None):
    """Reduce row-time generators from P=D^dagger or (Z+)^dagger.

    The full matrix zonotope has one generator direction per PE sample. We use
    the leading right-singular directions as shared zonotope generators and wrap
    the residual in a coordinate box. For any vector p,
    sum(|P p|) <= sum(|P_red p|), while P_red has only keep + input_dim
    generators.
    """
    P = np.asarray(data_pinv, dtype=float)
    keep = MZ_KEEP_GENERATORS if keep is None else int(keep)
    keep = max(0, min(keep, min(P.shape)))
    if P.shape[0] == 0:
        return P.copy()

    parts = []
    if keep > 0:
        _, _, vh = np.linalg.svd(P, full_matrices=False)
        basis = vh[:keep, :]                         # (keep, input_dim)
        coeff = P @ basis.T                          # each full generator in the reduced basis
        projected_generators = np.sum(np.abs(coeff), axis=0)[:, None] * basis
        residual = P - coeff @ basis
        parts.append(projected_generators)
    else:
        residual = P

    if residual.size:
        box_radius = np.sum(np.abs(residual), axis=0)
        box = np.diag(box_radius)
        box = box[np.linalg.norm(box, axis=1) > 0]
        parts.append(box)

    return np.vstack(parts) if parts else np.zeros((0, P.shape[1]))


class CompactMatrixZonotope:
    """Compact realization of Lemma 1 matrix zonotopes.

    For M = (Y - M_err) P, where M_err has independent row/time box generators
    from Z_sigma or Z_rho, materializing all row-time matrix generators would be
    enormous. This class stores the center YP, the row-wise error bounds, and a
    reduced outer approximation of P's row generators.
    Multiplication by a zonotope returns the same box over-approximation used by
    matrix-zonotope reachability: center/generators from the nominal center
    matrix plus the interval hull of the matrix-uncertainty contribution.
    """

    def __init__(self, center, error_bounds, data_pinv, name):
        self.center = np.asarray(center, dtype=float)
        self.error_bounds = np.asarray(error_bounds, dtype=float)
        self.data_pinv = np.asarray(data_pinv, dtype=float)
        self.name = name

    def matmul_zonotope(self, zono):
        center = self.center @ zono.center
        generators = []
        if zono.num_generators:
            generators.append(self.center @ zono.generators)

        coeff = float(np.sum(np.abs(self.data_pinv @ zono.center)))
        if zono.num_generators:
            coeff += float(np.sum(np.abs(self.data_pinv @ zono.generators)))
        generators.append(np.diag(self.error_bounds * coeff))

        gen = np.hstack(generators) if generators else np.zeros((self.center.shape[0], 0))
        return Zonotope(center, gen)


def box_zonotope(bounds):
    bounds = np.asarray(bounds, dtype=float)
    return Zonotope(np.zeros(bounds.shape[0]), np.diag(bounds))


def cartesian_zonotope(parts):
    dim = sum(part.dimension for part in parts)
    num_gen = sum(part.num_generators for part in parts)
    center = np.concatenate([part.center for part in parts])
    generators = np.zeros((dim, num_gen))
    row = 0
    col = 0
    for part in parts:
        rows = part.dimension
        cols = part.num_generators
        if cols:
            generators[row:row + rows, col:col + cols] = part.generators
        row += rows
        col += cols
    return Zonotope(center, generators)


def build_rnddpc_bundle(num_HDV, idm_params, bundle_label=None):
    """Full offline pipeline for one (1 CAV + num_HDV HDV) subplatoon model."""
    model = load_rnddpc_lift_model(num_HDV)
    data = simulate_PE_data(model, num_HDV, idm_params)
    fit = fit_koopman_ls(data)
    sigma_max, rho_max = error_bounds(fit)
    D_mz_generators = reduce_pinv_generators(fit['D_pinv'])
    Zp_mz_generators = reduce_pinv_generators(fit['Zp_pinv'])

    # quick sanity: lifted one-step LS fit quality (rho≈0 by construction, since
    # x is embedded in z, so we report the lifted-state residual sigma instead)
    n_z = fit['n_z']
    sig_rmse = np.sqrt(np.mean(fit['sigma_res'] ** 2))
    rho_rmse = np.sqrt(np.mean(fit['rho_res'] ** 2))
    label = f" {bundle_label}" if bundle_label else ""
    print(f"[RNDDPC] num_HDV={num_HDV}{label}: lifted one-step RMSE sigma={sig_rmse:.3f} (rho={rho_rmse:.1e}) | "
          f"sigma_max∈[{sigma_max.min():.3f},{sigma_max.max():.3f}]  "
          f"rho_max∈[{rho_max.min():.3f},{rho_max.max():.3f}] | "
          f"MZ generators D:{fit['D_pinv'].shape[0]}->{D_mz_generators.shape[0]} "
          f"Zp:{fit['Zp_pinv'].shape[0]}->{Zp_mz_generators.shape[0]}")

    bundle = {
        'model': model, 'num_HDV': num_HDV,
        'model_ref': model.model_ref,
        'use_history_koopman': model.use_history,
        'A': fit['A'], 'B': fit['B'], 'H': fit['H'], 'C': fit['C'],
        'D_pinv': fit['D_pinv'], 'Zp_pinv': fit['Zp_pinv'],
        'D_mz_generators': D_mz_generators, 'Zp_mz_generators': Zp_mz_generators,
        'sigma_max': sigma_max, 'rho_max': rho_max,
        'M_ABH': CompactMatrixZonotope(
            np.hstack((fit['A'], fit['B'], fit['H'])),
            sigma_max,
            D_mz_generators,
            'M_ABH',
        ),
        'M_C': CompactMatrixZonotope(fit['C'], rho_max, Zp_mz_generators, 'M_C'),
        'Z_sigma': box_zonotope(sigma_max),
        'Z_rho': box_zonotope(rho_max),
        'n_z': n_z, 's_dim': fit['s_dim'],
        'mz_keep_generators': MZ_KEEP_GENERATORS,
        'mz_generator_counts': {
            'D_full': int(fit['D_pinv'].shape[0]),
            'D_reduced': int(D_mz_generators.shape[0]),
            'Zp_full': int(fit['Zp_pinv'].shape[0]),
            'Zp_reduced': int(Zp_mz_generators.shape[0]),
        },
    }
    return bundle


# =============================================================================
#  Online — matrix-zonotope reachable set -> interval tightening   [Lemma 2/eq45]
# =============================================================================
class Interval:
    """Mimics pyzonotope's interval interface used by the parent MPC solver."""
    __slots__ = ('left_limit', 'right_limit')

    def __init__(self, left, right):
        self.left_limit = np.asarray(left, dtype=float)
        self.right_limit = np.asarray(right, dtype=float)


def reachable_tube(bundle, ep_center, eps_max, Nfut, z_nom_seq, u_nom_seq):
    """Recursive over-approximated data-driven reachable set, returned as the
    per-step interval half-widths (centered at 0) used to tighten constraints.

    Implements eq (45):
        R^z(i+1) = M_ABH ( R^z(i) x Zu x Zeps ) + Zsigma
        R(i+1)   = M_C   R^z(i+1)              + Zrho
    The matrix-zonotope contribution is collapsed to its interval (box) hull each
    step (correlation-free, standard over-approximation); the nominal A/C
    propagation of the existing tube is kept as explicit zonotope generators.

    Z_u = <u_nom(i), 0>, Z_eps = <ep_center, eps_max>: the tube is evaluated on a
    *nominal* trajectory (z_nom_seq, u_nom_seq) so it is independent of the QP
    decision -> keeps (50) a QP (same idea as the parent constraint tightening).
    """
    C = bundle['C']
    M_ABH, M_C = bundle['M_ABH'], bundle['M_C']
    Z_sigma, Z_rho = bundle['Z_sigma'], bundle['Z_rho']
    n_z = bundle['n_z']

    Rz = Zonotope(z_nom_seq[:, 0], np.zeros((n_z, 0)))
    Ex_interval = []

    for i in range(Nfut):
        Zu = Zonotope(np.array([u_nom_seq[i]], dtype=float), np.zeros((1, 0)))
        Zeps = Zonotope(np.array([ep_center], dtype=float), np.array([[eps_max]], dtype=float))

        R_input = cartesian_zonotope([Rz, Zu, Zeps])
        Rz = M_ABH.matmul_zonotope(R_input) + Z_sigma
        R = M_C.matmul_zonotope(Rz) + Z_rho

        nominal_x = C @ z_nom_seq[:, i + 1]
        interval = R.interval
        Ex_interval.append(Interval(interval.left_limit - nominal_x,
                                    interval.right_limit - nominal_x))

    return Ex_interval


def nominal_lifted_rollout(bundle, z0, ep_center, u_nom_seq, Nfut):
    """Propagate the nominal lifted trajectory z_nom(i), i=0..Nfut (for the tube
    operating point). u_nom_seq has length Nfut; ep held constant (paper assumption)."""
    A, B, H = bundle['A'], bundle['B'], bundle['H']
    n_z = bundle['n_z']
    z_nom = np.zeros((n_z, Nfut + 1))
    z_nom[:, 0] = z0
    Bcol, Hcol = B.reshape(n_z), H.reshape(n_z)
    for i in range(Nfut):
        z_nom[:, i + 1] = A @ z_nom[:, i] + Bcol * u_nom_seq[i] + Hcol * ep_center
    return z_nom


# =============================================================================
#  RNDDPC convex QP  (executable problem, eq 50)
# =============================================================================
class RNDDPC_Solver:
    """Per-subplatoon convex QP. Identical structure to the parent decentralized
    Koopman-MPC, with the tube tightening supplied by `reachable_tube`."""

    def __init__(self, A, B, H, C, n_z, s_dim, Nfut, scenario):
        self.A, self.B, self.H, self.C = A, B, H, C
        self.n_z, self.s_dim, self.Nfut = n_z, s_dim, Nfut

        self.z_t = cp.Parameter((n_z,), name='z_t')
        self.u_prev = cp.Parameter((1,), name='u_prev')
        self.x_ref = cp.Parameter((s_dim, Nfut + 1), name='x_ref')
        self.ep = cp.Parameter((1, Nfut + 1), name='ep')
        self.Spa_lb = cp.Parameter((Nfut + 1,)); self.Spa_ub = cp.Parameter((Nfut + 1,))
        self.Vel_lb = cp.Parameter((Nfut + 1,)); self.Vel_ub = cp.Parameter((Nfut + 1,))

        self.z = cp.Variable((n_z, Nfut + 1))
        self.u = cp.Variable((1, Nfut))
        self.x = cp.Variable((s_dim, Nfut + 1))
        self.sl_spa_lb = cp.Variable((Nfut,), nonneg=True)
        self.sl_spa_ub = cp.Variable((Nfut,), nonneg=True)
        self.sl_vel_lb = cp.Variable((Nfut,), nonneg=True)
        self.sl_vel_ub = cp.Variable((Nfut,), nonneg=True)
        self.sl_term = cp.Variable(nonneg=True)
        self.sl_term_spa = cp.Variable(nonneg=True)
        self.sl_spa_ref = cp.Variable((s_dim // 2, Nfut), nonneg=True)

        if scenario in ('tracking', 'braking', 'cutin'):
            Q = np.diag([rnddpc_q_spa, rnddpc_q_vel] * (s_dim // 2))
            R = rnddpc_r * np.eye(1)
            Rd = rnddpc_rd * np.eye(1)
        else:
            Q = np.diag([1, 2] * (s_dim // 2))
            R = 12.5 * np.eye(1)
            Rd = 0.0 * np.eye(1)

        obj = 0
        cons = [self.z[:, 0] == self.z_t]
        for i in range(Nfut):
            cons += [
                self.z[:, i + 1] == A @ self.z[:, i] + B @ self.u[:, i] + H @ self.ep[:, i],
                self.x[:, i + 1] == C @ self.z[:, i + 1],
                self.x[0, i + 1] <= self.Spa_ub[i + 1] + self.sl_spa_ub[i],
                self.x[0, i + 1] >= self.Spa_lb[i + 1] - self.sl_spa_lb[i],
                self.x[1, i + 1] <= self.Vel_ub[i + 1] + self.sl_vel_ub[i],
                self.x[1, i + 1] >= self.Vel_lb[i + 1] - self.sl_vel_lb[i],
                self.sl_spa_ref[:, i] >= self.x_ref[0::2, i + 1] - self.x[0::2, i + 1],
                self.u[0, i] >= AccMin, self.u[0, i] <= AccMax,
            ]
            obj += (
                cp.quad_form(self.x[:, i + 1] - self.x_ref[:, i + 1], Q) +
                cp.quad_form(self.u[:, i], R) +
                W_SPACING_LOW * cp.sum_squares(self.sl_spa_ref[:, i])
            )
        obj += cp.quad_form(self.u[:, 0] - self.u_prev, Rd)
        for i in range(1, Nfut):
            obj += cp.quad_form(self.u[:, i] - self.u[:, i - 1], Rd)

        slack_all = (
            cp.sum(self.sl_spa_lb) + cp.sum(self.sl_spa_ub) +
            cp.sum(self.sl_vel_lb) + cp.sum(self.sl_vel_ub) + self.sl_term
        )
        slack_l2 = (
            cp.sum_squares(self.sl_spa_lb) + cp.sum_squares(self.sl_spa_ub) +
            cp.sum_squares(self.sl_vel_lb) + cp.sum_squares(self.sl_vel_ub) +
            cp.square(self.sl_term)
        )
        obj += W_SLACK * slack_all + W_SLACK_L2 * slack_l2
        obj += W_TERM_SPACING * cp.square(self.sl_term_spa)
        cons += [
            self.x[1, Nfut] <= self.x_ref[1, Nfut] + self.sl_term,
            self.x[1, Nfut] >= self.x_ref[1, Nfut] - self.sl_term,
            self.x[0, Nfut] >= self.x_ref[0, Nfut] - self.sl_term_spa,
        ]

        self.x0 = cp.Parameter((s_dim,), name='x0')
        cons += [self.x[:, 0] == self.x0]
        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def _slack_report(self):
        return {
            'spa_lb': np.asarray(self.sl_spa_lb.value).copy(),
            'spa_ub': np.asarray(self.sl_spa_ub.value).copy(),
            'vel_lb': np.asarray(self.sl_vel_lb.value).copy(),
            'vel_ub': np.asarray(self.sl_vel_ub.value).copy(),
            'term': float(self.sl_term.value),
            'term_spa': float(self.sl_term_spa.value),
        }

    def solve(self, z_t_val, x0_val, x_ref_val, ep_val, Ex_interval, u_prev_val):
        self.z_t.value = z_t_val
        self.u_prev.value = np.asarray(u_prev_val, dtype=float).reshape(1)
        self.x0.value = x0_val
        self.x_ref.value = x_ref_val
        self.ep.value = ep_val

        N = self.Nfut
        Spa_lb = np.full(N + 1, SpaMin); Spa_ub = np.full(N + 1, SpaMax)
        Vel_lb = np.full(N + 1, VelMin); Vel_ub = np.full(N + 1, VelMax)
        for i in range(1, N + 1):
            iv = Ex_interval[i - 1]
            Spa_lb[i] = SpaMin - iv.left_limit[0]
            Spa_ub[i] = SpaMax - iv.right_limit[0]
            Vel_lb[i] = VelMin - iv.left_limit[1]
            Vel_ub[i] = VelMax - iv.right_limit[1]
        if not (np.all(np.isfinite(Spa_lb)) and np.all(np.isfinite(Spa_ub)) and
                np.all(np.isfinite(Vel_lb)) and np.all(np.isfinite(Vel_ub))):
            raise ValueError("RNDDPC tightened bounds contain non-finite values")
        self.Spa_lb.value = Spa_lb; self.Spa_ub.value = Spa_ub
        self.Vel_lb.value = Vel_lb; self.Vel_ub.value = Vel_ub

        t0 = time.time()
        self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=20000)
        compute_time = time.time() - t0
        solver_time = self.prob.solver_stats.solve_time
        solver_time = float(solver_time) if solver_time is not None else np.nan
        if self.prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or self.u.value is None:
            raise RuntimeError(f"RNDDPC QP failed: {self.prob.status}")
        return self.x.value, self.z.value, self.u.value, compute_time, solver_time, self._slack_report()


class RNDDPC_PaperConicSolver:
    """Executable RNDDPC (50) with Zu=<u(i|k),0> inside the optimization.

    The matrix-zonotope reachable sets are represented by an interval/box hull at
    each horizon step. This keeps the problem convex: interval half-widths are
    convex functions of the decision sequence through norm-1 terms, so state-set
    containment constraints become conic constraints. Use a conic solver such as
    CLARABEL/SCS/MOSEK rather than OSQP.
    """

    def __init__(self, bundle, Nfut, scenario):
        A, B, H, C = bundle['A'], bundle['B'], bundle['H'], bundle['C']
        D_mz = bundle['D_mz_generators']
        Zp_mz = bundle['Zp_mz_generators']
        sigma_max, rho_max = bundle['sigma_max'], bundle['rho_max']
        n_z, s_dim = bundle['n_z'], bundle['s_dim']
        self.n_z, self.s_dim, self.Nfut = n_z, s_dim, Nfut

        self.z_t = cp.Parameter((n_z,), name='z_t')
        self.x0 = cp.Parameter((s_dim,), name='x0')
        self.u_prev = cp.Parameter((1,), name='u_prev')
        self.x_ref = cp.Parameter((s_dim, Nfut + 1), name='x_ref')
        self.ep = cp.Parameter((1, Nfut + 1), name='ep')

        self.u = cp.Variable((1, Nfut))
        self.z = cp.Variable((n_z, Nfut + 1))
        self.x = cp.Variable((s_dim, Nfut + 1))
        self.hz = cp.Variable((n_z, Nfut + 1), nonneg=True)
        self.hx = cp.Variable((s_dim, Nfut), nonneg=True)
        self.eta_z = cp.Variable((Nfut,), nonneg=True)
        self.eta_x = cp.Variable((Nfut,), nonneg=True)
        self.sl_spa_lb = cp.Variable((Nfut,), nonneg=True)
        self.sl_spa_ub = cp.Variable((Nfut,), nonneg=True)
        self.sl_vel_lb = cp.Variable((Nfut,), nonneg=True)
        self.sl_vel_ub = cp.Variable((Nfut,), nonneg=True)
        self.sl_term = cp.Variable(nonneg=True)
        self.sl_term_spa = cp.Variable(nonneg=True)
        self.sl_spa_ref = cp.Variable((s_dim // 2, Nfut), nonneg=True)

        if scenario in ('tracking', 'braking', 'cutin'):
            Q = np.diag([rnddpc_q_spa, rnddpc_q_vel] * (s_dim // 2))
            R = rnddpc_r * np.eye(1)
            Rd = rnddpc_rd * np.eye(1)
        else:
            Q = np.diag([1, 2] * (s_dim // 2))
            R = 12.5 * np.eye(1)
            Rd = 0.0 * np.eye(1)

        abs_A = np.abs(A)
        abs_H = np.abs(H.reshape(n_z))
        abs_C = np.abs(C)
        D_colsum = np.sum(np.abs(D_mz), axis=0)
        Zp_colsum = np.sum(np.abs(Zp_mz), axis=0)

        Bcol = B.reshape(n_z)
        Hcol = H.reshape(n_z)

        cons = [
            self.z[:, 0] == self.z_t,
            self.x[:, 0] == self.x0,
            self.hz[:, 0] == 0,
        ]
        obj = 0

        for i in range(Nfut):
            u_i = self.u[0, i]
            ep_i = self.ep[0, i]
            p_center = cp.hstack([self.z[:, i], cp.reshape(u_i, (1,), order='C'), cp.reshape(ep_i, (1,), order='C')])
            coeff_z_radius = D_colsum[:n_z] @ self.hz[:, i] + D_colsum[n_z + 1] * EPS_MAX
            coeff_x_radius = Zp_colsum @ self.hz[:, i + 1]

            cons += [
                self.eta_z[i] >= cp.norm1(D_mz @ p_center),
                self.z[:, i + 1] == A @ self.z[:, i] + Bcol * u_i + Hcol * ep_i,
                self.hz[:, i + 1] == (
                    abs_A @ self.hz[:, i]
                    + abs_H * EPS_MAX
                    + sigma_max * (self.eta_z[i] + coeff_z_radius)
                    + sigma_max
                ),
                self.eta_x[i] >= cp.norm1(Zp_mz @ self.z[:, i + 1]),
                self.x[:, i + 1] == C @ self.z[:, i + 1],
                self.hx[:, i] == (
                    abs_C @ self.hz[:, i + 1]
                    + rho_max * (self.eta_x[i] + coeff_x_radius)
                    + rho_max
                ),
                self.x[0, i + 1] - self.hx[0, i] >= SpaMin - self.sl_spa_lb[i],
                self.x[0, i + 1] + self.hx[0, i] <= SpaMax + self.sl_spa_ub[i],
                self.x[1, i + 1] - self.hx[1, i] >= VelMin - self.sl_vel_lb[i],
                self.x[1, i + 1] + self.hx[1, i] <= VelMax + self.sl_vel_ub[i],
                self.sl_spa_ref[:, i] >= self.x_ref[0::2, i + 1] - self.x[0::2, i + 1],
                u_i >= AccMin,
                u_i <= AccMax,
            ]
            obj += (
                cp.quad_form(self.x[:, i + 1] - self.x_ref[:, i + 1], Q) +
                cp.quad_form(self.u[:, i], R) +
                W_SPACING_LOW * cp.sum_squares(self.sl_spa_ref[:, i])
            )

        obj += cp.quad_form(self.u[:, 0] - self.u_prev, Rd)
        for i in range(1, Nfut):
            obj += cp.quad_form(self.u[:, i] - self.u[:, i - 1], Rd)

        slack_all = (
            cp.sum(self.sl_spa_lb) + cp.sum(self.sl_spa_ub) +
            cp.sum(self.sl_vel_lb) + cp.sum(self.sl_vel_ub) + self.sl_term
        )
        slack_l2 = (
            cp.sum_squares(self.sl_spa_lb) + cp.sum_squares(self.sl_spa_ub) +
            cp.sum_squares(self.sl_vel_lb) + cp.sum_squares(self.sl_vel_ub) +
            cp.square(self.sl_term)
        )
        obj += W_SLACK * slack_all + W_SLACK_L2 * slack_l2
        obj += W_TERM_SPACING * cp.square(self.sl_term_spa)
        # Epigraph tie-breaker: keeps the norm/interval auxiliary variables at
        # their tight values without changing the control objective materially.
        obj += 1e-8 * (cp.sum(self.eta_z) + cp.sum(self.eta_x) + cp.sum(self.hz) + cp.sum(self.hx))
        cons += [
            self.x[1, Nfut] <= self.x_ref[1, Nfut] + self.sl_term,
            self.x[1, Nfut] >= self.x_ref[1, Nfut] - self.sl_term,
            self.x[0, Nfut] >= self.x_ref[0, Nfut] - self.sl_term_spa,
        ]

        self.prob = cp.Problem(cp.Minimize(obj), cons)
        self.solver_name = CONIC_SOLVER

    def _slack_report(self):
        return {
            'spa_lb': np.asarray(self.sl_spa_lb.value).copy(),
            'spa_ub': np.asarray(self.sl_spa_ub.value).copy(),
            'vel_lb': np.asarray(self.sl_vel_lb.value).copy(),
            'vel_ub': np.asarray(self.sl_vel_ub.value).copy(),
            'term': float(self.sl_term.value),
            'term_spa': float(self.sl_term_spa.value),
        }

    def _tube_report(self):
        if self.hx.value is None:
            return np.zeros((self.s_dim, 0))
        return np.asarray(self.hx.value, dtype=float).copy()

    def solve(self, z_t_val, x0_val, x_ref_val, ep_val, u_prev_val):
        self.z_t.value = z_t_val
        self.x0.value = x0_val
        self.x_ref.value = x_ref_val
        self.ep.value = ep_val
        self.u_prev.value = np.asarray(u_prev_val, dtype=float).reshape(1)

        t0 = time.time()
        self.prob.solve(solver=self.solver_name, warm_start=True, verbose=False)
        compute_time = time.time() - t0
        solver_time = self.prob.solver_stats.solve_time
        solver_time = float(solver_time) if solver_time is not None else np.nan
        if self.prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or self.u.value is None:
            raise RuntimeError(f"RNDDPC paper-conic problem failed: {self.prob.status}")
        return self.x.value, self.z.value, self.u.value, compute_time, solver_time, self._slack_report(), self._tube_report()


if __name__ == '__main__':
    # Smoke test of the offline pipeline using the same CSV-backed subplatoon
    # parameters as RNDDPC.py.
    sub_numveh = [1 + nh for nh in subplatoon_spec]
    sub_start = [0] * len(subplatoon_spec)
    for k in range(1, len(subplatoon_spec)):
        sub_start[k] = sub_start[k - 1] + sub_numveh[k - 1]
    param_data = load_vehicle_parameters(sum(sub_numveh))
    for k, nh in enumerate(subplatoon_spec):
        base = sub_start[k]
        nv = sub_numveh[k]
        idm_params = (
            param_data[base:base + nv, 0],
            param_data[base:base + nv, 1],
            param_data[base:base + nv, 2],
            param_data[base:base + nv, 5],
            param_data[base:base + nv, 6],
            param_data[base:base + nv, 7],
        )
        build_rnddpc_bundle(nh, idm_params=idm_params, bundle_label=f"sub{k}")
