"""RNDDPC-HI configuration, history-informed lifting, and offline identification.

The empirical residual quantile defaults to 0.10, without shrinkage. This
adapted baseline uses compressed matrix zonotopes and hard CAV constraints.
Use run_rnddpc.py to run the main results serially.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_IDM_DIR = _HERE.parent.parent
PROJECT_ROOT = str(_IDM_DIR.parent)
sys.path.insert(0, str(_IDM_DIR))
import _heter_helperfuns_torch_IDM_learning_set as helper
helper.project_path = PROJECT_ROOT
from _heter_helperfuns_torch_IDM_learning_set import (
    IDM_dynamics, compute_IDM_steady_s, generate_history_seq, plot,
    compute_tracking_metrics, save_vehicle_data_to_csv, save_metrics,
    save_computation_time, _env_float, _env_int, _env_int_list,
)

scenario = os.environ.get('KMPC_SCENARIO', 'tracking')
subplatoon_spec = [2, 3, 1, 4, 2]
Nfut = _env_int('KMPC_NFUT', 5)
Tstep = 0.12
SpaMin, SpaMax = 10.0, 80.0
VelMin, VelMax = 0.0, 35.0
AccMin, AccMax = -6.0, 4.0
sdes, vdes_stab = 40.0, 25.0
vehicle_csv_path = str(_IDM_DIR / 'vehicle_parameters.csv')
rnddpc_q_spa, rnddpc_q_vel = 2.5, 1.5
rnddpc_r, rnddpc_rd = 20.0, 30.0
Q_COVER = _env_float('RNDDPC_QCOVER', 0.10)
output_dir = str(_HERE / 'results' / 'main_results' / scenario)
EPS_MAX = 0.2
RCOND_PINV = 0.001
MZ_KEEP_GENERATORS = 999
CONIC_SOLVER = 'CLARABEL'
EP_EXTRAPOLATE = True
DATA_ROLLOUTS, DATA_ROLLOUT_LEN = 30, 120
DATA_NOISE, DATA_SEED = 0.03, 0
DATA_VBASE_MIN, DATA_VBASE_MAX = 10.0, 30.0
DATA_HEAD_MIN, DATA_HEAD_MAX = 10.0, 30.0
DATA_HEAD_STD, DATA_PE_STD = 0.8, 1.5
WITH_HISTORY_MODEL_PATHS = {1: helper.HDV_1_MODEL_PATH, 2: helper.HDV_2_MODEL_PATH,
                          3: helper.HDV_3_MODEL_PATH, 4: helper.HDV_4_MODEL_PATH}
class KoopmanLiftWrapper:
    """NumPy interface to the pretrained history-informed encoder."""

    def __init__(self, model, args, num_HDV, use_history, model_ref, device='cpu'):
        self.model = model
        self.args = args
        self.num_HDV = num_HDV
        self.use_history = use_history
        self.model_ref = model_ref
        self.device = device
        self.s_dim = args['s_dim']
        self.lift_dim = args['lift_dim']
        self.n_z = self.s_dim + self.lift_dim
        self.Npst = args['Npst']
        self.slice_dim = args['HV_hist_slice_dim']

    @torch.no_grad()
    def lift(self, x_np, x_history_np=None):
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
    args = vars(helper.get_args_local(False, num_HDV_override=num_HDV))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rel_path = WITH_HISTORY_MODEL_PATHS[num_HDV]
    model_path = os.path.join(PROJECT_ROOT, 'Deep_Koop_v1', 'model_highd_multistep', rel_path)
    model = helper.load_trained_model(args, model_path, device)
    print(f"[RNDDPC] loaded with-history Koopman encoder num_HDV={num_HDV}: {rel_path} s_dim={args['s_dim']}, n_z={args['s_dim'] + args['lift_dim']}")
    return KoopmanLiftWrapper(model, args, num_HDV, True, rel_path, device)

def load_vehicle_parameters(required_rows=None):
    param_data = np.genfromtxt(vehicle_csv_path, delimiter=',', skip_header=1)
    param_data = np.atleast_2d(param_data)
    if required_rows is not None and param_data.shape[0] < required_rows:
        raise ValueError(f'{vehicle_csv_path}: {param_data.shape[0]} rows < {required_rows} vehicles')
    if param_data.shape[1] < 8:
        raise ValueError(f'{vehicle_csv_path}: need columns Tgap,v0_IDM,veh_len,Tgap_rate,v0_rate,a_IDM,b_IDM,s0_IDM')
    return param_data

def simulate_PE_data(model, num_HDV, idm_params, n_rollouts=DATA_ROLLOUTS, rollout_len=DATA_ROLLOUT_LEN, noise=DATA_NOISE, seed=DATA_SEED):
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
    (X_, Xp, U_, E_, Z_, Zp) = ([], [], [], [], [], [])
    for _ in range(n_rollouts):
        (Tgap, v0, vlen, a_idm, b_idm, s0_idm) = [np.asarray(x, dtype=float).copy() for x in idm_params]
        if len(Tgap) != num_veh:
            raise ValueError(f'idm_params length {len(Tgap)} does not match 1+num_HDV={num_veh}')
        v_base = rng.uniform(DATA_VBASE_MIN, DATA_VBASE_MAX)
        head_v = np.zeros(sim_len + 1)
        head_v[0] = v_base
        for k in range(sim_len):
            head_v[k + 1] = np.clip(head_v[k] + rng.normal(0, DATA_HEAD_STD), DATA_HEAD_MIN, DATA_HEAD_MAX)
        du = np.zeros(sim_len)
        for k in range(1, sim_len):
            du[k] = 0.7 * du[k - 1] + rng.normal(0, DATA_PE_STD)
        du = np.clip(du, AccMin, AccMax)
        S = np.zeros((sim_len + 1, num_veh + 1, 3))
        S[:, 0, 1] = head_v
        S[0, 0, 0] = 0.0
        for j in range(num_veh):
            s_eq = compute_IDM_steady_s(v_base, v_base, v0[j], Tgap[j], a_idm[j], b_idm[j], s0_idm[j]) + vlen[j]
            S[0, j + 1, 0] = S[0, j, 0] - s_eq
            S[0, j + 1, 1] = v_base
        for k in range(sim_len):
            acel = IDM_dynamics(S[k, :, :], Tgap, v0, vlen, a_idm, b_idm, s0_idm)
            S[k, 1:, 2] = acel
            S[k, 1, 2] = float(np.clip(acel[0] + du[k], AccMin, AccMax))
            S[k + 1, :, 0] = S[k, :, 0] + Tstep * S[k, :, 1]
            S[k + 1, 1:, 1] = S[k, 1:, 1] + Tstep * S[k, 1:, 2]
        xk = np.zeros((rollout_len + 1, s_dim))
        xhist = np.zeros((rollout_len + 1, Npst_local, slice_dim, num_HDV)) if model.use_history else None
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
            zk = model.lift(xk_noisy, xhist_noisy)
        else:
            zk = model.lift(xk_noisy)
        uk = S[Npst_local:Npst_local + rollout_len, 1, 2]
        epk = S[Npst_local:Npst_local + rollout_len, 0, 1]
        X_.append(xk_noisy[:-1].T)
        Xp.append(xk_noisy[1:].T)
        Z_.append(zk[:-1].T)
        Zp.append(zk[1:].T)
        U_.append(uk[None, :])
        E_.append(epk[None, :])
    data = {'X_': np.hstack(X_), 'X+': np.hstack(Xp), 'Z_': np.hstack(Z_), 'Z+': np.hstack(Zp), 'U_': np.hstack(U_), 'E_': np.hstack(E_)}
    print(f"[RNDDPC] collected PE data num_HDV={num_HDV}: T={data['Z_'].shape[1]}, n_z={data['Z_'].shape[0]}")
    return data

def fit_koopman_ls(data):
    """LS solution = centers of the matrix zonotopes (eq 39/40):
       [A B H] = Z+ [Z-;U-;E-]^†,   C = X+ [Z+]^†.
       Also returns the pseudo-inverses needed online for the matrix-zono tube.
    """
    (Z_, Zp) = (data['Z_'], data['Z+'])
    (U_, E_) = (data['U_'], data['E_'])
    Xp = data['X+']
    n_z = Z_.shape[0]
    D = np.vstack([Z_, U_, E_])
    D_pinv = np.linalg.pinv(D, rcond=RCOND_PINV)
    M = Zp @ D_pinv
    A = M[:, :n_z]
    B = M[:, n_z:n_z + 1]
    H = M[:, n_z + 1:n_z + 2]
    Zp_pinv = np.linalg.pinv(Zp, rcond=RCOND_PINV)
    C = Xp @ Zp_pinv
    sigma_res = Zp - M @ D
    rho_res = Xp - C @ Zp
    return {'A': A, 'B': B, 'H': H, 'C': C, 'D_pinv': D_pinv, 'Zp_pinv': Zp_pinv, 'sigma_res': sigma_res, 'rho_res': rho_res, 'n_z': n_z, 's_dim': C.shape[0]}

def error_bounds(fit, q_cover=None):
    """Coordinate-wise absolute-residual quantiles; floor 1e-6, no extra scaling."""
    q_cover = Q_COVER if q_cover is None else q_cover
    sigma_max = np.quantile(np.abs(fit['sigma_res']), q_cover, axis=1)
    rho_max = np.quantile(np.abs(fit['rho_res']), q_cover, axis=1)
    sigma_max = np.maximum(sigma_max, 1e-06)
    rho_max = np.maximum(rho_max, 1e-06)
    return (sigma_max, rho_max)

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
        (_, _, vh) = np.linalg.svd(P, full_matrices=False)
        basis = vh[:keep, :]
        coeff = P @ basis.T
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

def build_rnddpc_bundle(num_HDV, idm_params, bundle_label=None):
    """Fit current matrix centers/bounds from the frozen, reproducible PE data."""
    if bundle_label is None:
        raise ValueError('A subsystem label is required for the frozen data cache')
    directory = _HERE / 'offline_data'
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    if hashlib.sha256(Path(vehicle_csv_path).read_bytes()).hexdigest() != metadata['vehicle_csv_sha256']:
        raise ValueError('Vehicle parameters differ from the retained offline calibration.')
    index = int(bundle_label.removeprefix('sub'))
    if metadata['subsystem_sizes'][index] != num_HDV:
        raise ValueError('Subsystem size does not match its offline data')
    cache = directory / f'data_sub{index}.npz'
    if hashlib.sha256(cache.read_bytes()).hexdigest() != metadata['data_sha256'][cache.name]:
        raise ValueError(f'Offline data checksum mismatch: {cache}')
    with np.load(cache) as archived:
        data = {k: np.asarray(archived['train_' + k], dtype=np.float64)
                for k in ['Z_', 'Z+', 'X_', 'X+', 'U_', 'E_']}
    model = load_rnddpc_lift_model(num_HDV)
    fitted = fit_koopman_ls(data)
    fitted['sigma_max'], fitted['rho_max'] = error_bounds(fitted)
    fitted['D_mz_generators'] = reduce_pinv_generators(fitted['D_pinv'])
    fitted['Zp_mz_generators'] = reduce_pinv_generators(fitted['Zp_pinv'])
    fitted.update(model=model, model_ref=model.model_ref, num_HDV=num_HDV,
                  n_z=fitted['A'].shape[0], s_dim=fitted['C'].shape[0],
                  use_history_koopman=True,
                  mz_generator_counts={key: fitted[key].shape[0] for key in ['D_mz_generators', 'Zp_mz_generators']})
    return fitted
