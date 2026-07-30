"""
Plot Nfut-step prediction comparisons for each model on one trajectory from the dataset (default: the first trajectory).

Configuration: 1 CAV + 2 HDV (num_HDV=2), dataset highd_traj_*_2HDV_u4f.pkl。

Compared models:
    - Ours               : Direct multi-step predictor (_5)        2026_06_29_00_39_02
    - Calibrated IDM     : calibrated IDM rule-based model                    (identified_IDM)
    - Seq-to-Seq LSTM    : Seq2Seq LSTM                           2026_06_02_10_14_08
    - Deep Koopman       : DeepEDMD without LSTM style (NoLSTM, _3)    2026_06_01_14_54_33
    - Deep Linear Model  : linear model in the original state (NoKoopman, _4)          2026_06_01_15_12_58
    - One-Step Predictor : DeepEDMD + LSTM style + Koopman (_2)   2026_06_14_11_48_32

Starting from current time t0, encode the initial state using the Npst-step history before t0 and predict only
Multi-step predictor fixed Nfut steps, overlaid with the true trajectory in the same figure.

Run (using the python310 conda environment):
    conda run -n python310 python plot_model_comparison_nfut.py
    conda run -n python310 python plot_model_comparison_nfut.py --traj 0 --t0 51
"""

import os
import sys
import argparse
import importlib

# Avoid duplicate OpenMP runtime aborts on some Windows/conda stacks.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

# ---------- Global plotting style: Times New Roman ----------
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'axes.linewidth': 0.8,
    'font.size': 11,
})


# --------------------------------------------------------------------------
# Path setup: add three subdirectories to sys.path to import each model module
# --------------------------------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.path.dirname(CUR_DIR)
HETER_DIR = os.path.join(CUR_DIR, 'Heter_HDV_HighD')
LSTM_DIR = os.path.join(CUR_DIR, 'SeqtoSeq_LSTM')
IDM_DIR = os.path.join(CUR_DIR, 'identified_IDM')
for _d in (HETER_DIR, LSTM_DIR, IDM_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

MODEL_COLLECTION = os.path.join(PROJECT_PATH, 'Deep_Koop_v1', 'model_highd')
MULTISTEP_MODEL_COLLECTION = os.path.join(PROJECT_PATH, 'Deep_Koop_v1', 'model_highd_multistep')

# Date folder for each model; best checkpoint is read from test_metrics.txt in that folder
MODEL_DIRS = {
    'Ours': '2026_06_29_00_39_02',
    'Seq-to-Seq LSTM': '2026_06_02_10_14_08',
    'Deep Koopman': '2026_06_01_14_54_33',
    'Deep Linear Model': '2026_06_01_15_12_58',
    'One-Step Predictor': '2026_06_14_11_48_32',
}

# Plot configuration: colors / line styles / markers; line style plus marker keeps black-and-white distinguishability
PLOT_STYLE = {
    'True':         dict(color='#222222', linestyle='-',  linewidth=2.6, marker=None,
                         zorder=10),
    'Calibrated IDM': dict(color='#0066CC', linestyle=(0, (4, 2)), linewidth=1.6, marker='D',
                         markersize=3.6, markevery=6, markerfacecolor='white',
                         markeredgewidth=1.1),
    'Seq-to-Seq LSTM': dict(color='#E35A58', linestyle=(0, (1, 1)), linewidth=1.6, marker='v',
                         markersize=4.5, markevery=6, markerfacecolor='white',
                         markeredgewidth=1.1),
    'Deep Koopman': dict(color='#009999', linestyle='--', linewidth=1.6, marker='s',
                         markersize=4, markevery=6, markerfacecolor='white',
                         markeredgewidth=1.1),
    'Deep Linear Model': dict(color='#7A42F4', linestyle='-.', linewidth=1.6, marker='^',
                         markersize=4.5, markevery=6, markerfacecolor='white',
                         markeredgewidth=1.1),
    'One-Step Predictor': dict(color='#FF8000', linestyle=':', linewidth=1.8, marker='P',
                         markersize=4.8, markevery=5, markerfacecolor='white',
                         markeredgewidth=1.2, zorder=8),
    'Ours':         dict(color='#2FA84F', linestyle='-',  linewidth=2.1, marker='o',
                         markersize=4.5, markevery=6, markerfacecolor='white',
                         markeredgewidth=1.2, zorder=9),
}

# Legend display order
LEGEND_ORDER = [
    'True',
    'Calibrated IDM',
    'Seq-to-Seq LSTM',
    'Deep Koopman',
    'Deep Linear Model',
    'One-Step Predictor',
    'Ours',
]

DEVICE = torch.device('cpu')


# --------------------------------------------------------------------------
# Read the best checkpoint under each model folder (local path)
# --------------------------------------------------------------------------
def resolve_best_checkpoint(date_folder, collection=MODEL_COLLECTION):
    """Parse the best_model_path filename from <date_folder>/test_metrics.txt and build the local path."""
    folder = os.path.join(collection, date_folder)
    metrics_path = os.path.join(folder, 'test_metrics.txt')
    ckpt_name = None
    with open(metrics_path, 'r') as f:
        for line in f:
            if line.startswith('best_model_path:'):
                raw = line.split(':', 1)[1].strip()
                ckpt_name = os.path.basename(raw.replace('\\', '/'))
                break
    if ckpt_name is None or not ckpt_name.endswith('.weights.h5'):
        raise RuntimeError(f'Unable to parse {metrics_path} best checkpoint from')
    ckpt_path = os.path.join(folder, 'past_models', ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'checkpoint does not exist: {ckpt_path}')
    return ckpt_path


# --------------------------------------------------------------------------
# Load the dataset (compact pkl) and extract X / U / EP for the selected trajectory
# --------------------------------------------------------------------------
def load_trajectory(helper_mod, num_HDV, traj_index):
    data_path = helper_mod.get_pkl_path(num_HDV)
    data_dict = helper_mod.read_pkl_as_dict(data_path)
    X = np.asarray(data_dict['X'][traj_index], dtype=np.float32)   # (traj_len, s_dim)
    U = np.asarray(data_dict['U'][traj_index], dtype=np.float32)   # (traj_len or -1, u_dim)
    EP = np.asarray(data_dict['EP'][traj_index], dtype=np.float32)
    if U.ndim == 1:
        U = U[:, None]
    if EP.ndim == 1:
        EP = EP[:, None]
    print(f'Trajectory #{traj_index}: X={X.shape}, U={U.shape}, EP={EP.shape}, dataset={os.path.basename(data_path)}')
    return X, U, EP


# --------------------------------------------------------------------------
# Open-loop multistep predictions for each model; returns (H, s_dim), corresponding to X[t0+1 : t0+1+H]
# --------------------------------------------------------------------------
def _build_hist(X, tc, Npst, slice_dim, num_HDV):
    """Build the history required by DeepEDMD encode: (1, Npst, slice_dim, num_HDV)。"""
    hist_raw = X[tc - Npst:tc]  # (Npst, s_dim)
    hist = np.zeros((Npst, slice_dim, num_HDV), dtype=np.float32)
    for hdv_idx in range(num_HDV):
        c0 = 2 * hdv_idx + 1
        hist[:, :, hdv_idx] = hist_raw[:, c0:c0 + slice_dim]
    return torch.tensor(hist, dtype=torch.float32, device=DEVICE).unsqueeze(0)


def koopman_rollout(model, X, U, EP, t0, H, Npst, slice_dim, num_HDV, mode='openloop'):
    """Generic prediction for the DeepEDMD series (_2 one-step / _3 Deep Koopman / _4 Deep Linear).

    g_{k+1} = g_k A + u_k B + ep_k H;  y = decode(g)
    - openloop: t0 encode once to obtain g0, then propagate only in the lifted space to the end using its own predicted g。
    - onestep : re-encode with the **true state** at every time and predict only one step ahead (one-step prediction).
    Deep Koopman / Deep Linear encode ignores history, but it is still passed uniformly.
    """
    with torch.no_grad():
        preds = []
        if mode == 'openloop':
            x0 = torch.tensor(X[t0], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            g = model.encode(x0, _build_hist(X, t0, Npst, slice_dim, num_HDV), training=False)
            for k in range(H):
                u_k = torch.tensor(U[t0 + k], dtype=torch.float32, device=DEVICE).reshape(1, -1)
                ep_k = torch.tensor(EP[t0 + k], dtype=torch.float32, device=DEVICE).reshape(1, -1)
                g = g @ model.A + u_k @ model.B + ep_k @ model.H
                preds.append(model.decode(g).squeeze(0).cpu().numpy())
        else:  # onestep
            for k in range(H):
                tc = t0 + k                      # current true time
                x_c = torch.tensor(X[tc], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                g = model.encode(x_c, _build_hist(X, tc, Npst, slice_dim, num_HDV), training=False)
                u_k = torch.tensor(U[tc], dtype=torch.float32, device=DEVICE).reshape(1, -1)
                ep_k = torch.tensor(EP[tc], dtype=torch.float32, device=DEVICE).reshape(1, -1)
                g1 = g @ model.A + u_k @ model.B + ep_k @ model.H
                preds.append(model.decode(g1).squeeze(0).cpu().numpy())
    return np.stack(preds, axis=0)


def multistep_rollout(model, X, U, EP, t0, H, Npst, slice_dim, num_HDV):
    """_5 direct multi-step predictor.

    g(t+k+1) = g0 @ G0[k] + sum_{j=0..k} u(t+j) @ Gu[k][j]
             + sum_{j=0..k} ep(t+j) @ Ge[k][j]
    """
    if H > model.Nfut:
        raise ValueError(f'H={H} exceeds model.Nfut={model.Nfut}')
    with torch.no_grad():
        x0 = torch.tensor(X[t0], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        g0 = model.encode(x0, _build_hist(X, t0, Npst, slice_dim, num_HDV), training=False)
        preds = []
        for step in range(H):
            g_step = g0 @ model.G0[step]
            for lag in range(step + 1):
                u_lag = torch.tensor(U[t0 + lag], dtype=torch.float32, device=DEVICE).reshape(1, -1)
                ep_lag = torch.tensor(EP[t0 + lag], dtype=torch.float32, device=DEVICE).reshape(1, -1)
                g_step = g_step + u_lag @ model.Gu[step][lag] + ep_lag @ model.Ge[step][lag]
            preds.append(model.decode(g_step).squeeze(0).cpu().numpy())
    return np.stack(preds, axis=0)


def lstm_rollout(model, X, U, EP, t0, H, Npst, mode='openloop'):
    """Seq2Seq LSTM prediction. x_hist uses Npst steps ending at the current state.

    - openloop: encode once, then autoregressively decode H steps, feeding each step its own prediction.
    - onestep : encode each step using true history ending at the true current state and decode only one step.
    """
    with torch.no_grad():
        preds = []
        if mode == 'openloop':
            x_hist_t = torch.tensor(X[t0 - Npst + 1:t0 + 1], dtype=torch.float32,
                                    device=DEVICE).unsqueeze(0)  # (1, Npst, s_dim)
            h_T, c_T = model.encode(x_hist_t)
            h, c = h_T.unsqueeze(0), c_T.unsqueeze(0)
            prev_x = x_hist_t[:, -1, :]
            for k in range(H):
                u_k = torch.tensor(U[t0 + k], dtype=torch.float32, device=DEVICE).reshape(1, -1)
                ep_k = torch.tensor(EP[t0 + k], dtype=torch.float32, device=DEVICE).reshape(1, -1)
                dec_in = torch.cat([prev_x, u_k, ep_k], dim=-1).unsqueeze(1)
                out, (h, c) = model.LSTM_decoder(dec_in, (h, c))
                x_pred = model.fc(out).squeeze(1)
                preds.append(x_pred.squeeze(0).cpu().numpy())
                prev_x = x_pred
        else:  # onestep
            for k in range(H):
                tc = t0 + k
                x_hist_t = torch.tensor(X[tc - Npst + 1:tc + 1], dtype=torch.float32,
                                        device=DEVICE).unsqueeze(0)
                h_T, c_T = model.encode(x_hist_t)
                u_k = torch.tensor(U[tc], dtype=torch.float32, device=DEVICE).reshape(1, -1)
                ep_k = torch.tensor(EP[tc], dtype=torch.float32, device=DEVICE).reshape(1, -1)
                dec_in = torch.cat([x_hist_t[:, -1, :], u_k, ep_k], dim=-1).unsqueeze(1)
                out, _ = model.LSTM_decoder(dec_in, (h_T.unsqueeze(0), c_T.unsqueeze(0)))
                x_pred = model.fc(out).squeeze(1)
                preds.append(x_pred.squeeze(0).cpu().numpy())
    return np.stack(preds, axis=0)


def idm_rollout(idm_mod, X, U, EP, t0, H, dt, mode='openloop'):
    """Calibrated IDM prediction. openloop: roll continuously; onestep: predict one step from the true state each time."""
    params = idm_mod.DEFAULT_IDM_PARAMS
    if mode == 'openloop':
        return idm_mod.predict_window(X[t0], U[t0:t0 + H], EP[t0:t0 + H], params, dt=dt)
    preds = []
    for k in range(H):
        tc = t0 + k
        p = idm_mod.predict_window(X[tc], U[tc:tc + 1], EP[tc:tc + 1], params, dt=dt)[0]
        preds.append(p)
    return np.stack(preds, axis=0)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_comparison(X, t0, H, preds_dict, num_CAV, num_HDV, hist_context, dt, save_path, traj_index, mode,
                    rmse_dict=None):
    num_veh = num_CAV + num_HDV
    veh_names = ['CAV'] + [f'HDV{i}' for i in range(1, num_HDV + 1)]

    # True trajectory display range: x-axis starts from Time Step 50
    x_axis_start = 50
    lo = max(0, min(t0 - hist_context, x_axis_start))
    hi = t0 + H
    t_true = np.arange(lo, hi + 1)
    # openloop: prediction connects from t0 (known current state) to t0+H so the line is continuous;
    # onestep : each prediction is a one-step prediction for t0+1..t0+H, each sourced from the previous true time, without connecting to x0.
    t_pred_open = np.arange(t0, t0 + H + 1)
    t_pred_one = np.arange(t0 + 1, t0 + H + 1)

    fig, axs = plt.subplots(num_veh, 2, figsize=(13, 2.5 * num_veh), sharex=True)
    if num_veh == 1:
        axs = axs[None, :]

    for i in range(num_veh):
        for col, (sub, ylab) in enumerate([(0, 'Spacing (m)'), (1, 'Velocity (m/s)')]):
            ax = axs[i, col]
            comp = 2 * i + sub

            # True trajectory
            ax.plot(t_true, X[lo:hi + 1, comp], label='True', **PLOT_STYLE['True'])

            # Predictions from each model
            x0_val = X[t0, comp]
            for name, pred in preds_dict.items():
                if mode == 'openloop':
                    y = np.concatenate([[x0_val], pred[:, comp]])
                    ax.plot(t_pred_open, y, label=name, **PLOT_STYLE[name])
                else:
                    ax.plot(t_pred_one, pred[:, comp], label=name, **PLOT_STYLE[name])

            # Mark prediction start
            ax.axvline(t0, color='0.6', linestyle=(0, (2, 2)), linewidth=0.9, alpha=0.8)

            ax.set_ylabel(f'{veh_names[i]} {ylab}', fontsize=11)
            ax.set_xlim(x_axis_start, hi)
            ax.tick_params(labelsize=9, direction='in', top=True, right=True)
            ax.grid(True, linestyle=(0, (2, 3)), linewidth=0.5, alpha=0.45)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

    axs[-1, 0].set_xlabel('Time Step', fontsize=11)
    axs[-1, 1].set_xlabel('Time Step', fontsize=11)

    # ---------- RMSE metric table: placed in the CAV spacing / velocity subplots respectively ----------
    if rmse_dict:
        spacing_lines = [f'{"Method":<21}{"Spacing RMSE":>14}']
        velocity_lines = [f'{"Method":<21}{"Velocity RMSE":>14}']
        for name in LEGEND_ORDER:
            if name in rmse_dict:
                rs, rv = rmse_dict[name]
                spacing_lines.append(f'{name:<21}{rs:>14.4f}')
                velocity_lines.append(f'{name:<21}{rv:>14.4f}')
        table_box = dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='0.7', alpha=0.9)
        axs[0, 0].text(
            0.015, 0.97, '\n'.join(spacing_lines),
            transform=axs[0, 0].transAxes, ha='left', va='top',
            family='monospace', fontsize=7.6, bbox=table_box)
        axs[0, 1].text(
            0.015, 0.03, '\n'.join(velocity_lines),
            transform=axs[0, 1].transAxes, ha='left', va='bottom',
            family='monospace', fontsize=7.6, bbox=table_box)

    # ---------- Place the legend at the top center ----------
    handles, labels = axs[0, 0].get_legend_handles_labels()
    hmap = dict(zip(labels, handles))
    ordered = [(hmap[n], n) for n in LEGEND_ORDER if n in hmap]
    fig.legend([h for h, _ in ordered], [n for _, n in ordered],
               loc='upper center', ncol=len(ordered), frameon=False,
               fontsize=12, handlelength=2.6, columnspacing=1.8,
               bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✅ Comparison figure saved: {save_path}')


def plot_comparison_v2(X, t0, H, preds_dict, num_CAV, num_HDV, hist_context, dt, save_path, traj_index, mode):
    """Single-column paper version: compact width, larger print-friendly fonts, no RMSE boxes."""
    num_veh = num_CAV + num_HDV
    veh_names = ['CAV'] + [f'HDV{i}' for i in range(1, num_HDV + 1)]

    x_axis_start = 50
    lo = max(0, min(t0 - hist_context, x_axis_start))
    hi = t0 + H
    t_true = np.arange(lo, hi + 1)
    t_pred_open = np.arange(t0, t0 + H + 1)
    t_pred_one = np.arange(t0 + 1, t0 + H + 1)

    FS_LABEL = 6
    FS_TICK = 5
    FS_LEGEND = 6

    fig, axs = plt.subplots(num_veh, 2, figsize=(3.5, 3.5), sharex=True)
    if num_veh == 1:
        axs = axs[None, :]

    def compact_style(style, is_true=False, is_ours=False):
        out = style.copy()
        out['linewidth'] = 1.45 if is_true else (1.2 if is_ours else 1.05)
        if out.get('marker') is not None:
            out['markersize'] = 2.6 if not is_ours else 2.8
            out['markeredgewidth'] = 0.8
            out['markevery'] = 4
        return out

    for i in range(num_veh):
        for col, (sub, ylab) in enumerate([(0, 'Spacing (m)'), (1, 'Velocity (m/s)')]):
            ax = axs[i, col]
            comp = 2 * i + sub

            ax.plot(t_true, X[lo:hi + 1, comp], label='True',
                    **compact_style(PLOT_STYLE['True'], is_true=True))

            x0_val = X[t0, comp]
            for name, pred in preds_dict.items():
                style = compact_style(PLOT_STYLE[name], is_ours=(name == 'Ours'))
                if mode == 'openloop':
                    y = np.concatenate([[x0_val], pred[:, comp]])
                    ax.plot(t_pred_open, y, label=name, **style)
                else:
                    ax.plot(t_pred_one, pred[:, comp], label=name, **style)

            ax.axvline(t0, color='0.6', linestyle=(0, (2, 2)), linewidth=0.7, alpha=0.8)
            ax.set_ylabel(f'{veh_names[i]} {ylab}', fontsize=FS_LABEL, labelpad=2)
            ax.set_xlim(x_axis_start, hi)
            ax.tick_params(axis='both', which='both', labelsize=FS_TICK, direction='in',
                           top=True, right=True, pad=1.5)
            ax.grid(True, linestyle=(0, (2, 3)), linewidth=0.45, alpha=0.45)
            for spine in ax.spines.values():
                spine.set_linewidth(0.75)

    axs[-1, 0].set_xlabel('Time Step', fontsize=FS_LABEL, labelpad=2)
    axs[-1, 1].set_xlabel('Time Step', fontsize=FS_LABEL, labelpad=2)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    hmap = dict(zip(labels, handles))
    legend_rows = [
        ['True', 'Calibrated IDM', 'Seq-to-Seq LSTM'],
        ['Deep Koopman', 'Deep Linear Model', 'One-Step Predictor'],
        ['Ours'],
    ]
    legend_opts = dict(
        loc='upper center',
        frameon=False,
        prop={'family': 'Times New Roman', 'size': FS_LEGEND},
        handlelength=1.45,
        handletextpad=0.35,
        columnspacing=1.15,
        borderaxespad=0.0,
    )
    for row_labels, y_anchor in zip(legend_rows, [0.985, 0.958, 0.931]):
        row_items = [(hmap[n], n) for n in row_labels if n in hmap]
        fig.legend(
            [h for h, _ in row_items],
            [n for _, n in row_items],
            ncol=len(row_items),
            bbox_to_anchor=(0.5, y_anchor),
            **legend_opts,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.subplots_adjust(wspace=0.34, hspace=0.24)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Single-column v2 comparison figure saved: {save_path}')


# --------------------------------------------------------------------------
def compute_rmse(X, t0, H, pred):
    """Overall RMSE (spacing columns 0::2, velocity columns 1::2)."""
    true = X[t0 + 1:t0 + 1 + H]
    rmse_spa = np.sqrt(np.mean((pred[:, 0::2] - true[:, 0::2]) ** 2))
    rmse_vel = np.sqrt(np.mean((pred[:, 1::2] - true[:, 1::2]) ** 2))
    return rmse_spa, rmse_vel


def main():
    parser = argparse.ArgumentParser(description='Nfut-step prediction comparison for all models on one trajectory')
    parser.add_argument('--traj', type=int, default=0, help='trajectory index (default: first trajectory = 0)')
    parser.add_argument('--t0', type=int, default=51,
                        help='current time where prediction starts (default: 51, the earliest usable window)')
    parser.add_argument('--hist_context', type=int, default=0,
                        help='number of true historical steps shown before the prediction start; no history is shown by default')
    parser.add_argument('--multistep_dir', type=str, default=MODEL_DIRS['Ours'],
                        help='date folder for Ours / multi-step predictor')
    parser.add_argument('--save', type=str, default=None, help='output figure path')
    parser.add_argument('--save_v2', type=str, default=None, help='output figure path for the single-column v2 version')
    args = parser.parse_args()

    num_CAV, num_HDV = 1, 2
    Npst_koop = 50      # history length for the DeepEDMD series
    Npst_lstm = 51      # history length for Seq2Seq LSTM
    slice_dim = 3
    dt = 0.12

    # ---------- Import each model module ----------
    m_multi = importlib.import_module('_5_heter_mainfuns_highd')
    m_ours = importlib.import_module('_2_heter_mainfuns_highd')
    m_koop = importlib.import_module('_3_heter_mainfuns_highd_NoLSTM')
    m_lin = importlib.import_module('_4_heter_mainfuns_highd_NoKoopman')
    m_lstm = importlib.import_module('_1_heter_mainfuns_LSTM')
    m_idm = importlib.import_module('_1_heter_mainfuns_IDM_calibrated')

    # ---------- Data ----------
    X, U, EP = load_trajectory(m_ours.helper, num_HDV, args.traj)
    traj_len = X.shape[0]

    # ---------- Build args for each model and load it ----------
    args_multi = vars(m_multi.helper.get_args_local(['--num_HDV', str(num_HDV)]))
    Nfut = int(args_multi['Nfut'])

    model_multi = m_multi.load_trained_model(
        args_multi,
        resolve_best_checkpoint(args.multistep_dir, collection=MULTISTEP_MODEL_COLLECTION),
        DEVICE,
    )

    args_ours = vars(m_ours.helper.get_args_local(['--num_HDV', str(num_HDV)]))
    args_koop = vars(m_koop.helper.get_args_local(['--num_HDV', str(num_HDV)]))
    args_lin = vars(m_lin.helper.get_args_local(['--num_HDV', str(num_HDV)]))

    model_ours = m_ours.load_trained_model(args_ours, resolve_best_checkpoint(MODEL_DIRS['One-Step Predictor']), DEVICE)
    model_koop = m_koop.load_trained_model(args_koop, resolve_best_checkpoint(MODEL_DIRS['Deep Koopman']), DEVICE)
    model_lin = m_lin.load_trained_model(args_lin, resolve_best_checkpoint(MODEL_DIRS['Deep Linear Model']), DEVICE)

    # LSTM: num_HDV=2 configuration; build args manually to avoid default num_HDV=1
    args_lstm = {
        'num_CAV': num_CAV, 'num_HDV': num_HDV,
        's_dim': 2 * (num_CAV + num_HDV), 'u_dim': num_CAV, 'ep_dim': 1,
        'Npst': Npst_lstm, 'Nfut': Nfut, 'LSTM_dim': 2 * (num_CAV + num_HDV) + 8 * num_HDV,
    }
    model_lstm = m_lstm.load_trained_model(args_lstm, resolve_best_checkpoint(MODEL_DIRS['Seq-to-Seq LSTM']), DEVICE)

    # Prediction start t0: requires max(Npst) historical steps and sufficient U/EP/X length
    t0 = max(args.t0, Npst_lstm)
    # Maximum feasible steps to the trajectory end, limited by X / U / EP length
    H_end = min(traj_len - 1 - t0, U.shape[0] - t0, EP.shape[0] - t0)
    if H_end <= 0:
        raise ValueError(f't0={t0} is too late for prediction (traj_len={traj_len}, len(U)={U.shape[0]})')
    if H_end < Nfut:
        raise ValueError(f't0={t0} has only {H_end} steps left, less than multi-step Nfut={Nfut}')
    H = Nfut

    # ---------- Predictions from each model ----------
    mode = 'openloop'
    preds = {}
    preds['Ours'] = multistep_rollout(model_multi, X, U, EP, t0, H, Npst_koop, slice_dim, num_HDV)
    preds['One-Step Predictor'] = koopman_rollout(model_ours, X, U, EP, t0, H, Npst_koop, slice_dim, num_HDV, mode)
    preds['Deep Koopman'] = koopman_rollout(model_koop, X, U, EP, t0, H, Npst_koop, slice_dim, num_HDV, mode)
    preds['Deep Linear Model'] = koopman_rollout(model_lin, X, U, EP, t0, H, Npst_koop, slice_dim, num_HDV, mode)
    preds['Seq-to-Seq LSTM'] = lstm_rollout(model_lstm, X, U, EP, t0, H, Npst_lstm, mode)
    preds['Calibrated IDM'] = idm_rollout(m_idm, X, U, EP, t0, H, dt, mode)

    # Order according to legend display order
    order = [
        'Calibrated IDM',
        'Seq-to-Seq LSTM',
        'Deep Koopman',
        'Deep Linear Model',
        'One-Step Predictor',
        'Ours',
    ]
    preds = {k: preds[k] for k in order}

    # ---------- Print RMSE ----------
    print('\nModel RMSE (fixed Nfut={} steps):'.format(H))
    print(f'{"Model":<14}{"RMSE_spacing":>14}{"RMSE_velocity":>15}')
    rmse_dict = {}
    for name in order:
        rs, rv = compute_rmse(X, t0, H, preds[name])
        rmse_dict[name] = (rs, rv)
        print(f'{name:<14}{rs:>14.4f}{rv:>15.4f}')

    # ---------- Plotting ----------
    save_path = args.save or os.path.join(
        CUR_DIR, 'comparison_figures', f'pred_compare_nfut_traj{args.traj}_t0{t0}_H{H}.pdf'
    )
    plot_comparison(X, t0, H, preds, num_CAV, num_HDV, args.hist_context, dt, save_path, args.traj, mode,
                    rmse_dict=rmse_dict)

    root, ext = os.path.splitext(save_path)
    save_path_v2 = args.save_v2 or f'{root}_v2{ext or ".pdf"}'
    plot_comparison_v2(X, t0, H, preds, num_CAV, num_HDV, args.hist_context, dt, save_path_v2, args.traj, mode)


if __name__ == '__main__':
    main()
