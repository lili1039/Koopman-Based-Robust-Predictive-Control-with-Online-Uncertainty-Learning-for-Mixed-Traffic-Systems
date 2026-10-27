# ========== Standard imports ==========
import argparse
import math
import numpy as np
import pandas as pd
import time
import os, sys
import pickle as pkl
import re
from pathlib import Path

import torch
import torch.nn as nn

np.set_printoptions(threshold=sys.maxsize)
cur_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(cur_path)) # Directory of the full project.

# 1 CAV, num_HDV HDVs, states=output: si, vi, actions: u
# ===== Change only num_HDV here to switch platoon size; downstream dimensions are derived automatically. =====
num_CAV = 1
num_HDV = 2
num_veh = num_CAV + num_HDV

s_dim = 2*(num_CAV + num_HDV)
# state_name is generated dynamically from num_veh: [sCAV, vCAV, sHDV1, vHDV1, ...]
state_name = ['sCAV', 'vCAV'] + [name for i in range(1, num_HDV + 1) for name in (f'sHDV{i}', f'vHDV{i}')]
u_dim = num_CAV
ep_dim = 1

Nfut = 15               # Number of future steps predicted by Koopman linear evolution.
Npst = 50               # Number of historical x steps used to encode the state.

epochs = 1000
patience = 8
lr_step_size = 1


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ('true', '1', 'yes', 'y'):
        return True
    if str(v).lower() in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r}')

SUPPORTED_NUM_HDV = (1, 2, 3, 4)

def find_latest_pkl_path(n_hdv=num_HDV):
    """Return the largest-count compact pkl for the configured HDV count."""

    data_dir = Path(project_path) / 'HighDDatasets'
    pattern = f'highd_traj_*_{n_hdv}HDV_u4f.pkl'
    candidates = []
    for path in data_dir.glob(pattern):
        match = re.fullmatch(rf'highd_traj_(\d+)_{n_hdv}HDV_u4f\.pkl', path.name)
        if match:
            candidates.append((int(match.group(1)), path.stat().st_mtime, path))

    if not candidates:
        return None
    return str(max(candidates, key=lambda item: (item[0], item[1]))[2])


def get_pkl_path(n_hdv=num_HDV, file_num=None):
    """Resolve the compact pkl path for a given num_HDV.

    When file_num is omitted, the latest generated
    HighDDatasets/highd_traj_*_{n_hdv}HDV_u4f.pkl is selected automatically.
    """
    if n_hdv not in SUPPORTED_NUM_HDV:
        raise ValueError(f'num_HDV must be one of {SUPPORTED_NUM_HDV}, got {n_hdv}')

    if file_num is not None:
        return os.path.join(project_path, 'HighDDatasets', f'highd_traj_{file_num}_{n_hdv}HDV_u4f.pkl')

    latest_path = find_latest_pkl_path(n_hdv)
    if latest_path is not None:
        return latest_path
    else:
        raise FileNotFoundError(
            f'No compact pkl file found for num_HDV={n_hdv}.'
            f'Run the data-processing notebook first to generate the matching pkl, or specify file_num to access a specific file directly.'
        )   


# ====================================================================
# Switchable temporal module: unified interface (B, Npst, slice_dim) -> (B, style_dim)
# Processing flow: temporal encoding -> aggregate by pooling (mean or last) -> tanh(Linear -> style_dim)
# ====================================================================
class RNNStyleEncoder(nn.Module):
    """LSTM/GRU style encoder with optional bidirectionality."""
    def __init__(self, rnn_type, slice_dim, hidden_size, style_dim, bidirectional=False, pooling='mean'):
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type == 'lstm':
            rnn_cls = nn.LSTM
        elif rnn_type == 'gru':
            rnn_cls = nn.GRU
        else:
            raise ValueError(f'Unsupported rnn_type: {rnn_type}')
        self.rnn = rnn_cls(
            input_size=slice_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        proj_in = hidden_size * (2 if bidirectional else 1)
        self.style_proj = nn.Linear(proj_in, style_dim)
        self.pooling = pooling

    def forward(self, hist):
        # hist: (B, Npst, slice_dim)
        out, _ = self.rnn(hist)                  # (B, Npst, proj_in)
        pooled = out.mean(dim=1) if self.pooling == 'mean' else out[:, -1, :]
        return torch.tanh(self.style_proj(pooled))  # (B, style_dim)


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1)]


class TransformerStyleEncoder(nn.Module):
    """Transformer Encoder style encoder."""
    def __init__(self, slice_dim, d_model, nhead, num_layers, dim_feedforward, style_dim, max_len, pooling='mean'):
        super().__init__()
        self.input_proj = nn.Linear(slice_dim, d_model)
        self.pos_enc = _SinusoidalPositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.style_proj = nn.Linear(d_model, style_dim)
        self.pooling = pooling

    def forward(self, hist):
        # hist: (B, Npst, slice_dim)
        h = self.input_proj(hist)        # (B, Npst, d_model)
        h = self.pos_enc(h)
        h = self.transformer(h)          # (B, Npst, d_model)
        pooled = h.mean(dim=1) if self.pooling == 'mean' else h[:, -1, :]
        return torch.tanh(self.style_proj(pooled))  # (B, style_dim)


def build_temporal_encoder(args):
    """Return the style encoder module selected by args['temporal_module'].
    Supported values: 'lstm' (default), 'bilstm', 'gru', 'bigru', and 'transformer'.
    Pooling is controlled by args['style_pooling'] ('mean' or 'last').
    """
    kind = str(args['temporal_module']).lower()
    slice_dim = int(args['HV_hist_slice_dim'])
    hidden = int(args['LSTM_output_dim'])
    style_dim = int(args['style_dim'])
    pooling = str(args['style_pooling']).lower()

    if kind == 'lstm':
        return RNNStyleEncoder('lstm', slice_dim, hidden, style_dim, bidirectional=False, pooling=pooling)
    if kind == 'bilstm':
        return RNNStyleEncoder('lstm', slice_dim, hidden, style_dim, bidirectional=True, pooling=pooling)
    if kind == 'gru':
        return RNNStyleEncoder('gru', slice_dim, hidden, style_dim, bidirectional=False, pooling=pooling)
    if kind == 'bigru':
        return RNNStyleEncoder('gru', slice_dim, hidden, style_dim, bidirectional=True, pooling=pooling)
    if kind == 'transformer':
        return TransformerStyleEncoder(
            slice_dim=slice_dim,
            d_model=int(args['transformer_d_model']),
            nhead=int(args['transformer_nhead']),
            num_layers=int(args['transformer_num_layers']),
            dim_feedforward=int(args['transformer_ff']),
            style_dim=style_dim,
            max_len=int(args['Npst']),
            pooling=pooling,
        )
    raise ValueError(
        f"Unknown temporal_module={kind!r}. "
        "Choose from 'lstm', 'bilstm', 'gru', 'bigru', 'transformer'."
    )


def get_args_local(argv=None):
    parser = argparse.ArgumentParser()

    filename = os.path.basename(__file__)
    parser.add_argument('--Current_filename', type=str, default=filename, help='Current running file')

    Time_step = 0.12
    parser.add_argument('--Time_step', type=float, default=Time_step, help='Time step in simulation')

    parser.add_argument('--is_train', type=_str2bool, default=True, help='True: training False: Testing')

    # ------------------------------ model --------------------
    parser.add_argument('--num_CAV', type=int, default=num_CAV, help='number of CAVs')
    parser.add_argument('--num_HDV', type=int, default=num_HDV, help='number of HDVs')

    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the platoon dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the platoon dynamics')
    parser.add_argument('--ep_dim', type=int, default=ep_dim, help='the dim of the reference input of the platoon dynamics')

    # ---------------------- construction ------------------------
    # Driving-style feature dimension for each HDV
    style_dim = 16
    parser.add_argument('--style_dim', type=int, default=style_dim, help='the feature dim')

    # Lifted state dimension encoded(lift) = 8 * num_HDV
    lift_dim = 8 * num_HDV
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted (encoded) dim, = 8*num_HDV')

    # Encoder input dimension = s_dim + num_HDV*style_dim, output = lift_dim
    encoder_widths = [s_dim + num_HDV*style_dim, 96, 128, lift_dim] # layers
    parser.add_argument('--encoder_widths', default=encoder_widths, help='the construction of the auto-encoder')

    eact_type = ['relu', 'relu', 'relu']
    parser.add_argument('--eact_type', type=str, default=eact_type, help='the type of activation')

    # ---------- Temporal module (switchable)----------
    # 'lstm' (default/aligned with _1_), 'bilstm', 'gru', 'bigru', 'transformer'
    parser.add_argument('--temporal_module', type=str, default='lstm',
                        help="temporal encoder type: lstm/bilstm/gru/bigru/transformer")
    parser.add_argument('--style_pooling', '--pooling', dest='style_pooling', type=str,
                        default='last', choices=['mean', 'last'],
                        help="how the temporal encoder reduces the sequence to a style vector")
    HV_hist_slice_dim = 3
    parser.add_argument('--HV_hist_slice_dim', type=int, default= HV_hist_slice_dim, help='In each HV, the dim of each slice in temporal input')
    LSTM_output_dim = 24
    parser.add_argument('--LSTM_output_dim', type=int, default= LSTM_output_dim, help='RNN hidden size (LSTM/GRU)')
    # Transformer-specific hyperparameters
    parser.add_argument('--transformer_d_model', type=int, default=24, help='Transformer d_model')
    parser.add_argument('--transformer_nhead', type=int, default=4, help='Transformer attention heads')
    parser.add_argument('--transformer_num_layers', type=int, default=2, help='Transformer encoder layers')
    parser.add_argument('--transformer_ff', type=int, default=64, help='Transformer feedforward dim')

    # Training output: decoder is a linear net
    # g = [x, encoded] => total = s_dim + lift_dim
    total_state_dim = s_dim + lift_dim
    decoder_widths = [total_state_dim, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths, help='the construction of the auto-decoder')

    # ------------------------ loss function ------------
    parser.add_argument('--Nfut', type=int, default = Nfut,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--Npst', type=int, default = Npst,
                        help='the steps of history data for feature extraction')

    # Per-state loss weights, set according to training behavior
    weight_spacing = 1
    weight_velocity = 1
    state_weight = np.array([weight_spacing if i % 2 == 0 else weight_velocity
                           for i in range(2 * num_veh)])
    parser.add_argument('--state_weight', type=list, default=state_weight,
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')

    parser.add_argument('--linear_lam',     type=float,     default=1,              help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam',         type=float,     default=1e-4,           help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam',      type=float,     default=1,              help='the coefficient of reconstruction loss')
    parser.add_argument('--Linf_lam',       type=float,     default=1e-4,           help='the coefficient of infinite norm')
    parser.add_argument('--predict_lam',    type=float,     default=2,              help='the coefficient of prediction loss')

    # ---- omega(one-step residual) penalty switch: False=original scheme(only penalizes rollout e); True=additionally penalizes teacher-forcing one-step residual w ----
    parser.add_argument('--omega_penalty_on', type=_str2bool, default=False,
                        help='When true, add a teacher-forcing one-step residual w penalty to the loss (used to reduce the tube template)')
    parser.add_argument('--omega_lam',      type=float,     default=1.0,            help='omega(one-step residual) penalty coefficient, active only when omega_penalty_on=True')

    parser.add_argument('--opt_alg',        type=str,       default='adam',         help='optimizer')
    parser.add_argument('--batch_size',     type=int,       default=128,            help='the size of each batch')
    parser.add_argument('--learning_rate',  type=float,     default=5 *10 ** (-4),     help='learning rate')
    parser.add_argument('--decay_rate',     type=float,     default=0.97,            help='decay rate of learning rate')
    parser.add_argument('--keep_prob',      type=float,     default=0.8,            help='keep prob')
    parser.add_argument('--epochs',         type=int,       default=epochs,         help='maximum training epochs')
    parser.add_argument('--patience',       type=int,       default=patience,       help='early stopping patience')
    parser.add_argument('--lr_step_size',   type=int,       default=lr_step_size,   help='StepLR step size')

    # -------------- path ---------------------
    # path of model, params and loss during training
    date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    model_collection_path = os.path.join(project_path,'Deep_Koop_v1/model_highd/')
    path = model_collection_path + str(date)
    model_repo_path = path + '/past_models'

    parser.add_argument('--project_path', type=str, default=project_path,
                    help='the path for the whole project, including data, training, MPC, etc.')
    parser.add_argument('--path', type=str, default=path,
                    help='the path for saving model info')
    parser.add_argument('--model_repo_path', type=str, default=model_repo_path,help='the path for saving past model')
    parser.add_argument('--model_path', type=str, default=path + '/model.weights.h5',
                        help='the path for saving trained model')
    parser.add_argument('--loss_path', type=str, default=path + '/loss.csv',
                        help='the path for saving loss')
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')

    # path of the compact trajectory data; windows are sliced during training by Npst/Nfut
    # Automatically derived from num_HDV; changing the platoon size points to the matching pkl
    data_path = get_pkl_path(num_HDV)
    parser.add_argument('--data_path', type=str,
                        default=data_path, help='the path of compact trajectory data')

    # checkpoint used for evaluation (is_train=False); weight_path / fig_path are derived from its parent dir
    eval_model_path = os.path.join(
        project_path,
        'Deep_Koop_v1/model_highd/2026_06_14_22_20_38/past_models/model_epoch_143.weights.h5',
    )
    parser.add_argument('--eval_model_path', type=str, default=eval_model_path,
                        help='the path of model weights for testing')

    args = parser.parse_args(argv)

    # Recompute every value derived from num_HDV / num_CAV so that CLI overrides
    # (e.g. --num_HDV 3) propagate consistently. Pre-parse defaults are based on
    # the module-level num_HDV and become stale once the user passes a different
    # value on the command line.
    args.num_veh = args.num_CAV + args.num_HDV
    args.s_dim = 2 * args.num_veh
    args.u_dim = args.num_CAV
    args.lift_dim = 8 * args.num_HDV
    args.encoder_widths = [args.s_dim + args.num_HDV * args.style_dim, 96, 128, args.lift_dim]
    args.decoder_widths = [args.s_dim + args.lift_dim, args.s_dim]
    args.state_weight = np.array([1 if i % 2 == 0 else 1 for i in range(2 * args.num_veh)])
    args.data_path = get_pkl_path(args.num_HDV)

    return args

def save_dict_as_txt(dict, save_dir):
    # Check whether the directory exists and create it if needed
    save_dir_path = os.path.dirname(save_dir)
    if save_dir_path and not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path, exist_ok=True)

    with open(save_dir, 'w') as fw:
        for key in dict.keys():
            fw.writelines(key + ': ' + str(dict.get(key)) + '\n')
        fw.close()
    print('save %s dictionary as .txt successfully.' % save_dir)

def save_dict_as_pkl(dict, savePath):
    # Check whether the directory exists and create it if needed
    save_dir_path = os.path.dirname(savePath)
    if save_dir_path and not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path, exist_ok=True)

    file = open(savePath, 'wb+')
    pkl.dump(dict, file)
    file.close()
    print('save %s dictionary as .pkl successfully.' % savePath)

def read_pkl_as_dict(file_path):
    if not os.path.exists(file_path):
        raise Exception('The .pkl file is not exist. ----> %s ' % file_path)
    with open(file_path, 'rb') as f:
        dict = pkl.load(f)
    return dict
