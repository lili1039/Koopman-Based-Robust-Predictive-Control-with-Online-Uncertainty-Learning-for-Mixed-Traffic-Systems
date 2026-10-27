# ========== Standard imports ==========
from collections import deque
import argparse
import math
import numpy as np
import pandas as pd
import time
import os, sys
import pickle as pkl
import re
from matplotlib import pyplot as plt
from pyzonotope import Zonotope
import cvxpy as cp

import torch
import torch.nn as nn


def _env_flag(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {'0', 'false', 'no', 'off'}


def _env_float(name, default):
    value = os.environ.get(name)
    return default if value is None or not value.strip() else float(value)


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None or not value.strip() else int(value)


def _env_int_list(name, default):
    # Parse a comma-separated integer list, e.g., "3,1,2,4" -> [3, 1, 2, 4].
    value = os.environ.get(name)
    if value is None or not value.strip():
        return list(default)
    return [int(item) for item in value.split(',') if item.strip()]


# Delay occurrence times: parse a string into a list of simulation step indices.
# Example: input 10-12,20,30-32 gives [10, 11, 12, 20, 30, 31, 32].
# If no input is provided, use two default 5-step windows around 45% and 55% of the simulation.
def _parse_delay_step_windows(spec, total_time_steps):
    max_step = max(total_time_steps - 2, 0)
    steps = set()
    if spec.strip():
        for item in spec.split(','):
            item = item.strip()
            if not item:
                continue
            if '-' in item:
                start_s, end_s = item.split('-', 1)
                start, end = int(start_s), int(end_s)
            else:
                start = end = int(item)
            if start > end:
                start, end = end, start
            steps.update(range(max(1, start), min(max_step, end) + 1))
    else:
        # By default, inject two short delay windows near the middle of the simulation.
        for center in (int(0.45 * max_step), int(0.55 * max_step)):
            steps.update(range(max(1, center - 2), min(max_step, center + 2) + 1))
    return sorted(steps)


# Delayed vehicles: parse KMPC_DELAY_HDVS into the target HDV list.
# Example: 2:1,4:2 means HDV1 in subplatoon 2 and HDV2 in subplatoon 4.
# Note: input indices are 1-based and converted internally.
# If no input is provided, use default targets [(1, 1), (3, 2)].
def _parse_delay_targets(spec, sub_numHDV):
    targets = []
    if spec.strip():
        for item in spec.split(','):
            item = item.strip()
            if not item:
                continue
            sub_s, hdv_s = item.split(':', 1)
            sub_idx = int(sub_s) - 1
            hdv_idx = int(hdv_s)
            if sub_idx < 0 or sub_idx >= len(sub_numHDV):
                raise ValueError(f'Subplatoon index out of range in KMPC_DELAY_HDVS: {item}')
            if hdv_idx < 1 or hdv_idx > sub_numHDV[sub_idx]:
                raise ValueError(f'HDV index out of range in KMPC_DELAY_HDVS: {item}')
            targets.append((sub_idx, hdv_idx))
    else:
        defaults = [(1, 1), (3, 2)]  # zero-based subplatoon, one-based HDV
        for sub_idx, hdv_idx in defaults:
            if sub_idx < len(sub_numHDV) and hdv_idx <= sub_numHDV[sub_idx]:
                targets.append((sub_idx, hdv_idx))
        if not targets:
            for sub_idx, count in enumerate(sub_numHDV):
                if count > 0:
                    targets.append((sub_idx, 1))
                    break
    return targets

# ====== Persistent gurobipy solvers for faster online correction: build once, then update RHS/coefficients before optimize. ======
# Mathematically equivalent to the cvxpy version; single calls drop from about 37 ms to about 1 ms. Fall back to cvxpy when gurobipy is unavailable.
try:
    import gurobipy as _gp
    from gurobipy import GRB as _GRB
    _GRB_ENV = _gp.Env(empty=True)
    _GRB_ENV.setParam('OutputFlag', 0)
    if os.environ.get('KMPC_GRB_THREADS'):
        _GRB_ENV.setParam('Threads', int(os.environ['KMPC_GRB_THREADS']))
    _GRB_ENV.start()
    _HAS_GRB = True
except Exception as _e:  # pragma: no cover
    _HAS_GRB = False
    print(f"[warn] gurobipy is unavailable ({_e}); online correction falls back to the cvxpy solver (slower).")


class Whether_in_Zonotope_Solver_GRB:
    """Feasibility test for whether a point lies inside zonotope(c, G).
    The call convention keeps G as alpha*G0, a scalar scaling of the template
    generators, so the model fixes G0 and only updates variable bounds
    (+/-alpha) and RHS (omega-c) on each call."""
    def __init__(self, c_shape, G_shape, G0):
        self.G0 = np.asarray(G0, dtype=float)
        self.n, self.m = self.G0.shape
        self._g0n = float(np.linalg.norm(self.G0)) or 1.0
        mdl = _gp.Model(env=_GRB_ENV); self.mdl = mdl
        self.gam = mdl.addVars(self.m, lb=-1.0, ub=1.0)
        self._gam_vars = list(self.gam.values())
        self.eq = [mdl.addConstr(_gp.quicksum(self.G0[j, k] * self.gam[k] for k in range(self.m)) == 0.0)
                   for j in range(self.n)]
        mdl.setObjective(0.0, _GRB.MINIMIZE); mdl.update()

    def solve(self, omega_val, c_val, G_val):
        t0 = time.time()
        alpha = float(np.linalg.norm(np.asarray(G_val, dtype=float))) / self._g0n  # G_val = alpha*G0
        self.mdl.setAttr('LB', self._gam_vars, [-alpha] * self.m)
        self.mdl.setAttr('UB', self._gam_vars, [alpha] * self.m)
        self.mdl.setAttr('RHS', self.eq, list(np.asarray(omega_val, dtype=float) - np.asarray(c_val, dtype=float)))
        self.mdl.optimize()
        return (self.mdl.status == _GRB.OPTIMAL, time.time() - t0)


class Min_Zonotope_fixed_vG_Solver_GRB:
    """Optimize only alpha with fixed v and G so W(alpha,v) contains N historical points.
    The model is built once; each call updates RHS (e_s=buf-v). The alpha column
    coefficient (c-v) is changed only when v changes."""
    def __init__(self, c, G, N):
        self.c = np.asarray(c, dtype=float); self.G = np.asarray(G, dtype=float); self.N = N
        self.n, self.m = self.G.shape; self.vcache = None
        mdl = _gp.Model(env=_GRB_ENV); self.mdl = mdl
        self.alpha = mdl.addVar(lb=0.0, ub=1.0)
        self.gam = mdl.addVars(self.m, N, lb=-_GRB.INFINITY)
        self.eq = []   # Match e.reshape(order='F') with i outer and j inner ordering.
        for i in range(N):
            for j in range(self.n):
                self.eq.append(mdl.addConstr(
                    _gp.quicksum(self.G[j, k] * self.gam[k, i] for k in range(self.m)) + 0.0 * self.alpha == 0.0))
        for i in range(N):
            for k in range(self.m):
                mdl.addConstr(self.gam[k, i] - self.alpha <= 0.0)
                mdl.addConstr(-self.gam[k, i] - self.alpha <= 0.0)
        mdl.setObjective(self.alpha, _GRB.MINIMIZE); mdl.update()

    def solve(self, omega_s_val, v_val):
        t0 = time.time()
        v = np.asarray(v_val, dtype=float); buf = np.asarray(omega_s_val, dtype=float)
        e = buf - v[:, None]
        self.mdl.setAttr('RHS', self.eq, list(e.reshape(-1, order='F')))
        if self.vcache is None or not np.array_equal(v, self.vcache):
            h = self.c - v; idx = 0
            for i in range(self.N):
                for j in range(self.n):
                    self.mdl.chgCoeff(self.eq[idx], self.alpha, h[j]); idx += 1
            self.vcache = v.copy()
        self.mdl.optimize()
        if self.mdl.status not in (_GRB.OPTIMAL, _GRB.SUBOPTIMAL):
            return (None, time.time() - t0)
        return (float(self.alpha.X), time.time() - t0)


class Online_learning_Zonotope_Solver_GRB:
    """Online expansion with constant c and G, so all coefficients stay fixed.
    Each call only updates RHS values determined by omega_new, alpha_old, and v_old.
    The L1 term 0.05*||beta_tau_v||_1 uses auxiliary variables."""
    def __init__(self, c, G):
        self.c = np.asarray(c, dtype=float); self.G = np.asarray(G, dtype=float)
        self.n, self.m = self.G.shape
        mdl = _gp.Model(env=_GRB_ENV); self.mdl = mdl
        self.tau = mdl.addVar(lb=0.0, ub=1.0)
        self.y = mdl.addVars(self.n, lb=-_GRB.INFINITY)
        self.tbeta = mdl.addVars(self.m, lb=-_GRB.INFINITY)
        self.bo = mdl.addVars(self.m, lb=-_GRB.INFINITY)   # beta_tau_omega
        self.bv = mdl.addVars(self.m, lb=-_GRB.INFINITY)   # beta_tau_v
        self.tv = mdl.addVars(self.m, lb=0.0)              # L1 aux for |bv|
        c, G = self.c, self.G
        # eq1: -c[j]*tau + y[j] - sum G[j,k]*tbeta[k] == (alpha_old-1)c[j] + (1-alpha_old)v_old[j]  (RHS per call)
        self.eq1 = [mdl.addConstr(-c[j]*self.tau + self.y[j]
                    - _gp.quicksum(G[j, k]*self.tbeta[k] for k in range(self.m)) == 0.0) for j in range(self.n)]
        # |tbeta|<=1-tau-alpha_old: tbeta[k]+tau <= (1-alpha_old); -tbeta[k]+tau <= (1-alpha_old)  (RHS per call)
        self.tb_p = [mdl.addConstr(self.tbeta[k] + self.tau <= 0.0) for k in range(self.m)]
        self.tb_n = [mdl.addConstr(-self.tbeta[k] + self.tau <= 0.0) for k in range(self.m)]
        # eq_omega: -c[j]*tau + y[j] + sum G[j,k]*bo[k] == omega_new[j]-c[j]  (RHS per call)
        self.eqom = [mdl.addConstr(-c[j]*self.tau + self.y[j]
                     + _gp.quicksum(G[j, k]*self.bo[k] for k in range(self.m)) == 0.0) for j in range(self.n)]
        # |bo|<=1-tau: bo[k]+tau<=1; -bo[k]+tau<=1  (RHS const)
        for k in range(self.m):
            mdl.addConstr(self.bo[k] + self.tau <= 1.0); mdl.addConstr(-self.bo[k] + self.tau <= 1.0)
        # eq_y: y[j] - c[j]*tau - sum G[j,k]*bv[k] == 0  (RHS const)
        for j in range(self.n):
            mdl.addConstr(self.y[j] - c[j]*self.tau - _gp.quicksum(G[j, k]*self.bv[k] for k in range(self.m)) == 0.0)
        # |bv|<=tau: bv[k]-tau<=0; -bv[k]-tau<=0  (RHS const)
        for k in range(self.m):
            mdl.addConstr(self.bv[k] - self.tau <= 0.0); mdl.addConstr(-self.bv[k] - self.tau <= 0.0)
        # L1: tv>=bv, tv>=-bv
        for k in range(self.m):
            mdl.addConstr(self.tv[k] - self.bv[k] >= 0.0); mdl.addConstr(self.tv[k] + self.bv[k] >= 0.0)
        mdl.setObjective(-self.tau + 0.05*_gp.quicksum(self.tv[k] for k in range(self.m)), _GRB.MINIMIZE)
        mdl.update()

    def solve(self, omega_new_val, alpha_old_val, v_old_val):
        t0 = time.time()
        om = np.asarray(omega_new_val, dtype=float); a = float(alpha_old_val); v = np.asarray(v_old_val, dtype=float)
        c = self.c
        self.mdl.setAttr('RHS', self.eq1, list((a-1.0)*c + (1.0-a)*v))
        self.mdl.setAttr('RHS', self.tb_p, [1.0-a]*self.m)
        self.mdl.setAttr('RHS', self.tb_n, [1.0-a]*self.m)
        self.mdl.setAttr('RHS', self.eqom, list(om - c))
        self.mdl.optimize()
        if self.mdl.status != _GRB.OPTIMAL:
            return (None, None, None, None, None)
        tau = self.tau.X
        y = np.array([self.y[j].X for j in range(self.n)])
        bv = np.array([self.bv[k].X for k in range(self.m)])
        bo = np.array([self.bo[k].X for k in range(self.m)])
        return (tau, y, bv, bo, time.time() - t0)

# Project directory.
np.set_printoptions(threshold=sys.maxsize)
cur_path = os.getcwd()
project_path = os.path.dirname(cur_path) # Directory of the full project.

# Training result files for different HDVs: _5_ multi-step predictor checkpoints under model_highd_multistep/.
# NOTE: Currently only num_HDV=2 has a trained _5_ model; HDV_1/3/4 still use placeholder old _2_ paths.
#       Train the corresponding _5_ models and generate per-step zonotopes before using heterogeneous multi-segment simulations.
HDV_1_MODEL_PATH = "2026_06_30_17_30_51/past_models/model_epoch_34.weights.h5"
HDV_2_MODEL_PATH = "2026_06_29_00_39_02/past_models/model_epoch_31.weights.h5"
HDV_3_MODEL_PATH = "2026_06_30_17_32_32/past_models/model_epoch_44.weights.h5"
HDV_4_MODEL_PATH = "2026_06_30_17_33_06/past_models/model_epoch_101.weights.h5"

# Historical trajectory length and prediction horizon.
Nfut = 15               # Number of future steps predicted by Koopman linear evolution.
Npst = 50               # Number of historical steps used to encode the state.

# ====================================================================
# Switchable temporal module, aligned with _2_heter_helperfuns_highd.py.
# Unified interface: (B, Npst, slice_dim) -> (B, style_dim).
# Processing flow: temporal encoding -> pooling aggregation (mean or last) -> tanh(Linear -> style_dim).
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


class DeepEDMD(nn.Module):
    """Koopman lifting model used by the MPC script."""

    def __init__(self, args, device='cpu'):
        super().__init__()
        self.args = args
        self.device = device
        self.s_dim = args['s_dim']
        self.u_dim = args['u_dim']
        self.ep_dim = args['ep_dim']
        self.lift_dim = args['lift_dim']
        self.Nfut = args['Nfut']
        self.num_HDV = args['num_HDV']

        # Encoder
        encoder_widths = args['encoder_widths']
        prev = encoder_widths[0]
        encoder_layers = []
        self.keep_prob = self.args['keep_prob']
        encoder_acts = self.args['eact_type']
        for width, act in zip(encoder_widths[1:], encoder_acts):
            encoder_layers.append(nn.Linear(prev, width))
            if act is not None and act.lower() in ['relu', 'tanh', 'sigmoid', 'leakyrelu']:
                if act.lower() == 'relu':
                    encoder_layers.append(nn.ReLU())
                elif act.lower() == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif act.lower() == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif act.lower() == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU())
            if self.keep_prob < 1.0:
                encoder_layers.append(nn.Dropout(p=1 - self.keep_prob))
            prev = width
        self.encoder = nn.Sequential(*encoder_layers)

        # Switchable temporal module (LSTM/BiLSTM/GRU/BiGRU/Transformer): (B, Npst, slice_dim) -> (B, style_dim).
        # Aligned with _2_heter_mainfuns_highd.py.
        self.style_dim = args['style_dim']
        self.temporal_encoder = build_temporal_encoder(args)

        # Decoder
        decoder_widths = args['decoder_widths']
        w_in, w_out = decoder_widths[0], decoder_widths[1]
        self.decoder = nn.Linear(w_in, w_out, bias=False)

        # Direct multi-step predictor matrices, aligned with _5_heter_mainfuns_highd.py.
        # For each horizon k, G0[k], Gu[k][0:k+1], and Ge[k][0:k+1] independently map g0/u/ep
        # to g(t+k+1), without sharing single-step dynamics across horizons.
        total_state_dim = self.s_dim + self.lift_dim
        g_scale = math.sqrt(2 / (total_state_dim + total_state_dim))
        u_scale = math.sqrt(2 / (self.u_dim + total_state_dim))
        ep_scale = math.sqrt(2 / (self.ep_dim + total_state_dim))
        self.G0 = nn.ParameterList([
            nn.Parameter(torch.randn(total_state_dim, total_state_dim) * g_scale)
            for _ in range(self.Nfut)
        ])
        self.Gu = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(torch.randn(self.u_dim, total_state_dim) * u_scale)
                for _ in range(step + 1)
            ])
            for step in range(self.Nfut)
        ])
        self.Ge = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(torch.randn(self.ep_dim, total_state_dim) * ep_scale)
                for _ in range(step + 1)
            ])
            for step in range(self.Nfut)
        ])
        self.to(device)

    def encode(self, x, x_hist_seq, training: bool = False):
        """
        x: (batch, s_dim)
        x_hist_seq: (batch, Npst, HV_hist_slice_dim, num_HDV)
        Run the temporal encoder for each HDV separately:
        hist_i = x_hist_seq[:, :, :, i] -> (batch, Npst, HV_hist_slice_dim).
        Concatenate each HDV style vector, then concatenate with x before the encoder.
        """
        style_vecs = []
        # x_hist_seq expected (batch, Npst, slice_dim, num_HDV)
        for i in range(self.num_HDV):
            hist_i = x_hist_seq[:, :, :, i]              # (batch, Npst, slice_dim)
            style_vec_i = self.temporal_encoder(hist_i)  # (batch, style_dim)
            style_vecs.append(style_vec_i)
        style_concat = torch.cat(style_vecs, dim=-1)  # (batch, num_HDV * style_dim)

        encoder_input = torch.cat([x, style_concat], dim=-1)  # (batch, s_dim + num_HDV*style_dim)
        encoded = self.encoder(encoder_input)
        g = torch.cat([x, encoded], dim=-1)
        return g

    def decode(self, g):
        # linear decoder from lifted state to x
        return self.decoder(g)

    def forward(self, x_seq, u_seq, ep_seq, x_history_seq, training: bool = False):
        """
        x_seq: (T+1, B, s_dim)
        u_seq: (T,   B, u_dim)
        ep_seq:(T,   B, ep_dim)
        x_history_seq: (T+1, B, Npst, slice_dim, num_HDV)
        returns: y_list (list of length Nfut+1, each (B, s_dim)),
                 g_list_pred (list of length Nfut+1, each (B, total_state_dim)),
                 g_list_true (list of length Nfut+1, each (B, total_state_dim))
        """
        x0 = x_seq[0]  # (B, s_dim)
        # x_history_seq[0] has shape (B, Npst, slice_dim, num_HDV)
        g0 = self.encode(x0, x_history_seq[0], training=training)  # (B, total_state_dim)
        g_list_pred = [g0]

        y_list = [self.decode(g0)]

        # compute true encodings for all time steps (Nfut+1)
        g_list_true = [self.encode(x_seq[t], x_history_seq[t], training=training) for t in range(self.Nfut + 1)]

        for step in range(self.Nfut):
            g_step = g0 @ self.G0[step]
            for lag in range(step + 1):
                g_step = g_step + u_seq[lag] @ self.Gu[step][lag] + ep_seq[lag] @ self.Ge[step][lag]
            g_list_pred.append(g_step)
            y_list.append(self.decode(g_step))

        return y_list, g_list_pred, g_list_true

    def save_weights_to_csv(self, directory):
        os.makedirs(directory, exist_ok=True)

        for step in range(self.Nfut):
            step_name = f"step{step + 1:02d}"
            np.savetxt(os.path.join(directory, f"G0_{step_name}.csv"),
                       self.G0[step].detach().cpu().numpy(), delimiter=",")
            for lag in range(step + 1):
                lag_name = f"lag{lag:02d}"
                np.savetxt(os.path.join(directory, f"Gu_{step_name}_{lag_name}.csv"),
                           self.Gu[step][lag].detach().cpu().numpy(), delimiter=",")
                np.savetxt(os.path.join(directory, f"Ge_{step_name}_{lag_name}.csv"),
                           self.Ge[step][lag].detach().cpu().numpy(), delimiter=",")

        # decoder first layer weight
        w = self.decoder.weight.detach().cpu().numpy()
        np.savetxt(os.path.join(directory, "decoder_weight.csv"), w, delimiter=",")

        print(f"Multi-step predictor weights and decoder weights saved to '{directory}/'")


def load_trained_model(args, model_path, device):
    model = DeepEDMD(args, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_args_local(is_train=False, num_HDV_override=None):
    parser = argparse.ArgumentParser()

    # Heterogeneous platoon support: if num_HDV_override is provided, recompute all dimensions locally for that HDV count.
    # This builds matching network structures for different 1CAV+NHDV subplatoons; otherwise use default num_HDV=2.
    # One subplatoon: 1 CAV + N HDVs; state=output: si, vi; action: u.
    num_CAV  = 1
    _num_HDV = 2 if num_HDV_override is None else num_HDV_override
    _num_veh = num_CAV + _num_HDV
    _s_dim   = 2 * _num_veh
    _u_dim   = num_CAV
    _ep_dim  = 1

    filename = os.path.basename(__file__)
    parser.add_argument('--Current_filename', type=str, default=filename, help='Current running file')

    Time_step = 0.12
    parser.add_argument('--Time_step', type=float, default=Time_step, help='Time step in simulation')

    parser.add_argument('--is_train', type=bool, default=is_train, help='True: training False: Testing')

    # ------------------------------ model --------------------
    parser.add_argument('--num_CAV', type=int, default=num_CAV, help='number of CAVs')
    parser.add_argument('--num_HDV', type=int, default=_num_HDV, help='number of HDVs')

    parser.add_argument('--s_dim', type=int, default=_s_dim, help='the dim of the state of the platoon dynamics')
    parser.add_argument('--u_dim', type=int, default=_u_dim, help='the dim of the action of the platoon dynamics')
    parser.add_argument('--ep_dim', type=int, default=_ep_dim, help='the dim of the reference input of the platoon dynamics')

    # ---------------------- construction ------------------------
    # Aligned with _2_heter_mainfuns_highd.py.
    # Driving-feature dimension for each HDV.
    style_dim = 16
    parser.add_argument('--style_dim', type=int, default=style_dim, help='the feature dim')

    # Lifted encoded state dimension = 8 * num_HDV.
    lift_dim = 8 * _num_HDV
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted (encoded) dim, = 8*num_HDV')

    # Training input dimension.
    encoder_widths = [_s_dim + _num_HDV*style_dim, 96, 128, lift_dim] # layers
    parser.add_argument('--encoder_widths', default=encoder_widths, help='the construction of the auto-encoder')

    eact_type = ['relu', 'relu', 'relu']
    parser.add_argument('--eact_type', type=str, default=eact_type, help='the type of activation')

    # ---------- Temporal module, aligned with _2_ ----------
    # These checkpoints were trained with temporal_module='lstm' and style_pooling='last'.
    parser.add_argument('--temporal_module', type=str, default='lstm',
                        help="temporal encoder type: lstm/bilstm/gru/bigru/transformer")
    parser.add_argument('--style_pooling', '--pooling', dest='style_pooling', type=str,
                        default='last', choices=['mean', 'last'],
                        help="how the temporal encoder reduces the sequence to a style vector")
    HV_hist_slice_dim = 3
    parser.add_argument('--HV_hist_slice_dim', type=int, default= HV_hist_slice_dim, help='In each HV, the dim of each slice in temporal input')
    LSTM_output_dim = 24
    parser.add_argument('--LSTM_output_dim', type=int, default= LSTM_output_dim, help='RNN hidden size (LSTM/GRU)')
    # Transformer-specific hyperparameters.
    parser.add_argument('--transformer_d_model', type=int, default=24, help='Transformer d_model')
    parser.add_argument('--transformer_nhead', type=int, default=4, help='Transformer attention heads')
    parser.add_argument('--transformer_num_layers', type=int, default=2, help='Transformer encoder layers')
    parser.add_argument('--transformer_ff', type=int, default=64, help='Transformer feedforward dim')

    # Training output: decoder is a linear net.
    total_state_dim = lift_dim + _s_dim
    decoder_widths = [total_state_dim, _s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths, help='the construction of the auto-decoder')
    parser.add_argument('--keep_prob',      type=float,     default=0.8,            help='keep prob')
    # ------------------------ loss function ------------       
    parser.add_argument('--Nfut', type=int, default = Nfut,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--Npst', type=int, default = Npst,
                        help='the steps of history data for feature extraction')
    
    # -------------- path ---------------------
    # path of model, params and loss during training
    model_collection_path = os.path.join(project_path,'Deep_Koop_v1/')
    print('model_collection_path: ', model_collection_path)
    parser.add_argument('--project_path', type=str, default=project_path,
                    help='the path for the whole project, including data, training, MPC, etc.')
    parser.add_argument('--model_collection_path', type=str, default=model_collection_path,
                    help='the path for all the trained models.')
    
    args = parser.parse_args(args=[])
    return args

def read_csv_2_np(path):
    # Use numpy genfromtxt.
    data = np.genfromtxt(path, delimiter=',', skip_header=0)
    return data

def IDM_dynamics(S, Tgap, v0, veh_len, a_idm, b_idm, s0_idm):
    '''
    Inputs
    S: all-vehicle state data at time k, with state entries for position s and velocity v.
    Tgap: car-following parameter vector with shape (num_CAV+num_HDV,).
    veh_len: vehicle lengths for num_CAV+num_HDV vehicles.
    a_idm/b_idm/s0_idm: heterogeneous IDM parameters for each vehicle, read from CSV
        (maximum acceleration, comfortable deceleration, and minimum spacing).

    Output
    acel: acceleration vector.
    '''
    # limitation of actuators
    acel_max = 4
    dcel_max = -6

    V_diff = S[0:-1,1] - S[1:,1] # the velocity error with former car
    D_diff = S[0:-1,0] - S[1:,0] - veh_len # the pos error with former car
    # D_diff = S[0:-1,0] - S[1:,0]

    s_star = s0_idm + np.maximum(0, Tgap*S[1:,1] - (S[1:,1]*V_diff)/(2*np.sqrt(a_idm*b_idm)))
    acel = a_idm*(1-(S[1:,1]/v0)**4 - (s_star/D_diff)**2)

    # acceleration saturation
    acel = np.where(acel > acel_max, acel_max, acel)
    acel = np.where(acel < dcel_max, dcel_max, acel)

    return acel

def compute_IDM_steady_s(v_PV,v,v0,Tgap,a_idm,b_idm,s0_idm):
    s_star = s0_idm+v*Tgap-(v*(v_PV-v))/(2*np.sqrt(a_idm*b_idm))
    s = s_star/(np.sqrt(1-(v/v0)**4))
    return s

def generate_NEDC_velocity_profile(Tstep=0.12):
    segments = [10, 8, 40, 13, 40, 25, 40, 10, 80]
    total_time = sum(segments)
    time = np.arange(1, total_time, Tstep)
    vel = np.zeros_like(time)

    thresholds = np.cumsum(segments)

    for idx, t in enumerate(time):
        if t <= thresholds[0]:
            vel[idx] = 70
        elif t <= thresholds[1]:
            vel[idx] = 70 - 20/8*(t - thresholds[0])
        elif t <= thresholds[2]:
            vel[idx] = 50
        elif t <= thresholds[3]:
            vel[idx] = 50 + 20/13*(t - thresholds[2])
        elif t <= thresholds[4]:
            vel[idx] = 70
        elif t <= thresholds[5]:
            vel[idx] = 70 + 30/25*(t - thresholds[4])
        elif t <= thresholds[6]:
            vel[idx] = 100
        elif t <= thresholds[7]:
            vel[idx] = 100 - 30/10*(t - thresholds[6])
        elif t <= thresholds[8]:
            vel[idx] = 70

    vel = vel/3.6

    return time, vel

def generate_braking_velocity_profile(Tstep=0.12):
    """
    Emergency braking scenario (total 30 s):
      Phase 1 [0,      1):    constant 20 m/s
      Phase 2 [1,     13):    braking at -5 m/s², 20 → 5 m/s
      Phase 3 [13,     18):    constant 5 m/s
      Phase 4 [18,     21):    acceleration at +5 m/s², 5 → 20 m/s
      Phase 5 [21,     30):    constant 20 m/s
    """
    # Braking intensity/depth, fixed value.
    v_high  = 20.0    # m/s
    v_low   = 8.0     # m/s
    a_brake = -4.0    # m/s^2, braking is negative.
    a_accel =  4.0    # m/s²

    # segment durations (s)
    t_const1  = 1.0
    t_brake   = (v_low - v_high) / a_brake   # = 3 s
    t_const2  = 5.0    # Low-speed hold duration.
    t_accel   = (v_high - v_low) / a_accel   # = 3 s
    t_const3  = 100.0  # Final constant-speed duration.

    segments = [t_const1, t_brake, t_const2, t_accel, t_const3]
    total_time = sum(segments)   # = 30 s
    thresholds = np.cumsum(segments)

    time = np.arange(0, total_time, Tstep)
    vel  = np.zeros_like(time)

    for idx, t in enumerate(time):
        if t < thresholds[0]:
            vel[idx] = v_high
        elif t < thresholds[1]:
            vel[idx] = v_high + a_brake * (t - thresholds[0])
        elif t < thresholds[2]:
            vel[idx] = v_low
        elif t < thresholds[3]:
            vel[idx] = v_low + a_accel * (t - thresholds[2])
        else:
            vel[idx] = v_high

    return time, vel

def generate_cutin_velocity_profile(Tstep=0.12, cutin_time=10.08, brake_duration=5.5, final_speed=4.0):
    """
    Cut-in (safety-critical) leader-speed profile seen by CAV1:
      Phase 1 [0, t_accel):              cruise at v_const  (original PV)
      Phase 2 [t_accel, cutin_time):     original PV accelerates (+a_accel) and pulls
                                         away, opening a gap ahead of the CAV ("sudden acceleration")
      cutin_time:                        a vehicle cuts into the opened gap. The gap shrinks
                                         abruptly (position jump injected in the main loop).
                                         The merge is speed-matched (no velocity jump), so the
                                         predictor stays clean.
      Phase 3 [cutin_time, t_brake_end): the cut-in vehicle brakes hard (a_brake) down to
                                         v_low ("sharp speed drop after cut-in")
      Phase 4 [t_brake_end, end):        constant v_low
    Combined challenge at/after cut-in = small gap + a hard-decelerating leader.
    cutin_time should match the position-jump step in the main loop. brake_duration
    changes deceleration for the specified final speed while retaining the same peak.
    """
    # Fixed profile parameters.
    v_const    = 20.0   # Initial cruising speed, m/s.
    total_time = 80.0   # Total profile duration, s.
    accel_dur  = 3.0    # Acceleration duration before cut-in, s.
    a_accel    = 2.0    # Acceleration before cut-in, m/s^2.
    v_low      = float(final_speed)  # Target speed after deceleration, m/s.

    t_accel = cutin_time - accel_dur
    v_peak = v_const + a_accel * accel_dur
    if not np.isfinite(v_low) or not 0 <= v_low < v_peak:
        raise ValueError('Cut-in final speed must be finite and in [0, peak speed).')
    if not np.isfinite(brake_duration) or brake_duration <= 0:
        raise ValueError('Cut-in brake duration must be finite and positive.')
    t_brake = float(brake_duration)
    a_brake = (v_low - v_peak) / t_brake
    t_brake_end = cutin_time + t_brake

    time = np.arange(0, total_time, Tstep)
    vel = np.empty_like(time)
    for idx, t in enumerate(time):
        if t < t_accel:
            vel[idx] = v_const
        elif t < cutin_time:
            vel[idx] = v_const + a_accel * (t - t_accel)
        elif t < t_brake_end:
            vel[idx] = v_peak + a_brake * (t - cutin_time)
        else:
            vel[idx] = v_low
    return time, vel

def generate_sine_velocity_profile(Tstep=0.12):
    total_time = 180     # seconds
    time = np.arange(0, total_time, Tstep)
    vel = np.zeros_like(time)

    # Sine-wave parameters.
    center = 25         # Center speed: 25 m/s.
    amplitude = 3       # Oscillation range: +/-5 m/s.
    periods = 4         # Four complete sine waves.
    omega = 2 * np.pi * periods / total_time   # Angular frequency.

    # Generate the sine curve.
    for idx, t in enumerate(time):
        vel[idx] = center + amplitude * np.sin(omega * t)

    return time, vel

def generate_history_seq(args,sim_subplatoon_num,sim_num_veh,sub_start_list,sub_numHDV_list,vPV_init,v_init,s_init,Tgap,v0_IDM,Tgap_rate,v0_rate,veh_len,Tstep,a_idm,b_idm,s0_idm):
    """
    Generate historical state sequences with IDM for any number of controlled vehicles.
    Supports heterogeneous subplatoons: each segment is 1 CAV + N HDVs, and N may vary.
    State format: [s01,v1,s12,v2,...,s_(n-1)n, v_n]

    sub_start_list[k]: global vehicle index of the leader for segment k; the segment
        occupies base+1 .. base+num_veh_k.
    sub_numHDV_list[k]: number of HDVs in segment k; CAV count is fixed by args['num_CAV'].

    Returns:
    S shape=(Npst+1,1+sim_num_veh,3)
    x_history_seq_list: each element has shape (1, Npst, slice_dim, numHDV_k);
        dimensions may differ by segment.
    x_seq_list: each element has shape (1, 2*(num_CAV+numHDV_k));
        dimensions may differ by segment.
    """

    Npst = args['Npst']
    num_CAV = args['num_CAV']
    slice_dim = args['HV_hist_slice_dim']
    S = np.zeros([Npst+1,sim_num_veh+1,3]) #pos/vel/acc

    S[:,0,1] = vPV_init
    for i in range(sim_num_veh):
        S[0,i+1,0] = S[0,i,0] - s_init[i]
        S[0,i+1,1] = v_init[i]

    # Time-step simulation.
    for i in range(Npst):
        Tgap_this_step = Tgap * (1 + Tgap_rate/100)**i
        v0_IDM_this_step = v0_IDM * (1 + v0_rate/100)**i
        # print(f"Step {i}: Tgap_this_step: {Tgap_this_step}, v0_IDM_this_step: {v0_IDM_this_step}")
        acel = IDM_dynamics(S[i, :, :], Tgap_this_step, v0_IDM_this_step, veh_len, a_idm, b_idm, s0_idm)
        S[i,1:,2] = acel
        # Update speed and position.
        S[i+1,1:,1] = S[i,1:,1] + acel * Tstep
        S[i+1,:,0] = S[i,:,0] + S[i+1,:,1] * Tstep

    # Convert states to the model input format, extracting each segment by its num_HDV and global offset.
    x_history_seq_list = []
    x_seq_list = []

    for platoon_idx in range(sim_subplatoon_num):
        base = sub_start_list[platoon_idx]
        numHDV_k = sub_numHDV_list[platoon_idx]
        num_veh_k = num_CAV + numHDV_k

        x_history_seq = np.zeros([1, Npst, slice_dim, numHDV_k])
        x_seq = np.zeros([1, 2*num_veh_k])

        for i in range(numHDV_k):
            x_history_seq[0,:,0,i] = S[:-1,base+i+1,1]
            x_history_seq[0,:,1,i] = S[:-1,base+i+1,0] - S[:-1,base+i+2,0]
            x_history_seq[0,:,2,i] = S[:-1,base+i+2,1]
        for i in range(num_veh_k):
            x_seq[0,2*i] = S[Npst,base+i,0]-S[Npst,base+i+1,0]
            x_seq[0,2*i+1] =  S[Npst,base+i+1,1]

        x_history_seq_list.append(x_history_seq)
        x_seq_list.append(x_seq)

    return S,x_history_seq_list,x_seq_list

def plot(args, scenario, S, Tstep, sim_subplatoon_num, sub_numHDV_list, out_dir='temp_result'):
    # Time sequence.
    os.makedirs(out_dir, exist_ok=True)
    time_list = np.arange(S.shape[0]) * Tstep

    # Generate vehicle labels based on the heterogeneous subplatoon structure; each segment HDV count comes from sub_numHDV_list.
    veh_labels = ['Leader']  # Start with the leader vehicle
    veh_color = ['black']
    HDV_idx = 1  # Start counting from 1 (after leader)

    my_colors_cav = [
    '#0066CC',  # Bright blue.
    '#E35A58',  # Slightly bright dark red.
    '#009999',  # Teal green.
    '#7A42F4',  # Purple.
    '#FF8000'   # Orange.
    ]
    my_colors_veh = plt.cm.Greys(np.linspace(0.6,0.3,S.shape[1]-1))

    num_CAV = args['num_CAV']
    veh_pos = 0   # Cumulative index of non-leader vehicles, including CAVs, used as a gray colormap index and kept in range.
    for subplatoon_idx in range(sim_subplatoon_num):
        # Add CAV for this subplatoon
        veh_labels.append(f'CAV_{subplatoon_idx + 1}')
        veh_color.append(my_colors_cav[subplatoon_idx % len(my_colors_cav)])  # Example color for CAV
        veh_pos += num_CAV

        # Add HDVs for this subplatoon
        for hdv_idx in range(sub_numHDV_list[subplatoon_idx]):
            veh_labels.append(f'HDV_{HDV_idx}')
            veh_color.append(my_colors_veh[veh_pos])  # Example color for HDV
            HDV_idx += 1
            veh_pos += 1

    # Each state type: position, speed, and acceleration.
    state_names = ['Position', 'Velocity', 'Acceleration']

    for state_idx, state_name in enumerate(state_names):
        print("Generating figure:", state_name)
        plt.figure(figsize=(10, 6))
        for veh_idx in range(S.shape[1]):
            plt.plot(time_list, S[:, veh_idx, state_idx], label=veh_labels[veh_idx], color=veh_color[veh_idx])
        plt.xlabel('Time (s)')
        plt.ylabel(state_name)
        plt.title(f'{state_name} over Time')
        plt.legend()
        if state_name == 'Velocity':
            if scenario == 'tracking':
                plt.ylim(5,35)
            elif scenario == 'stabilization':
                plt.ylim(15,35)
            elif scenario == 'cutin':
                plt.ylim(0,30)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/plot_{state_name}.png')
        plt.close()

    # Compute vehicle spacing as leader position minus follower position.
    print("Generating figure: Vehicle Spacing")
    plt.figure(figsize=(10, 6))
    for i in range(1, S.shape[1]):  # Start from vehicle 1 and compute spacing with its leader.
        spacing = S[:, i-1, 0] - S[:, i, 0]  # Position difference.
        label = f'{veh_labels[i-1]} - {veh_labels[i]}'
        plt.plot(time_list, spacing, label=label, color=veh_color[i])

    plt.xlabel('Time (s)')
    plt.ylabel('Inter-vehicle Distance (m)')
    plt.title('Vehicle Spacing over Time')
    plt.legend()
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/plot_Spacing.png')
    plt.close()

def plot_alpha_evolution(alpha_collection, Tstep, out_dir='temp_result'):
    """Plot the online disturbance-set scaling factor alpha for each subplatoon."""
    os.makedirs(out_dir, exist_ok=True)
    alpha_collection = np.asarray(alpha_collection, dtype=float)
    if alpha_collection.ndim == 1:
        alpha_collection = alpha_collection[None, :]

    time_list = np.arange(alpha_collection.shape[1]) * Tstep
    print("Generating figure: Alpha Evolution")
    plt.figure(figsize=(10, 6))
    for platoon_idx in range(alpha_collection.shape[0]):
        plt.plot(
            time_list,
            alpha_collection[platoon_idx],
            label=f'Subplatoon {platoon_idx + 1}',
        )

    plt.xlabel('Time (s)')
    plt.ylabel('Scaling Factor Alpha')
    plt.title('Scaling Factor Alpha Evolution')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f'{out_dir}/plot_Alpha_Evolution.png'
    plt.savefig(filename)
    plt.close()
    print(f"Alpha evolution figure saved to '{filename}'")

def lv_acceleration_estimate(S, i, lead_idx, scenario, Tstep, history_steps=None):
    """Return nominal acceleration and observed bounds for the current LV identity.

    At the cut-in, the new LV's measured current acceleration initializes the
    estimate. Afterwards only observations of that LV enter the rolling window.
    The stored physical trajectory is never modified by the identity switch.
    """
    history_steps = Npst if history_steps is None else history_steps
    window = int(os.environ.get('KMPC_EP_WINDOW', '20'))
    event = int(round(float(os.environ.get('KMPC_CUTIN_TIME', '10.08')) / Tstep))
    post_cutin = scenario == 'cutin' and lead_idx == 0 and i >= event
    if post_cutin and i == event:
        current = float(S[history_steps + i, lead_idx, 2])
        return current, current, current
    start = max(i - window, event) if post_cutin else i - window
    past = S[history_steps + start:history_steps + i, lead_idx, 2]
    nominal = float(S[history_steps + i - 1, lead_idx, 2])
    return nominal, float(np.min(past)), float(np.max(past))


def predict_future_ep_and_constraints(scenario, control_method, disturbance_estimate, c_omega_list, G_omega_list, c_gamma_list, G_gamma_list, Nfut, lead_idx, platoon_idx, ep_t, S, i, Tstep, Ge, D):
    """Multi-step predictor version: each prediction step directly uses its own omega/gamma template.
    The propagation no longer goes through A.
    PV velocity uncertainty uses a single correlated delta-a model: the whole horizon shares
    one acceleration deviation delta_a in [diff_min,diff_max]. At lag l it creates velocity
    error delta_a*Tstep*l, aggregated through each Ge[j-1][l] into one direction M_j in the
    lifted space.
      Es_j = Zono_omega_j ⊕ Eep_j,   Eep_j = Zonotope([mid],[[rad]]) @ M_j,
      M_j  = Σ_{l=1}^{j-1}(Tstep·l)·Ge[j-1][l],  mid=(min+max)/2, rad=(max-min)/2
      Ex_j = Es_j @ D ⊕ Zono_gamma_j
    c_omega_list[j-1]/G_omega_list[j-1] is already the omega set for this step
    (possibly after online correction). Ge is a list of lists with Ge[j-1][l]
    shaped (n_high, ep_dim); D is the decoder with shape (n_low, n_high).
    """
    # lead_idx: global index of the current subplatoon leader, obtained from cumulative offsets in the heterogeneous layout.
    ep = np.zeros([1,Nfut+1])
    if disturbance_estimate == False:
        # Method 1: estimate future ep as constant ep_t; no error set or constraint tightening.
        if scenario == 'stabilization':
            ep = np.tile(25, (1, Nfut+1))
        elif scenario == 'tracking' or scenario == 'braking' or scenario == 'cutin':
            ep = np.tile(ep_t, (1, Nfut+1))
        return ep, None

    elif disturbance_estimate == True:
        def _build_Ex_interval(carry_ep, diff_min, diff_max):
            """Build Ex_interval step by step. When carry_ep=True, add the single correlated delta-a PV velocity uncertainty set."""
            mid = (diff_min + diff_max) / 2.0
            rad = (diff_max - diff_min) / 2.0
            Ex_interval = []
            for j in range(1, Nfut+1):
                Es_j = Zonotope(c_omega_list[j-1], G_omega_list[j-1])
                if carry_ep and j > 1:
                    # Single correlated delta-a term: aggregate Ge columns across lags into one direction, weighted by elapsed time Tstep*l.
                    M_j = np.zeros_like(Ge[j-1][0])           # (n_high, ep_dim)
                    for l in range(1, j):
                        M_j = M_j + (Tstep * l) * Ge[j-1][l]
                    Eep_j = Zonotope(np.array([mid]), np.array([[rad]])) @ M_j  # R^n_high, single generator.
                    Es_j = Es_j + Eep_j
                Ex_j = Es_j @ D + Zonotope(c_gamma_list[j-1], G_gamma_list[j-1])
                Ex_interval.append(Ex_j.interval)
            return Ex_interval

        # Method 2: evolve ep_t with the acceleration estimate for the current LV identity.
        if control_method == 'Distributed MPC': # For Distributed MPC, the first subsystem includes both ep and omega errors; later subsystems include only omega errors.
            if platoon_idx == 0:
                ep[0,0] = ep_t
                aPV_t, a_min, a_max = lv_acceleration_estimate(S, i, lead_idx, scenario, Tstep)
                for j in range(1,Nfut+1):
                    ep[0,j] = max(0.0, ep[0,j-1] + aPV_t * Tstep)
                diff_max = a_max-aPV_t
                diff_min = a_min-aPV_t
                Ex_interval = _build_Ex_interval(True, diff_min, diff_max)
                return ep, Ex_interval
            else:
                # Later subsystems: when i>0, the caller overwrites ep using the previous subplatoon optimal sequence.
                # When i==0, no previous sequence exists, so use the current local LV speed as a constant; the error set contains only omega.
                ep = np.tile(ep_t, (1, Nfut+1))
                Ex_interval = _build_Ex_interval(False, 0.0, 0.0)
                return ep, Ex_interval

        elif control_method == 'Decentralized MPC': # In Decentralized MPC, all subsystems estimate ep error.
            ep[0,0] = ep_t
            aPV_t, a_min, a_max = lv_acceleration_estimate(S, i, lead_idx, scenario, Tstep)
            for j in range(1,Nfut+1):
                ep[0,j] = max(0.0, ep[0,j-1] + aPV_t * Tstep)
            diff_max = a_max-aPV_t
            diff_min = a_min-aPV_t
            Ex_interval = _build_Ex_interval(True, diff_min, diff_max)
            return ep, Ex_interval

def save_vehicle_data_to_csv(S, filename_prefix='temp_result/vehicle_data'):
    """
    Save vehicle trajectory data to separate CSV files for each vehicle.
    
    Args:
        S: numpy array with shape (total_time_steps, num_vehicles+1, 3)
            Last dimension: [position, velocity, acceleration]
        filename_prefix: prefix for output CSV files
    """
    num_vehicles = S.shape[1] - 1
    
    # Create directory if it doesn't exist
    os.makedirs(filename_prefix, exist_ok=True)
    
    for veh_idx in range(0, num_vehicles + 1):
        # Extract data for this vehicle (skip PV at index 0)
        vehicle_data = S[:, veh_idx, :]  # Shape: (total_time_steps, 3)
        
        # Create DataFrame with column headers
        df = pd.DataFrame(vehicle_data, columns=['position', 'velocity', 'acceleration'])
        
        # Save to CSV
        filename = f'{filename_prefix}/veh_{veh_idx}.csv'
        df.to_csv(filename, index=False)
        
    print(f"Saved {num_vehicles} vehicle data files to '{filename_prefix}' directory")

def compute_spacing_from_position(S):
    """Compute spacing s_i(k)=p_{i-1}(k)-p_i(k) from positions; vehicle 0 is the PV."""
    pos = S[:, :, 0]                              # (T, N)
    spacing = np.full_like(pos, np.nan, dtype=float)
    spacing[:, 1:] = pos[:, :-1] - pos[:, 1:]
    return spacing

def compute_tracking_metrics(S, scenario, sdes=40.0, vdes_stab=25.0):
    """Compute RMSVE/RMSSE for all vehicles except the PV (indices 1..N-1).
       Ported from Analysis_and_plot.ipynb.
       - stabilization: vdes = vdes_stab (constant)
       - tracking / braking: vdes(k) = v_PV(k), the PV speed at each time step
       sdes is the desired spacing (default 40). Returns {'RMSVE':..., 'RMSSE':...}.
       Note: statistics cover all time steps in S; callers choose the interval, e.g.
       pass S[Npst:] to evaluate only the control segment after historical warm-up.
    """
    vel = S[:, :, 1]                              # (T, N)
    spacing = compute_spacing_from_position(S)    # (T, N)
    vel_eval = vel[:, 1:]                         # (T, N-1), excluding the PV.
    spacing_eval = spacing[:, 1:]                 # (T, N-1)

    if scenario == 'stabilization':
        vel_err = vel_eval - vdes_stab
    else:  # Tracking/braking: use PV speed as the reference.
        vdes = vel[:, 0]                          # (T,)
        vel_err = vel_eval - vdes[:, None]
    rmsve = float(np.sqrt(np.mean(vel_err ** 2)))

    spacing_err = spacing_eval - sdes
    rmsse = float(np.sqrt(np.mean(spacing_err ** 2)))
    return {'RMSVE': rmsve, 'RMSSE': rmsse}

def save_metrics(metrics, filename='temp_result/metrics.csv'):
    """Save the metrics dictionary as a two-column CSV (metric,value)."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pd.DataFrame(list(metrics.items()), columns=['metric', 'value']).to_csv(filename, index=False)
    print(f"Evaluation metrics saved to '{filename}'")

def save_computation_time(compute_time_collection, type, out_dir='temp_result'):
    """
    Save computation time collection to CSV file.
    
    Args:
        compute_time_collection: numpy array of shape (sim_subplatoon_num, total_time_steps-1)
        filename: output CSV filename
    """
    filename=f'{out_dir}/{type}_computation_time.csv'
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(filename, compute_time_collection, delimiter=',')
    print(f"Computation time data saved to '{filename}'")

def save_disturbance_set_data(alpha_collection, tightened_constraint_collection, out_dir='temp_result'):
    """
    Save disturbance set data to CSV files.

    Args:
        alpha_collection: numpy array of shape (sim_subplatoon_num, total_time_steps-1)
        tightened_constraint_collection: list including numpy array of shape (total_time_steps-1, Nfut*4)
    """
    sim_subplatoon_num = alpha_collection.shape[0]
    filename_prefix = f'{out_dir}/disturbance_set'
    # Create directory if it doesn't exist
    os.makedirs(filename_prefix, exist_ok=True)

    # Save alpha
    alpha_filename = f'{filename_prefix}/alpha.csv'
    np.savetxt(alpha_filename, alpha_collection, delimiter=',')

    for platoon_idx in range(sim_subplatoon_num):
        # Save tightened constraints
        tightened_constraint_filename = f'{filename_prefix}/subplatoon_{platoon_idx}_tightened_constraints.csv'
        np.savetxt(tightened_constraint_filename, tightened_constraint_collection[platoon_idx], delimiter=',')
        
    print(f"Disturbance set data saved to '{filename_prefix}/' directory")

# Includes a variable substitution that converts the problem to an LP.
class Online_learning_Zonotope_Solver_CVX:
    def __init__(self, c, G):
        # ===== Parameter definitions; values are updated at each solve. =====
        self.omega_new = cp.Parameter((c.shape[0],)) # Each column is an error point.
        self.alpha_old = cp.Parameter()
        self.v_old = cp.Parameter((c.shape[0],))

        # ===== Decision variables =====
        self.tau = cp.Variable()
        # self.slack_v = cp.Variable() # slack variable
        # self.slack_omega = cp.Variable() # slack variable
        self.y = cp.Variable((c.shape[0],))
        self.tbeta = cp.Variable((G.shape[1],))
        self.beta_tau_omega = cp.Variable((G.shape[1],))
        self.beta_tau_v = cp.Variable((G.shape[1],))

        # ===== template =====
        self.c = c
        self.G = G

        # ===== Constraints and objective =====
        # tau belongs to [0, 1].
        # Add a small penalty weight for the L1 norm of beta_tau_v
        penalty_weight = 0.05
        objective = -self.tau + penalty_weight * cp.norm(self.beta_tau_v, 1)
        constraints = [ self.tau >= 0, self.tau <= 1]

        # old set is contained in new set
        constraints += [
            (1-self.tau)*self.c + self.y - self.alpha_old*self.c - (1-self.alpha_old)*self.v_old == self.G @ self.tbeta,
            cp.abs(self.tbeta) <= 1 - self.tau- self.alpha_old,
        ]

        # omega_new lies in W_new.
        constraints += [
            self.omega_new == (1-self.tau) * self.c + self.y + self.G @ self.beta_tau_omega,
            cp.abs(self.beta_tau_omega) <= (1-self.tau),
        ]

        # v lies in W.
        constraints += [
            self.y == self.tau * self.c + self.G @ self.beta_tau_v,
            cp.abs(self.beta_tau_v) <= self.tau,
        ]

        # ===== Build the optimization problem =====
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def solve(self, omega_new_val, alpha_old_val, v_old_val):
        # Update parameters.
        self.omega_new.value = omega_new_val
        self.alpha_old.value = alpha_old_val
        self.v_old.value = v_old_val

        # Start timing.
        t0 = time.time()

        # Solve.
        self.prob.solve(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
        )

        compute_time = time.time() - t0
        
        if self.prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            print(f"Optimization status: {self.prob.status}")
            return (None,None,None,None,None)
        # if self.prob.status in [cp.INFEASIBLE, cp.UNBOUNDED]:
        #     raise RuntimeError(f"Optimization failed: {self.prob.status}")

        return (self.tau.value, self.y.value, self.beta_tau_v.value, self.beta_tau_omega.value, compute_time)

class Whether_in_Zonotope_Solver_CVX:
    def __init__(self, c_shape, G_shape, G0=None):
        # ===== Parameter definitions; values are updated at each solve. =====
        self.omega = cp.Parameter(c_shape) # Each column is an error point.
        self.c = cp.Parameter(c_shape)
        self.G = cp.Parameter(G_shape)

        # ===== Decision variables =====
        self.beta = cp.Variable((G_shape[1],))

        # ===== Constraints and objective =====
        objective = 0
        constraints = [ 
            self.beta >= -1, self.beta <= 1,
            self.omega == self.c + self.G @ self.beta,
        ]

        # ===== Build the optimization problem =====
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def solve(self, omega_val, c_val, G_val):
        # Update parameters.
        self.omega.value = omega_val
        self.c.value = c_val
        self.G.value = G_val

        # Start timing.
        t0 = time.time()

        # Solve.
        self.prob.solve(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False
        )

        compute_time = time.time() - t0
        
        is_inside = self.prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        return (is_inside, compute_time)

class Min_Zonotope_fixed_vG_Solver_CVX:
    """
    With fixed center offset v and generator direction G_template, optimize only
    the scaling factor alpha so zonotope W(alpha,v) =
    { alpha*c+(1-alpha)*v + alpha*G*beta : |beta|<=1 } contains N historical
    error points.

    Let gamma_i = alpha*beta_i to eliminate beta_i, yielding the LP:
        min  alpha
        s.t. e_s[:,i] == alpha*h + G @ gamma[:,i],
             gamma[:,i] >= -alpha,  gamma[:,i] <= alpha,
             0 <= alpha <= 1
    The parameters h = c_template - v_last and e_s[:,i] = omega_i - v_last are
    precomputed in solve().
    """
    def __init__(self, c, G, N):
        n, m = c.shape[0], G.shape[1]

        # ===== Parameters updated before each solve. =====
        self.e_s = cp.Parameter((n, N))  # e_s[:,i] = omega_i - v_last
        self.h   = cp.Parameter((n,))    # h = c_template - v_last

        # ===== Decision variables =====
        self.alpha = cp.Variable()
        self.gamma = cp.Variable((m, N))  # gamma_i = alpha * beta_i

        # ===== template =====
        self.c = c
        self.G = G
        self.N = N

        # ===== Constraints and objective =====
        objective = self.alpha
        constraints = [self.alpha >= 0, self.alpha <= 1]
        for i in range(N):
            constraints += [
                self.e_s[:, i] == self.alpha * self.h + self.G @ self.gamma[:, i],
                self.gamma[:, i] >= -self.alpha,
                self.gamma[:, i] <= self.alpha,
            ]

        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def solve(self, omega_s_val, v_val):
        """
        omega_s_val : shape (n, N), the N historical error points in the sliding window.
        v_val       : shape (n,), previous offset vector v_last.
        Returns: (alpha_star, compute_time); alpha_star is None if the solve fails.
        """
        self.e_s.value = omega_s_val - v_val[:, np.newaxis]
        self.h.value   = self.c - v_val

        t0 = time.time()
        self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        compute_time = time.time() - t0

        if self.prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Min_Zonotope_fixed_vG_Solver status: {self.prob.status}")
            return (None, compute_time)

        return (float(self.alpha.value), compute_time)

# ====== Solver aliases: use persistent gurobipy versions when available, about 30x faster; otherwise fall back to cvxpy versions. ======
if _HAS_GRB:
    Whether_in_Zonotope_Solver = Whether_in_Zonotope_Solver_GRB
    Min_Zonotope_fixed_vG_Solver = Min_Zonotope_fixed_vG_Solver_GRB
    Online_learning_Zonotope_Solver = Online_learning_Zonotope_Solver_GRB
else:
    Whether_in_Zonotope_Solver = Whether_in_Zonotope_Solver_CVX
    Min_Zonotope_fixed_vG_Solver = Min_Zonotope_fixed_vG_Solver_CVX
    Online_learning_Zonotope_Solver = Online_learning_Zonotope_Solver_CVX
