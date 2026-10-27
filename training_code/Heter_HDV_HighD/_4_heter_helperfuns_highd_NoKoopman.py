# ========== Standard imports ==========
import argparse
import numpy as np
import pandas as pd
import time
import os, sys
import pickle as pkl
import re
from pathlib import Path

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
    # Training output: decoder is a linear net (NoKoopman: g = x, total_state_dim = s_dim)
    total_state_dim = s_dim
    decoder_widths = [total_state_dim, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths, help='the construction of the auto-decoder')

    # Historical-window slice dimension shared by the data-processing pipeline (highd_compact_dataset) and plotting, consistent with _2_
    HV_hist_slice_dim = 3
    parser.add_argument('--HV_hist_slice_dim', type=int, default=HV_hist_slice_dim, help='In each HV, the dim of each slice in temporal input')

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

    parser.add_argument('--opt_alg',        type=str,       default='adam',         help='optimizer')
    parser.add_argument('--batch_size',     type=int,       default=128,            help='the size of each batch')
    parser.add_argument('--learning_rate',  type=float,     default=10 ** (-2),     help='learning rate')
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
        'Deep_Koop_v1/model_highd/2025_12_22_14_24/past_models/model_epoch_146.weights.h5',
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
    args.decoder_widths = [args.s_dim, args.s_dim]
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
