# ========== Standard imports ==========
import argparse
import re
import time
import os
import pickle as pkl
from pathlib import Path

cur_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(cur_path)) # Directory of the full project.

# 1 CAV, 2HDVs, states=output: si, vi, actions: u
num_CAV = 1
num_HDV = 1

s_dim = 2*(num_CAV + num_HDV)
u_dim = num_CAV
ep_dim = 1

Nfut = 15               # Number of future prediction steps.
Npst = 50+1               # Number of historical x steps.
is_train = True
# Aligned with the Koopman series (g_dim = s_dim + 8*num_HDV).
LSTM_dim = s_dim + 8 * num_HDV
epochs = 1000
patience = 8
batch_size = 128
learning_rate = 10 ** (-4)
decay_rate = 0.97
lr_step_size = 1

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
    """Resolve the compact pkl path for a given num_HDV."""
    if n_hdv not in SUPPORTED_NUM_HDV:
        raise ValueError(f'num_HDV must be one of {SUPPORTED_NUM_HDV}, got {n_hdv}')

    if file_num is not None:
        return os.path.join(project_path, 'HighDDatasets', f'highd_traj_{file_num}_{n_hdv}HDV_u4f.pkl')

    latest_path = find_latest_pkl_path(n_hdv)
    if latest_path is not None:
        return latest_path
    else:
        raise FileNotFoundError(f'No compact pkl found for num_HDV={n_hdv} in {project_path}/HighDDatasets')


def get_args_local(is_train=is_train):
    parser = argparse.ArgumentParser()

    parser.add_argument('--is_train', type=bool, default=is_train, help='True: training False: Testing')

    # ------------------------------ model --------------------
    parser.add_argument('--num_CAV', type=int, default=num_CAV, help='number of CAVs')
    parser.add_argument('--num_HDV', type=int, default=num_HDV, help='number of HDVs')
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the platoon dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the platoon dynamics')
    parser.add_argument('--ep_dim', type=int, default=ep_dim, help='the dim of the reference input of the platoon dynamics')

    # ---------------------- construction ------------------------
    parser.add_argument('--LSTM_dim', type=int, default=LSTM_dim, help='the dim of LSTM')

    # ------------------------ loss function ------------       
    parser.add_argument('--Nfut', type=int, default = Nfut,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--Npst', type=int, default = Npst,
                        help='the steps of history data for feature extraction')

    parser.add_argument('--epochs', type=int, default=epochs, help='maximum training epochs')
    parser.add_argument('--patience', type=int, default=patience, help='early stopping patience')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=decay_rate, help='decay rate of learning rate')
    parser.add_argument('--lr_step_size', type=int, default=lr_step_size, help='StepLR step size')
    
    # -------------- path ---------------------
    # path of model, params and loss during training
    date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    model_collection_path = os.path.join(project_path,'Deep_Koop_v1/model_highd/')
    path = model_collection_path + str(date)
    model_repo_path = path + '/past_models'

    parser.add_argument('--path', type=str, default=path,
                        help='the path for saving model info')
    parser.add_argument('--model_repo_path', type=str, default=model_repo_path,help='the path for saving past model')
    parser.add_argument('--params_path', type=str, default=path + '/params.txt', help='the path for saving parameters')
    
    # path of the compact trajectory data; windows are sliced during training by Npst/Nfut
    data_path = get_pkl_path(num_HDV)
    parser.add_argument('--data_path', type=str, default=data_path, help='the path of compact trajectory data')

    eval_model_path = os.path.join(
        project_path,
        'Deep_Koop_v1/model_highd/2026_05_27_23_22/past_models/model_epoch_63.weights.h5'
    )
    parser.add_argument('--eval_model_path', type=str, default=eval_model_path, help='the path of model weights for testing')
    
    args = parser.parse_args(args=[])
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