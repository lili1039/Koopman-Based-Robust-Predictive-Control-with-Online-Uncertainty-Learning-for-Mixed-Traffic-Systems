# ========== System settings ==========
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # show only warnings and errors

# ========== Standard imports ==========
# Adapted for the _5_ direct multistep predictor, where each prediction step has independent G0/Gu/Ge; exports training-set prediction error points step by step.
# Unlike error_statistic.py, which pairs with the _2_ single-step Koopman model and exports only one-step errors, this script handles horizon
# k = 1..Nfut and exports one set of (omega_k, gamma_k) error points for every step.
import numpy as np
import _5_heter_helperfuns_highd as helper
from _5_heter_mainfuns_highd import DeepEDMD, get_train_val_test, load_trained_model
import sys
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Training-result checkpoints for different HDV counts, relative to model_collection_path
# Key is the HDV count; value is the model checkpoint path under model_highd_multistep
# NOTE: This must use a _5_ multistep checkpoint; _2_ weights are incompatible with this structure.
HDV_MODEL_PATHS = {
    2: "2026_06_29_00_39_02/past_models/model_epoch_31.weights.h5",  # saved checkpoint with the lowest val_loss
    1: "2026_06_30_17_30_51/past_models/model_epoch_34.weights.h5",
    3: "2026_06_30_17_32_32/past_models/model_epoch_44.weights.h5",
    4: "2026_06_30_17_33_06/past_models/model_epoch_101.weights.h5",
}

# Temporal module and pooling used by these trained models; must match params.txt and override current defaults
TEMPORAL_MODULE = "lstm"
STYLE_POOLING = "last"


def read_csv_2_np(path):
    # Use numpy genfromtxt.
    data = np.genfromtxt(path, delimiter=',', skip_header=0)
    return data


def compute_multistep_errors(model, loaders, Nfut):
    """Compute error points on the given DataLoaders, here train+val+test over the full dataset.

    For horizon k = 1..Nfut:
        omega_k = g_true(k) - g_pred(k)        direct multistep prediction error in the lifted state
                  where g_pred(k) is directly mapped by _5_ G0[k-1]/Gu/Ge without chaining single-step dynamics

    For each sliding window, use only t = 0:
        gamma = x(0) - decode(g_true(0))       decoder reconstruction error (autoencoder error), not separated by prediction step
                  As the sliding window advances point by point, t=0 already covers all available dataset times, avoiding duplicate sampling over t=0..Nfut

    Args:
        loaders -- iterable list of DataLoaders; error points from all loaders are concatenated

    Returns:
        omega_steps -- dict {k: (N, total_state_dim)}, k = 1..Nfut
        gamma       -- (N, s_dim)
    """
    omega_lists = {k: [] for k in range(1, Nfut + 1)}
    gamma_list = []

    for loader in loaders:
        for batch_idx, (x_hist_seq, x_seq, u_seq, ep_seq) in enumerate(loader):
            # forward directly returns g_list_pred (direct multistep prediction) and g_list_true (stepwise true lifting) for each step
            # g_list_pred / g_list_true are both lists of length Nfut+1; index 0 corresponds to g0
            _, g_list_pred, g_list_true = model(
                x_seq=x_seq, u_seq=u_seq, ep_seq=ep_seq, x_history_seq=x_hist_seq, training=False
            )

            for k in range(1, Nfut + 1):
                # lifted multistep prediction error: true lifted state - direct multistep prediction
                omega_k = g_list_true[k].detach().cpu().numpy() - g_list_pred[k].detach().cpu().numpy()
                omega_lists[k].append(omega_k)

            # decoder reconstruction error(autoencoder error): use only t=0 for each sliding window, avoid repeatedly counting future points inside the sliding window
            gamma_0 = x_seq[0, :, :].detach().cpu().numpy() - model.decode(g_list_true[0]).detach().cpu().numpy()
            gamma_list.append(gamma_0)

    omega_steps = {k: np.vstack(omega_lists[k]) for k in range(1, Nfut + 1)}
    gamma = np.vstack(gamma_list)
    return omega_steps, gamma


def _save_csv(arr, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(arr.tolist())


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    project_path = helper.project_path
    model_collection_path = os.path.join(project_path, 'Deep_Koop_v1/model_highd_multistep/')
    # Error points are saved under Zonosets_and_preparations/error_points_multistep/<date>/
    error_points_root = os.path.join(project_path, 'Zonosets_and_preparations','_5_multi_step', 'error_points_multistep')

    for num_HDV, rel_model_path in HDV_MODEL_PATHS.items():
        print(f"\n================ num_HDV = {num_HDV} ================")

        # 1) Rebuild args consistent with this model from the HDV count; s_dim, lift_dim, Nfut, data_path, etc. are derived automatically
        args = helper.get_args_local([
            '--num_HDV', str(num_HDV),
            '--temporal_module', TEMPORAL_MODULE,
            '--pooling', STYLE_POOLING,
        ])
        args = vars(args)

        s_dim = args['s_dim']
        Nfut = args['Nfut']
        total_input_dim = int(args['lift_dim'] + args['s_dim'])

        # Model checkpoint path and date folder name, e.g. 2026_06_02_11_06_59
        model_path = os.path.join(model_collection_path, rel_model_path)
        date_folder = rel_model_path.split('/')[0]

        # 2) Load the model and use save_weights_to_csv to save each step G0/Gu/Ge and decoder as CSV
        #    Save directory: <date>/multistep_weights/
        model = load_trained_model(args, model_path, device)
        weight_dir = os.path.join(model_collection_path, date_folder, 'multistep_weights')
        model.save_weights_to_csv(directory=weight_dir)

        print(f"Prediction steps Nfut = {Nfut}, total_state_dim = {total_input_dim}, s_dim = {s_dim}")
        print(f"G0[0] shape: {tuple(model.G0[0].shape)}, "
              f"Gu[Nfut-1] contains {len(model.Gu[Nfut-1])} lags, "
              f"Ge[Nfut-1] contains {len(model.Ge[Nfut-1])} lags")

        # 3) Compute multistep prediction errors step by step on the full dataset (train+val+test)
        """
        x_seq -- shape: Nfut+1 * batch_size * s_dim
        u_seq -- shape: Nfut   * batch_size * u_dim
        ep_seq -- shape: Nfut  * batch_size * ep_dim
        x_hist_seq -- shape: Nfut+1 * batch_size * Npst * slice_dim * num_HDV
        """
        batch_size = args['batch_size']
        train_loader, val_loader, test_loader = get_train_val_test(
            batch_size, device, path=args['data_path'], args=args
        )

        omega_steps, gamma = compute_multistep_errors(
            model, [train_loader, val_loader, test_loader], Nfut
        )

        # 4) Save error points step by step to error_points_multistep/<date>/
        #    omega one file group per prediction step: omega_full_set_stepKK.csv
        #    gamma autoencoder error points for the full dataset: gamma_full_set.csv
        #    use only t=0 for each sliding window, avoiding duplicate points caused by t=0..Nfut
        #    (full_set = train+val+test full dataset)
        error_dir = os.path.join(error_points_root, date_folder)
        os.makedirs(error_dir, exist_ok=True)

        gamma_path = os.path.join(error_dir, 'gamma_full_set.csv')
        _save_csv(gamma, gamma_path)
        print(f"  gamma {gamma.shape} -> {os.path.basename(gamma_path)}")

        for k in range(1, Nfut + 1):
            step_name = f"step{k:02d}"
            omega_path = os.path.join(error_dir, f'omega_full_set_{step_name}.csv')

            _save_csv(omega_steps[k], omega_path)

            print(f"  [k={k:2d}] omega {omega_steps[k].shape} -> {os.path.basename(omega_path)}")

        print(f"✔️ num_HDV={num_HDV}: {Nfut} step omega sets and the shared gamma error points were saved to '{error_dir}'")
