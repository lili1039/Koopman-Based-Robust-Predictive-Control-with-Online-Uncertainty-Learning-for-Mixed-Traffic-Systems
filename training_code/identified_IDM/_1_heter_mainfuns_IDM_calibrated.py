import math
import os
import pickle as pkl
import re
import time
from pathlib import Path

import numpy as np


os.environ["PYTHONWARNINGS"] = "ignore"


COMPACT_FORMAT = "highd_compact_traj_v1"
DEFAULT_NPST = 50
DEFAULT_NFUT = 15
DEFAULT_DT = 0.12
DEFAULT_RANDOM_STATE = 42
# Platoon = 1 CAV + DEFAULT_NUM_HDV HDVs (s_dim = 2*(DEFAULT_NUM_HDV+1)). The pkl
# path, validation and IDM rollout all derive from this, matching the other
# training scripts, so the IDM baseline can be evaluated for any num_HDV.
DEFAULT_NUM_HDV = 4
DEFAULT_IDM_PARAMS = [
    31.63398677,
    0.8485575,
    0.35958698,
    0.39194875,
    2.89009438,
    3.4968872,
]


def find_latest_pkl_path(project_path, n_hdv=DEFAULT_NUM_HDV):
    """Resolve the largest-count compact pkl for the given HDV count.

    Mirrors Heter_HDV_HighD/_2_heter_helperfuns_highd.find_latest_pkl_path so the
    IDM baseline evaluates on exactly the same dataset file as the Koopman models.
    """
    data_dir = Path(project_path) / "HighDDatasets"
    pattern = re.compile(rf"highd_traj_(\d+)_{n_hdv}HDV_u4f\.pkl")
    candidates = []
    for path in data_dir.glob(f"highd_traj_*_{n_hdv}HDV_u4f.pkl"):
        match = pattern.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path.stat().st_mtime, path))

    if not candidates:
        raise FileNotFoundError(
            f"No compact pkl found for num_HDV={n_hdv} under {data_dir}. "
            f"Expected a file like highd_traj_<count>_{n_hdv}HDV_u4f.pkl."
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def read_pkl_as_dict(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The .pkl file does not exist: {file_path}")
    with open(file_path, "rb") as f:
        return pkl.load(f)


def validate_compact_data(data_dict):
    if data_dict.get("format") != COMPACT_FORMAT:
        raise ValueError(
            f"Expected compact HighD pkl format {COMPACT_FORMAT!r}. "
            "Please regenerate data with Heter_HDV_HighD/dataset_process_HighD.ipynb."
        )

    required_keys = {"X", "U", "EP", "lengths", "s_dim", "u_dim", "ep_dim"}
    missing = required_keys.difference(data_dict)
    if missing:
        raise KeyError(f"Compact HighD pkl is missing keys: {sorted(missing)}")

    if int(data_dict["s_dim"]) != 2*(DEFAULT_NUM_HDV + 1):
        raise ValueError(f"Expected s_dim={2*(DEFAULT_NUM_HDV + 1)}, got {data_dict['s_dim']}")
    if int(data_dict["u_dim"]) != 1:
        raise ValueError(f"Expected u_dim=1, got {data_dict['u_dim']}")
    if int(data_dict["ep_dim"]) != 1:
        raise ValueError(f"Expected ep_dim=1, got {data_dict['ep_dim']}")


def build_window_indices(lengths, Npst, Nfut):
    counts = [max(0, int(length) - int(Npst) - int(Nfut)) for length in lengths]
    total = sum(counts)
    indices = np.empty((total, 2), dtype=np.int64)

    offset = 0
    for traj_idx, count in enumerate(counts):
        if count == 0:
            continue
        next_offset = offset + count
        indices[offset:next_offset, 0] = traj_idx
        indices[offset:next_offset, 1] = np.arange(count, dtype=np.int64)
        offset = next_offset

    return indices


def split_indices(sample_count, random_state=DEFAULT_RANDOM_STATE):
    if sample_count < 3:
        raise ValueError(f"Need at least 3 window samples for train/val/test split, got {sample_count}")

    rng = np.random.RandomState(random_state)
    shuffled = rng.permutation(sample_count)

    test_count = int(math.ceil(sample_count * 0.1))
    test_idx = shuffled[:test_count]
    temp_idx = shuffled[test_count:]

    val_count = int(math.ceil(len(temp_idx) * 0.222))
    val_idx = temp_idx[:val_count]
    train_idx = temp_idx[val_count:]
    return train_idx, val_idx, test_idx


def split_trajectory_indices(num_trajectories, random_state=DEFAULT_RANDOM_STATE):
    if num_trajectories < 3:
        raise ValueError(f"Need at least 3 trajectories for train/val/test split, got {num_trajectories}")

    rng = np.random.RandomState(random_state)
    shuffled = rng.permutation(num_trajectories)

    test_count = int(math.ceil(num_trajectories * 0.1))
    test_traj = shuffled[:test_count]
    temp_traj = shuffled[test_count:]

    val_count = int(math.ceil(len(temp_traj) * 0.222))
    val_traj = temp_traj[:val_count]
    train_traj = temp_traj[val_count:]
    return train_traj, val_traj, test_traj


def load_compact_eval_data(path, Npst=DEFAULT_NPST, Nfut=DEFAULT_NFUT, random_state=DEFAULT_RANDOM_STATE):
    data_dict = read_pkl_as_dict(path)
    validate_compact_data(data_dict)

    compact_data = {
        "X": [np.asarray(x, dtype=np.float32) for x in data_dict["X"]],
        "U": [np.asarray(u, dtype=np.float32) for u in data_dict["U"]],
        "EP": [np.asarray(ep, dtype=np.float32) for ep in data_dict["EP"]],
        "lengths": [int(length) for length in data_dict["lengths"]],
    }
    num_traj = len(compact_data["X"])
    train_traj, val_traj, test_traj = split_trajectory_indices(num_traj, random_state=random_state)

    window_indices = build_window_indices(compact_data["lengths"], Npst, Nfut)
    traj_of_window = window_indices[:, 0]
    train_idx = np.where(np.isin(traj_of_window, train_traj))[0]
    val_idx = np.where(np.isin(traj_of_window, val_traj))[0]
    test_idx = np.where(np.isin(traj_of_window, test_traj))[0]

    print(
        "HighD compact IDM data:",
        f"trajectories={num_traj}",
        f"windows={len(window_indices)}",
        f"Npst={Npst}",
        f"Nfut={Nfut}",
    )
    print("split trajectories:", f"train={len(train_traj)}", f"val={len(val_traj)}", f"test={len(test_traj)}")
    print("split windows:", f"train={len(train_idx)}", f"val={len(val_idx)}", f"test={len(test_idx)}")

    return compact_data, window_indices[test_idx], window_indices, train_idx, val_idx


def get_eval_window(compact_data, sample_index, Npst=DEFAULT_NPST, Nfut=DEFAULT_NFUT):
    traj_idx, start = sample_index
    X = compact_data["X"][traj_idx]
    U = compact_data["U"][traj_idx]
    EP = compact_data["EP"][traj_idx]

    current_idx = int(start) + int(Npst)
    x_current = X[current_idx]
    x_future = X[current_idx + 1:current_idx + 1 + Nfut]
    u_future = U[current_idx:current_idx + Nfut]
    ep_future = EP[current_idx:current_idx + Nfut]
    return x_current, x_future, u_future, ep_future


def idm_acceleration(v, dv, s, params):
    """
    IDM acceleration.
    params = [v0, Tgap, a, b, delta, s0]
    """
    v0, Tgap, a, b, delta, s0 = params

    v = max(float(v), 0.0)
    s = max(float(s), 1e-2)

    s_star = s0 + v * Tgap + (v * dv) / (2 * np.sqrt(a * b))
    acc = a * (1 - (v / v0) ** delta - (s_star / s) ** 2)
    return float(np.clip(acc, -5.0, 5.0))


def predict_window(x_current, u_future, ep_future, params, dt=DEFAULT_DT):
    """Roll the platoon forward Nfut steps for an arbitrary number of vehicles.

    State layout (length s_dim = 2*num_veh, num_veh = 1 CAV + num_HDV HDVs):
        [s_{0,1}, v_1, s_{1,2}, v_2, ..., s_{M-1,M}, v_M]
    where x[0::2] are gaps-ahead and x[1::2] are velocities. Vehicle 1 is the CAV
    (velocity driven by the recorded action u, gap to a phantom leader at speed ep);
    vehicles 2..M are HDVs whose accelerations follow the calibrated IDM.
    """
    Nfut = len(u_future)
    state = np.asarray(x_current, dtype=np.float64).copy()
    s_dim = state.shape[0]
    num_veh = s_dim // 2
    x_pred = np.zeros((Nfut, s_dim), dtype=np.float32)

    for k in range(Nfut):
        ep = float(ep_future[k, 0])
        u = float(u_future[k, 0])

        v = state[1::2]                       # current velocities v_1..v_M
        s = state[0::2]                        # current gaps s_{0,1}..s_{M-1,M}
        new_state = np.empty_like(state)

        # vehicle 1: CAV controlled by u, gap to phantom leader moving at ep
        new_state[0] = s[0] + (ep - v[0]) * dt
        new_state[1] = max(v[0] + u * dt, 0.0)

        # vehicles 2..M: HDVs following the calibrated IDM
        for j in range(1, num_veh):
            v_lead, v_self, s_gap = v[j - 1], v[j], s[j]
            acc = idm_acceleration(v_self, v_self - v_lead, s_gap, params)
            new_state[2 * j] = s_gap + (v_lead - v_self) * dt
            new_state[2 * j + 1] = max(v_self + acc * dt, 0.0)

        state = new_state
        x_pred[k] = state

    return x_pred


def run_test_on_dataset(compact_data, test_indices, params, Npst=DEFAULT_NPST, Nfut=DEFAULT_NFUT, dt=DEFAULT_DT):
    print("\nRunning calibrated IDM evaluation on test set...")

    RMSE_vel = 0.0
    RMSE_spa = 0.0
    MAPE_vel = 0.0
    MAPE_spa = 0.0

    for sample_index in test_indices:
        x_current, x_true, u_future, ep_future = get_eval_window(compact_data, sample_index, Npst, Nfut)
        x_pred = predict_window(x_current, u_future, ep_future, params, dt)

        for i in range(Nfut):
            RMSE_vel += np.sum((x_pred[i, 1::2] - x_true[i, 1::2]) ** 2)
            RMSE_spa += np.sum((x_pred[i, 0::2] - x_true[i, 0::2]) ** 2)

            MAPE_vel += np.sum(np.abs(x_pred[i, 1::2] - x_true[i, 1::2]) / (np.abs(x_true[i, 1::2]) + 1e-2))
            MAPE_spa += np.sum(np.abs(x_pred[i, 0::2] - x_true[i, 0::2]) / (np.abs(x_true[i, 0::2]) + 1e-2))

    sample_count = len(test_indices)
    RMSE_vel = np.sqrt(RMSE_vel / Nfut / sample_count)
    RMSE_spa = np.sqrt(RMSE_spa / Nfut / sample_count)
    MAPE_vel = MAPE_vel / Nfut / sample_count * 100.0
    MAPE_spa = MAPE_spa / Nfut / sample_count * 100.0

    print(f"Sample count: {sample_count}")
    print(
        f"Test RMSE_vel: {RMSE_vel:.6f}, "
        f"Test RMSE_spa: {RMSE_spa:.6f}, "
        f"Test MAPE_vel: {MAPE_vel:.6f}%, "
        f"Test MAPE_spa: {MAPE_spa:.6f}%\n"
    )

    return RMSE_vel, RMSE_spa, MAPE_vel, MAPE_spa


def _write_kv_txt(file_path, items):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as fw:
        for key, value in items.items():
            fw.write(f"{key}: {value}\n")


if __name__ == "__main__":
    project_path = Path(__file__).resolve().parents[2]

    Npst = DEFAULT_NPST
    Nfut = DEFAULT_NFUT
    dt = DEFAULT_DT
    num_HDV = DEFAULT_NUM_HDV
    data_path = find_latest_pkl_path(project_path, num_HDV)

    date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    save_dir = os.path.join(project_path, 'Deep_Koop_v1', 'model_highd', f'IDM_{date}')
    _write_kv_txt(os.path.join(save_dir, 'params.txt'), {
        'IDM_params': DEFAULT_IDM_PARAMS,
        'num_HDV': num_HDV,
        'Npst': Npst,
        'Nfut': Nfut,
        'dt': dt,
        'data_path': str(data_path),
        'random_state': DEFAULT_RANDOM_STATE,
    })

    compact_data, test_indices, window_indices, train_idx, val_idx = load_compact_eval_data(
        data_path,
        Npst=Npst,
        Nfut=Nfut,
        random_state=DEFAULT_RANDOM_STATE,
    )

    first_current, first_future, first_u, first_ep = get_eval_window(compact_data, test_indices[0], Npst, Nfut)
    print(
        "first test window shapes:",
        f"x_current={first_current.shape}",
        f"x_future={first_future.shape}",
        f"u_future={first_u.shape}",
        f"ep_future={first_ep.shape}",
    )

    RMSE_vel, RMSE_spa, MAPE_vel, MAPE_spa = run_test_on_dataset(
        compact_data, test_indices, DEFAULT_IDM_PARAMS, Npst, Nfut, dt
    )
    _write_kv_txt(os.path.join(save_dir, 'test_metrics.txt'), {
        'best_model_path': 'N/A (calibrated IDM, no model checkpoint)',
        'num_samples': len(test_indices),
        'RMSE_vel': float(RMSE_vel),
        'RMSE_spa': float(RMSE_spa),
        'MAPE_vel': float(MAPE_vel),
        'MAPE_spa': float(MAPE_spa),
    })
    print(f"✅ IDM eval artifacts saved to {save_dir}")
