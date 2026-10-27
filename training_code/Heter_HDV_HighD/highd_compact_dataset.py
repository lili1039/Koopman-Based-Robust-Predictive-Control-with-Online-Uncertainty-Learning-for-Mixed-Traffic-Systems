import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


COMPACT_FORMAT = "highd_compact_traj_v1"


class CompactHighDWindowDataset(Dataset):
    def __init__(self, data_dict, sample_indices, args):
        super().__init__()
        self.X = [np.asarray(x, dtype=np.float32) for x in data_dict["X"]]
        self.U = [np.asarray(u, dtype=np.float32) for u in data_dict["U"]]
        self.EP = [np.asarray(ep, dtype=np.float32) for ep in data_dict["EP"]]
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)

        self.Npst = int(args["Npst"])
        self.Nfut = int(args["Nfut"])
        self.s_dim = int(args["s_dim"])
        self.u_dim = int(args["u_dim"])
        self.ep_dim = int(args["ep_dim"])
        self.num_HDV = int(args["num_HDV"])
        self.slice_dim = int(args.get("HV_hist_slice_dim", 3))

        if self.slice_dim != 3:
            raise ValueError("Compact HighD history slicing currently expects HV_hist_slice_dim == 3")
        if self.s_dim < 2 * self.num_HDV + 2:
            raise ValueError("s_dim is too small for the configured num_HDV history slices")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        traj_idx, start = self.sample_indices[idx]
        X = self.X[traj_idx]
        U = self.U[traj_idx]
        EP = self.EP[traj_idx]

        x_hist = np.empty((self.Nfut + 1, self.Npst, self.slice_dim, self.num_HDV), dtype=np.float32)
        for k in range(self.Nfut + 1):
            hist = X[start + k:start + k + self.Npst]
            for hdv_idx in range(self.num_HDV):
                col_start = 2 * hdv_idx + 1
                col_end = col_start + self.slice_dim
                x_hist[k, :, :, hdv_idx] = hist[:, col_start:col_end]

        x = X[start + self.Npst:start + self.Npst + self.Nfut + 1]
        u = U[start + self.Npst:start + self.Npst + self.Nfut]
        ep = EP[start + self.Npst:start + self.Npst + self.Nfut]
        return x_hist, x, u, ep


def collate_fn(batch, device="cpu", dtype=torch.float32):
    x_hist = np.stack([b[0] for b in batch], axis=0)
    x = np.stack([b[1] for b in batch], axis=0)
    u = np.stack([b[2] for b in batch], axis=0)
    ep = np.stack([b[3] for b in batch], axis=0)

    x_hist = np.transpose(x_hist, (1, 0, 2, 3, 4))
    x = np.transpose(x, (1, 0, 2))
    u = np.transpose(u, (1, 0, 2))
    ep = np.transpose(ep, (1, 0, 2))

    return (
        torch.as_tensor(x_hist, dtype=dtype, device=device),
        torch.as_tensor(x, dtype=dtype, device=device),
        torch.as_tensor(u, dtype=dtype, device=device),
        torch.as_tensor(ep, dtype=dtype, device=device),
    )


def build_window_indices(lengths, Npst, Nfut):
    counts = [max(0, int(length) - int(Npst) - int(Nfut)) for length in lengths]
    total = sum(counts)
    indices = np.empty((total, 2), dtype=np.int64)

    # Concatenate windows from all trajectories into one (total number of windows, 2) array，each row is one window
    # the first column is the trajectory index that the window belongs to (traj_idx)
    # the second column is the start position of the window within that trajectory
    offset = 0
    for traj_idx, count in enumerate(counts):
        if count == 0:
            continue
        next_offset = offset + count
        indices[offset:next_offset, 0] = traj_idx
        indices[offset:next_offset, 1] = np.arange(count, dtype=np.int64)
        offset = next_offset

    return indices


def split_indices(sample_count, random_state=42):
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


def split_trajectory_indices(num_trajectories, random_state=42):
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


def validate_compact_data(data_dict, args):
    if data_dict.get("format") != COMPACT_FORMAT:
        raise ValueError(
            f"Expected compact HighD pkl format {COMPACT_FORMAT!r}. "
            "Please regenerate data with dataset_process_HighD.ipynb."
        )

    required_keys = {"X", "U", "EP", "lengths", "s_dim", "u_dim", "ep_dim", "num_HDV"}
    missing = required_keys.difference(data_dict)
    if missing:
        raise KeyError(f"Compact HighD pkl is missing keys: {sorted(missing)}")

    if int(data_dict["s_dim"]) != int(args["s_dim"]):
        raise ValueError(f"Data s_dim={data_dict['s_dim']} does not match args s_dim={args['s_dim']}")
    if int(data_dict["u_dim"]) != int(args["u_dim"]):
        raise ValueError(f"Data u_dim={data_dict['u_dim']} does not match args u_dim={args['u_dim']}")
    if int(data_dict["ep_dim"]) != int(args["ep_dim"]):
        raise ValueError(f"Data ep_dim={data_dict['ep_dim']} does not match args ep_dim={args['ep_dim']}")
    if int(data_dict["num_HDV"]) != int(args["num_HDV"]):
        raise ValueError(f"Data num_HDV={data_dict['num_HDV']} does not match args num_HDV={args['num_HDV']}")


def create_torch_dataloader(data_dict, sample_indices, args, batch_size, device="cpu", shuffle=True):
    dataset = CompactHighDWindowDataset(data_dict, sample_indices, args)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, device=device),
    )


def get_train_val_test_from_compact(helper, args, batch_size, device="cpu", path=None):
    data_path = Path(path or args["data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Compact HighD data pkl does not exist: {data_path}")

    data_dict = helper.read_pkl_as_dict(str(data_path))
    validate_compact_data(data_dict, args)

    num_traj = len(data_dict["X"])
    train_traj, val_traj, test_traj = split_trajectory_indices(num_traj, random_state=42)

    window_indices = build_window_indices(data_dict["lengths"], args["Npst"], args["Nfut"]) # (total number of windows, 2)，each row is one window; the first column is the trajectory index and the second column is the start position within that trajectory
    traj_of_window = window_indices[:, 0]
    train_idx = np.where(np.isin(traj_of_window, train_traj))[0]
    val_idx = np.where(np.isin(traj_of_window, val_traj))[0]
    test_idx = np.where(np.isin(traj_of_window, test_traj))[0]

    print(
        "HighD compact data:",
        f"trajectories={num_traj}",
        f"windows={len(window_indices)}",
        f"Npst={args['Npst']}",
        f"Nfut={args['Nfut']}",
    )
    print("split trajectories:", f"train={len(train_traj)}", f"val={len(val_traj)}", f"test={len(test_traj)}")
    print("split windows:", f"train={len(train_idx)}", f"val={len(val_idx)}", f"test={len(test_idx)}")

    train_loader = create_torch_dataloader(data_dict, window_indices[train_idx], args, batch_size, device, shuffle=True)
    val_loader = create_torch_dataloader(data_dict, window_indices[val_idx], args, batch_size, device, shuffle=True)
    test_loader = create_torch_dataloader(data_dict, window_indices[test_idx], args, batch_size, device, shuffle=True)
    return train_loader, val_loader, test_loader
