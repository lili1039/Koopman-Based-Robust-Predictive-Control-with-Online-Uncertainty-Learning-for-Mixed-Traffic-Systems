# HighD Compact Dataset Processing
# This code reads raw highD CSV files from `HighD_preprocessing/data/`, 
# extracts 3/4/5/6-vehicle car-following platoons, 
# and writes compact pkl files directly to `HighDDatasets/`.

from itertools import product
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd


FORMAT = 'highd_compact_traj_v1'
VEHICLE_COUNTS = (3, 4, 5, 6)
MIN_RAW_FRAMES = 300
DOWNSAMPLE_STEP = 3
U_ROUND_DECIMALS = 4
REFERENCE_NPST = 50
REFERENCE_NFUT = 15

TRACK_COLUMNS = [
    'frame',
    'id',
    'xVelocity',
    'xAcceleration',
    'dhw',
    'precedingId',
    'followingId',
    'laneId',
]


def find_project_root(start=None):
    start_path = Path(start or Path.cwd()).resolve()
    for path in (start_path, *start_path.parents):
        if (path / 'HighD_preprocessing').exists():
            return path
    raise FileNotFoundError('Cannot find project root containing HighD_preprocessing')


PROJECT_ROOT = find_project_root()
PREPROCESSING_DIR = PROJECT_ROOT / 'HighD_preprocessing'
RAW_DATA_DIR = PREPROCESSING_DIR / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'HighDDatasets'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Return all recording IDs from *_tracks files, sorted numerically.
def discover_recording_ids(raw_data_dir=RAW_DATA_DIR):
    recording_ids = []
    for path in raw_data_dir.glob('*_tracks.csv'):
        match = re.fullmatch(r'(\d+)_tracks\.csv', path.name)
        if match:
            recording_ids.append(int(match.group(1)))
    if not recording_ids:
        raise FileNotFoundError(f'No *_tracks.csv files found in {raw_data_dir}')
    return sorted(recording_ids)

# Read the required columns and keep only vehicles marked as numLaneChanges=0 in metadata.
def read_recording(recording_id, raw_data_dir=RAW_DATA_DIR):
    prefix = f'{recording_id:02d}'
    meta_path = raw_data_dir / f'{prefix}_tracksMeta.csv'
    tracks_path = raw_data_dir / f'{prefix}_tracks.csv'
    if not meta_path.exists() or not tracks_path.exists():
        raise FileNotFoundError(f'Missing highD files for recording {prefix}')

    meta = pd.read_csv(meta_path, usecols=['id', 'numLaneChanges'])
    keep_ids = set(meta.loc[meta['numLaneChanges'].eq(0), 'id'].astype(int))

    record = pd.read_csv(tracks_path, usecols=TRACK_COLUMNS)
    # Keep only vehicles marked as numLaneChanges=0 in metadata.
    record = record[record['id'].astype(int).isin(keep_ids)].copy()
    record['id'] = record['id'].astype(int)
    record['precedingId'] = record['precedingId'].astype(int)
    record['followingId'] = record['followingId'].astype(int)
    return record.sort_values(['id', 'frame']).reset_index(drop=True)


def iter_following_ids(df):
    for value in pd.unique(df['followingId']):
        if pd.isna(value):
            continue
        vehicle_id = int(value)
        if vehicle_id != 0:
            yield vehicle_id


def build_vehicle_chains(by_id, vehicle_count):
    seen = set()

    def extend(chain):
        if len(chain) == vehicle_count:
            yield tuple(chain)
            return

        current = by_id.get(chain[-1])
        if current is None:
            return

        for follower_id in iter_following_ids(current):
            if follower_id in chain or follower_id not in by_id:
                continue
            yield from extend(chain + [follower_id])

    for leader_id in by_id:
        for chain in extend([leader_id]):
            if chain in seen:
                continue
            seen.add(chain)
            yield chain


def extract_lane_consistent_segments(df, min_len=MIN_RAW_FRAMES):
    if df.empty:
        return []
    df = df.sort_values('frame')
    lane_blocks = df['laneId'].ne(df['laneId'].shift()).cumsum()
    return [segment for _, segment in df.groupby(lane_blocks, sort=False) if len(segment) >= min_len]


def extract_continuous_frame_segments(frames, min_len=MIN_RAW_FRAMES):
    frames = sorted(int(frame) for frame in frames)
    if not frames:
        return []

    segments = []
    current = [frames[0]]
    for frame in frames[1:]:
        if frame == current[-1] + 1:
            current.append(frame)
        else:
            if len(current) >= min_len:
                segments.append(current)
            current = [frame]
    if len(current) >= min_len:
        segments.append(current)
    return segments

# Given a leader-follower chain, find continuous windows where all vehicles are
# present, lane-consistent, and preserve the following relationship. Each window
# is a candidate training sample.
def iter_chain_segments(by_id, chain, min_len=MIN_RAW_FRAMES):
    role_segments = []
    for position, vehicle_id in enumerate(chain):
        vehicle_df = by_id[vehicle_id]
        if position > 0:
            vehicle_df = vehicle_df[vehicle_df['precedingId'].eq(chain[position - 1])]

        segments = extract_lane_consistent_segments(vehicle_df, min_len=min_len)
        if not segments:
            return
        role_segments.append(segments)

    for segment_combo in product(*role_segments):
        common_frames = set(segment_combo[0]['frame'])
        for segment in segment_combo[1:]:
            common_frames.intersection_update(set(segment['frame']))

        for frame_segment in extract_continuous_frame_segments(common_frames, min_len=min_len):
            frame_set = set(frame_segment)
            role_dfs = []
            valid = True
            for segment in segment_combo:
                part = segment[segment['frame'].isin(frame_set)].sort_values('frame').copy()
                if len(part) != len(frame_segment) or list(part['frame'].astype(int)) != frame_segment:
                    valid = False
                    break
                role_dfs.append(part)
            if valid:
                yield frame_segment, role_dfs


def downsample_role_df(df, selected_frames):
    downsampled = df[df['frame'].isin(selected_frames)].copy()
    downsampled = downsampled.set_index('frame').loc[selected_frames].reset_index()

    averaged_acc = []
    for frame in selected_frames:
        window = df[(df['frame'] >= frame) & (df['frame'] < frame + DOWNSAMPLE_STEP)]
        averaged_acc.append(window['xAcceleration'].mean())
    downsampled['xAcceleration'] = averaged_acc
    return downsampled


def build_training_trajectory(role_dfs):
    selected_frames = [int(frame) for frame in role_dfs[0]['frame'] if int(frame) % DOWNSAMPLE_STEP == 0]
    if not selected_frames:
        return None

    role_downsampled = [downsample_role_df(df, selected_frames) for df in role_dfs]
    if any(len(df) != len(selected_frames) for df in role_downsampled):
        return None

    mean_velocity = np.mean([df['xVelocity'].mean() for df in role_downsampled])
    if mean_velocity < 0:
        for df in role_downsampled:
            df['xVelocity'] = -df['xVelocity']
            df['xAcceleration'] = -df['xAcceleration']

    data = {}
    for follower_idx in range(1, len(role_downsampled)):
        data[f'spacing_{follower_idx - 1}_{follower_idx}'] = role_downsampled[follower_idx]['dhw'].to_numpy(dtype=np.float32)
        data[f'velocity_{follower_idx}'] = role_downsampled[follower_idx]['xVelocity'].to_numpy(dtype=np.float32)
    data['u'] = role_downsampled[1]['xAcceleration'].to_numpy(dtype=np.float32)
    data['ep'] = role_downsampled[0]['xVelocity'].to_numpy(dtype=np.float32)

    trajectory = pd.DataFrame(data)
    if trajectory.isna().any().any():
        return None
    return trajectory


def columns_for_num_hdv(num_hdv):
    vehicle_count = num_hdv + 2
    columns = []
    for follower_idx in range(1, vehicle_count):
        columns.extend([f'spacing_{follower_idx - 1}_{follower_idx}', f'velocity_{follower_idx}'])
    return columns + ['u', 'ep']


def build_compact_data(trajectories, trajectory_meta, num_hdv):
    s_dim = 2 * (1 + num_hdv)
    u_dim = 1
    ep_dim = 1
    columns = columns_for_num_hdv(num_hdv)

    traj_x = []
    traj_u = []
    traj_ep = []
    lengths = []

    for trajectory in trajectories:
        if list(trajectory.columns) != columns:
            raise ValueError('Unexpected trajectory columns')

        x = trajectory.iloc[:, :s_dim].to_numpy(dtype=np.float32, copy=True)
        u = trajectory.iloc[:-1, s_dim:s_dim + u_dim].round(U_ROUND_DECIMALS).to_numpy(dtype=np.float32, copy=True)
        ep = trajectory.iloc[:-1, -ep_dim:].to_numpy(dtype=np.float32, copy=True)

        if u.shape[0] != x.shape[0] - 1 or ep.shape[0] != x.shape[0] - 1:
            raise ValueError('Length mismatch while building compact data')

        traj_x.append(x)
        traj_u.append(u)
        traj_ep.append(ep)
        lengths.append(int(x.shape[0]))

    return {
        'format': FORMAT,
        'file_num': len(trajectories),
        'columns': columns,
        'state_columns': columns[:s_dim],
        'u_columns': columns[s_dim:s_dim + u_dim],
        'ep_columns': columns[-ep_dim:],
        's_dim': s_dim,
        'u_dim': u_dim,
        'ep_dim': ep_dim,
        'num_CAV': 1,
        'num_HDV': num_hdv,
        'u_round_decimals': U_ROUND_DECIMALS,
        'lengths': lengths,
        'trajectory_meta': trajectory_meta,
        'X': traj_x,
        'U': traj_u,
        'EP': traj_ep,
    }


def reference_window_count(lengths, Npst=REFERENCE_NPST, Nfut=REFERENCE_NFUT):
    return sum(max(0, int(length) - int(Npst) - int(Nfut)) for length in lengths)


def validate_compact_data(data):
    assert data['format'] == FORMAT
    assert data['file_num'] == len(data['X']) == len(data['U']) == len(data['EP']) == len(data['lengths'])
    assert data['file_num'] == len(data['trajectory_meta'])

    for x, u, ep, length in zip(data['X'], data['U'], data['EP'], data['lengths']):
        assert x.dtype == np.float32
        assert u.dtype == np.float32
        assert ep.dtype == np.float32
        assert x.shape == (length, data['s_dim'])
        assert u.shape == (length - 1, data['u_dim'])
        assert ep.shape == (length - 1, data['ep_dim'])
        assert np.allclose(u, np.round(u, data['u_round_decimals']), atol=5e-6)

    windows = reference_window_count(data['lengths'])
    if data['lengths']:
        print(
            f"{data['num_HDV']}HDV:",
            f"trajectories={data['file_num']}",
            f"min/max steps={min(data['lengths'])}/{max(data['lengths'])}",
            f"windows={windows}",
        )
    else:
        print(f"{data['num_HDV']}HDV: trajectories=0 windows=0")
    return windows


def save_compact_data(data, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"highd_traj_{data['file_num']}_{data['num_HDV']}HDV_u4f.pkl"
    with save_path.open('wb') as file:
        pickle.dump(data, file)
    print(f'saved {save_path} ({save_path.stat().st_size / 1024 / 1024:.2f} MB)')
    return save_path


def preprocess_highd(recording_ids=None, vehicle_counts=VEHICLE_COUNTS, output_dir=OUTPUT_DIR):
    recording_ids = discover_recording_ids() if recording_ids is None else list(recording_ids)
    collected = {vehicle_count: [] for vehicle_count in vehicle_counts}
    collected_meta = {vehicle_count: [] for vehicle_count in vehicle_counts}

    # Process each *_tracks file.
    for recording_id in recording_ids:
        # Filter columns.
        record = read_recording(recording_id)
        # Split the full DataFrame by vehicle ID and store each group in a dictionary.
        by_id = {int(vehicle_id): group for vehicle_id, group in record.groupby('id', sort=False)}
        # For each vehicle count (3/4/5/6), find valid platoons in this recording,
        # build training trajectories, and collect metadata.
        recording_counts = {vehicle_count: 0 for vehicle_count in vehicle_counts}

        for vehicle_count in vehicle_counts:
            for chain in build_vehicle_chains(by_id, vehicle_count):
                for frame_segment, role_dfs in iter_chain_segments(by_id, chain):
                    trajectory = build_training_trajectory(role_dfs)
                    if trajectory is None:
                        continue

                    collected[vehicle_count].append(trajectory)
                    collected_meta[vehicle_count].append({
                        'recording_id': int(recording_id),
                        'vehicle_ids': tuple(int(vehicle_id) for vehicle_id in chain),
                        'start_frame': int(frame_segment[0]),
                        'end_frame': int(frame_segment[-1]),
                    })
                    recording_counts[vehicle_count] += 1

        count_text = ', '.join(f'{vehicle_count}veh={count}' for vehicle_count, count in recording_counts.items())
        print(f'[{recording_id:02d}] {count_text}')

    results = {}
    for vehicle_count in vehicle_counts:
        num_hdv = vehicle_count - 2
        data = build_compact_data(collected[vehicle_count], collected_meta[vehicle_count], num_hdv)
        validate_compact_data(data)
        save_path = save_compact_data(data, output_dir=output_dir)
        results[num_hdv] = (data, save_path)
    return results

results = preprocess_highd()
