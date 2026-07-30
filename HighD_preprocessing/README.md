# highD Dataset Preprocessing

This directory contains the preprocessing pipeline used to convert raw
[highD dataset](https://www.highd-dataset.com/) recordings into compact
trajectory pickle files for model training.

The raw highD data is not included in this repository. Request the dataset from
the highD dataset owners and place the downloaded CSV files under `data/`
before running the notebook.

## Directory Structure

```text
HighD_preprocessing/
|-- data/                         # Raw highD recordings
|-- dataset_process_HighD.py      # Preprocessing pipeline
`-- README.md
```

The notebook writes final compact datasets directly to the project-level
`HighDDatasets/` directory.

## Raw Dataset Format

The highD dataset contains 60 recordings. Each recording is described by three
CSV files and should be stored in `data/`:

| File | Description |
| --- | --- |
| `*_recordingMeta.csv` | Metadata for each recording, including location, recording time, frame rate, and vehicle counts. |
| `*_tracksMeta.csv` | Per-vehicle trajectory summary and metadata. |
| `*_tracks.csv` | Frame-level vehicle trajectories. |

The preprocessing notebook uses these fields:

| Source | Fields |
| --- | --- |
| `*_tracksMeta.csv` | `id`, `numLaneChanges` |
| `*_tracks.csv` | `frame`, `id`, `xVelocity`, `xAcceleration`, `dhw`, `precedingId`, `followingId`, `laneId` |

## Preprocessing Workflow

Run `dataset_process_HighD.py` from this directory. The notebook:

- keeps only vehicles with `numLaneChanges == 0`;
- extracts continuous same-lane car-following platoons of 3, 4, 5, and 6
  vehicles;
- requires each raw segment to last at least 300 frames;
- downsamples from 25 Hz to 0.12 s by keeping frames where `frame % 3 == 0`;
- replaces each kept acceleration value with the mean acceleration over the
  three-frame raw window;
- flips longitudinal velocity and acceleration signs for opposite-direction
  trajectories so all longitudinal velocities are positive;
- writes compact `highd_compact_traj_v1` pickle files directly to
  `../HighDDatasets/`.

## Generated Files

The vehicle-count mapping follows the training code convention: the first
following vehicle is the CAV, and the remaining following vehicles are HDVs.

| Extracted platoon | Training scale | Output pattern |
| --- | --- | --- |
| 3 vehicles | `1HDV` | `HighDDatasets/highd_traj_<count>_1HDV_u4f.pkl` |
| 4 vehicles | `2HDV` | `HighDDatasets/highd_traj_<count>_2HDV_u4f.pkl` |
| 5 vehicles | `3HDV` | `HighDDatasets/highd_traj_<count>_3HDV_u4f.pkl` |
| 6 vehicles | `4HDV` | `HighDDatasets/highd_traj_<count>_4HDV_u4f.pkl` |

`<count>` is the actual number of trajectories generated for that scale.
