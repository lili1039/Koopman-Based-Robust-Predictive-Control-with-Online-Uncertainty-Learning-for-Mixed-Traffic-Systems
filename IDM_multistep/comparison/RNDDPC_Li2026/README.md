# RNDDPC-HI baseline

Run `run_rnddpc.bat`, or `python run_rnddpc.py`, from any working directory.
The default runs full tracking followed by braking, saving to
`results/main_results/tracking/` and `results/main_results/braking/`.

## Configuration

| Setting | Value |
|---|---|
| Following HDVs per subsystem | [2,3,1,4,2] |
| Horizon / sampling interval | 5 / 0.12 s |
| Encoder | Pretrained HighD history-informed encoder |
| Identification | Frozen IDM excitation data; pseudoinverse rcond=0.001 |
| Residual bounds | Coordinate-wise 10th absolute-residual percentile; scale 1; floor 1e-6 |
| Costs: spacing / speed / input / input variation | 2.5 / 1.5 / 20 / 30 |
| Hard CAV spacing / speed / acceleration bounds | [10,80] m / [0,35] m/s / [-6,4] m/s^2 |
| Terminal condition | Nominal CAV speed equality; feasible target projection when needed |
| Solver | CLARABEL; original norm support cuts; independent generator validation |

## Files

- run_rnddpc.py / run_rnddpc.bat: configuration and launch.
- RNDDPC.py: scenario dynamics, history, metrics and output.
- functions.py: encoder adapter, frozen data and identification.
- solver.py: robust MPC.
- offline_data/: frozen data, seeds and checksum metadata.
