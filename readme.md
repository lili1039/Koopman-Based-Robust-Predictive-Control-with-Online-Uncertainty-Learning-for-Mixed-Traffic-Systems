# Koopman-Based Robust Predictive Control with Online Uncertainty Learning for Mixed Traffic Systems

<p align="center">
  <strong>Hanli Chen, Li Dai*, Yaling Ma, Junxiao Zhao, Yuanqing Xia, Fellow, IEEE</strong>
</p>


> **Code availability.** This repository is an initial public snapshot. 
> 
<img src="IDM_multistep/figures%20for%20paper/mixed%20traffic%20flow.png" alt="Mixed traffic flow scenario" width="100%">
<img src="IDM_multistep/figures%20for%20paper/network%20architecture.png" alt="Network architecture" width="100%">

## Repository Structure

```text
.
|-- Deep_Koop_v1/                # trained model checkpoints, exported weights, and training results
|-- HighD_preprocessing/         # highD trajectory preprocessing script and instructions
|-- HighDDatasets/               # output directory for locally generated compact highD datasets
|-- IDM_multistep/               # HK-RPC simulation, comparison baselines, ablation studies, and results
|-- training_code/               # proposed-model and baseline training, evaluation, and plotting
|-- Zonosets_and_preparations/   # prediction error sets and MATLAB zonotope construction
|-- .gitignore                  # version-control exclusion rules
`-- readme.md
```

## HighD Dataset Preprocessing

1. Place the raw highD CSV files (which you must obtain from the official highD dataset provider) in `HighD_preprocessing/data/`.
2. Run `python HighD_preprocessing/dataset_process_HighD.py` from the repository root.
3. The script writes compact trajectory files directly to `HighDDatasets/`:
   - `highd_traj_<count>_1HDV_u4f.pkl`
   - `highd_traj_<count>_2HDV_u4f.pkl`
   - `highd_traj_<count>_3HDV_u4f.pkl`
   - `highd_traj_<count>_4HDV_u4f.pkl`

## Model Training

The training scripts are provided under `training_code/`. For the scripts in `training_code/Heter_HDV_HighD/`, use `--num_HDV` to set the platoon scale from 1 to 4. The helper functions resolve the matching `HighDDatasets/highd_traj_*_<num_HDV>HDV_u4f.pkl` file.

1. Train the proposed model:

```bash
cd training_code/Heter_HDV_HighD
python _5_heter_mainfuns_highd.py --Nfut 15 --Npst 50 --num_HDV 2 --is_train true
```

2. Train the single-step predictor baseline:

```bash
cd training_code/Heter_HDV_HighD
python _2_heter_mainfuns_highd.py --Nfut 15 --Npst 50 --num_HDV 2 --is_train true --omega_penalty_on false
```

3. Train the deep Koopman baseline:

```bash
cd training_code/Heter_HDV_HighD
python _3_heter_mainfuns_highd_NoLSTM.py --Nfut 15 --Npst 50 --num_HDV 2 --is_train true
```

4. Train the ablation linear model:

```bash
cd training_code/Heter_HDV_HighD
python _4_heter_mainfuns_highd_NoKoopman.py --Nfut 15 --Npst 50 --num_HDV 2 --is_train true
```

5. Evaluate the IDM baseline:
   set `DEFAULT_NUM_HDV` in the script to change the evaluation settings before running.

```bash
cd training_code/identified_IDM
python _1_heter_mainfuns_IDM_calibrated.py
```

6. Train the seq2seq LSTM baseline: set `num_HDV` in `_1_heter_helperfuns_LSTM.py` to change the evaluation settings before running.

```bash
cd training_code/SeqtoSeq_LSTM
python _1_heter_mainfuns_LSTM.py
```

## Prediction Error Sets

Before running the error-statistics script, edit `HDV_MODEL_PATHS` near the top of `training_code/Heter_HDV_HighD/error_statistic_multistep.py`. Fill in the trained proposed-model checkpoints for each platoon scale, using paths relative to `Deep_Koop_v1/model_highd_multistep/`. Then run:

```bash
cd training_code/Heter_HDV_HighD
python error_statistic_multistep.py
```

The script loads each checkpoint, exports the multistep Koopman weights to `Deep_Koop_v1/model_highd_multistep/<date>/multistep_weights/`, and computes full-dataset prediction error points. The error points are saved to `Zonosets_and_preparations/_5_multi_step/error_points_multistep/<date>/` as:

- `gamma_full_set.csv`
- `omega_full_set_step01.csv` to `omega_full_set_step<Nfut>.csv`

Then run `Zonosets_and_preparations/_5_multi_step/enclosing_set.m` in MATLAB with the CORA Toolbox to compute the outer-bounding zonotope sets of prediction errors and reconstruction errors.

## RPC Simulation

1. The proposed method can be run with `IDM_multistep/run_mpc.bat`. Before running it, edit `KMPC_SCENARIO` (`tracking` or `braking`) and `KMPC_CONTROL` (`All HDVs`, `Decentralized MPC`, or `Distributed MPC`) in the batch file to choose the simulation scenario and control method.

```bat
cd IDM_multistep
run_mpc.bat
```

2. Cut-in experiments are launched separately using `IDM_multistep/uncertainty_ablation.bat` for uncertainty ablation and `IDM_multistep/run_repeated_safety.bat` for repeated safety evaluation.

3. The four comparison baselines support the tracking and braking scenarios. Run them from their corresponding folders under `IDM_multistep/comparison/`. For distributed DeePC, dDeeP-LCC, and single-CAV MPC, set `KMPC_SCENARIO` to `tracking` or `braking` before launching:

```bat
cd IDM_multistep/comparison/distributed_DeePC
set "KMPC_SCENARIO=tracking"
python distributed_DeePC.py
```

```bat
cd IDM_multistep/comparison/dDeeP_LCC_Shang2024
set "KMPC_SCENARIO=tracking"
python dDeeP_LCC.py
```

```bat
cd IDM_multistep/comparison/single_MPC
set "KMPC_SCENARIO=tracking"
python single_decentralized_MPC.py
```

Replace `tracking` with `braking` in the commands above to run the braking scenario.

For RNDDPC-HI, the batch file runs tracking followed by braking by default:

```bat
cd IDM_multistep/comparison/RNDDPC_Li2026
run_rnddpc.bat
```

To run only one scenario, use `run_rnddpc.bat --scenario tracking` or `run_rnddpc.bat --scenario braking`.

## Plotting

Before running the plotting script, edit `MODEL_DIRS` near the top of `training_code/plot_model_comparison_nfut.py` so each entry points to the date folder of the trained result you want to compare. Then generate open-loop prediction comparison figures:

```bash
cd training_code
python plot_model_comparison_nfut.py --traj 0 --t0 51
```

 By default, figures are saved to `training_code/comparison_figures/` as:

- `pred_compare_nfut_traj<traj>_t0<t0>_H<Nfut>.pdf`
- `pred_compare_nfut_traj<traj>_t0<t0>_H<Nfut>_v2.pdf`

Run `IDM_multistep/Analysis_and_plot.ipynb` from the `IDM_multistep/` directory to produce simulation comparison figures. Before running the notebook, edit the result folder and file names inside it so they point to the simulation outputs you want to plot.

## Citation

Citation information will be added after publication.
