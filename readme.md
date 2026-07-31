# Koopman-Based Robust Predictive Control with Online Uncertainty Learning for Mixed Traffic Systems

<p align="center">
  <strong>Hanli Chen, Li Dai*, Yaling Ma, Junxiao Zhao, Yuanqing Xia, Fellow, IEEE</strong>
</p>


> **Code availability.** This repository is an initial public snapshot. The main training and control source code has not been uploaded yet. We will release the full main training and control code after the paper is accepted.
> 
| <img src="IDM_multistep/figures%20for%20paper/mixed-traffic-flow.png" alt="Mixed traffic flow scenario" width="100%"> 
<img src="IDM_multistep/figures%20for%20paper/network-architecture.png" alt="Network architecture" width="100%"> |

## Repository Structure

```text
.
|-- HighD_preprocessing/          # highD dataset preprocessing notebook
|-- HighDDatasets/                # processed compact trajectory datasets
|-- IDM_multistep/                # online simulation
|-- training_code/                # training, baselines, and plotting utilities
|-- Zonosets_and_preparations/    # zonotopic set preparation
`-- readme.md
```

## HighD Dataset Preprocessing

1. Place the raw highD CSV files in `HighD_preprocessing/data/`.
2. Run `HighD_preprocessing/dataset_process_HighD.ipynb`.
3. The notebook writes compact trajectory files directly to `HighDDatasets/`:
   - `highd_traj_<count>_1HDV_u4f.pkl`
   - `highd_traj_<count>_2HDV_u4f.pkl`
   - `highd_traj_<count>_3HDV_u4f.pkl`
   - `highd_traj_<count>_4HDV_u4f.pkl`

## Model Training

The main proposed-model training code is currently withheld and will be released after paper acceptance. Once the full code is available, the training scripts under `training_code/` can be run from the command line. Use `--num_HDV` to set the platoon scale from 1 to 4. The helper functions resolve the matching `HighDDatasets/highd_traj_*_<num_HDV>HDV_u4f.pkl` file.

Example commands for the complete release:

```bash
cd training_code/Heter_HDV_HighD
python _5_heter_mainfuns_highd.py --Nfut 15 --Npst 50 --num_HDV 2 --is_train true
```

Train the single-step predictor baseline:

```bash
cd training_code/Heter_HDV_HighD
python _2_heter_mainfuns_highd.py --Nfut 15 --Npst 50 --num_HDV 2 --is_train true --omega_penalty_on false
```

Train the deep Koopman baseline:

```bash
cd training_code/Heter_HDV_HighD
python _3_heter_mainfuns_highd_NoLSTM.py --Nfut 15 --Npst 50 --num_HDV 2 --is_train true
```

Train the ablation linear model:

```bash
cd training_code/Heter_HDV_HighD
python _4_heter_mainfuns_highd_NoKoopman.py --Nfut 15 --Npst 50 --num_HDV 2 --is_train true
```

Evaluate the IDM baseline:

```bash
cd training_code/identified_IDM
python _1_heter_mainfuns_IDM_calibrated.py --Nfut 15 --Npst 50 --num_HDV 2
```

Train the seq2seq LSTM baseline:

```bash
cd training_code/SeqtoSeq_LSTM
python _1_heter_mainfuns_LSTM.py --Nfut 15 --Npst 50 --num_HDV 2 --is_train true
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

Then run `Zonosets_and_preparations/enclosing_set.m` in MATLAB with the CORA Toolbox to compute the outer-bounding zonotope sets of prediction errors and reconstruction errors.

## RPC Simulation

The main robust predictive control script is currently withheld and will be released after paper acceptance. In the complete release, the proposed method can be run with `IDM_multistep/run_mpc.bat`. Before running it, edit the scenario and method settings in the batch file to choose the desired simulation scenario and control method.

```bat
cd IDM_multistep
run_mpc.bat
```

Comparison methods can be run from their corresponding folders under `IDM_multistep/comparison/`. For example:

```bash
cd IDM_multistep/comparison/distributed_DeePC
python distributed_DeePC.py
```

```bash
cd IDM_multistep/comparison/dDeeP_LCC_Shang2024
python dDeeP_LCC.py
```

```bash
cd IDM_multistep/comparison/RNDDPC_Li2026
python RNDDPC.py
```

```bash
cd IDM_multistep/comparison/single_MPC
python single_decentralized_MPC.py
```

## Plotting

Before running the plotting script, edit `MODEL_DIRS` near the top of `training_code/plot_model_comparison_nfut.py` so each entry points to the date folder of the trained result you want to compare. Then generate open-loop prediction comparison figures:

```bash
cd training_code
python plot_model_comparison_nfut.py --traj 0 --t0 51
```

 By default, figures are saved to `training_code/comparison_figures/` as:

- `pred_compare_nfut_traj<traj>_t0<t0>_H<Nfut>.pdf`
- `pred_compare_nfut_traj<traj>_t0<t0>_H<Nfut>_v2.pdf`

Run `IDM/Analysis_and_plot.ipynb` to produce paper figures. Before running the notebook, edit the result folder and file names inside it so they point to the simulation outputs you want to plot.

## Citation

Citation information will be added after publication.
