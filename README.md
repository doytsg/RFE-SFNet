# RFE-SFNet

Code for **RFE-SFNet: Robust Feature Extraction-Spectral Filtering Network** for lightweight, noise-robust bearing fault diagnosis from raw one-dimensional vibration signals.

This repository contains the training code, model definitions, dataset loaders, and baseline implementations used in the experiments. Datasets, trained checkpoints, generated figures, logs, and result files are intentionally not included.

## Highlights

- Raw 1-D vibration signal classification for PU and CWRU bearing datasets.
- RFE-SFNet implementation with a Robust Feature Extraction Stem (RFE-Stem) and attention-free Spectral Filter Mixer (SFM).
- Unified training entry for RFE-SFNet and several comparison models.
- SNR-controlled synthetic noise augmentation and evaluation.
- Few-shot training support through fixed samples per class.
- File-level splitting for PU data to reduce leakage between training and testing.

## Repository Structure

```text
.
|-- train_model.py              # Unified training entry
|-- train_rfe_sfnet.py          # Single-model RFE-SFNet training entry
|-- train_common.py             # Shared training, evaluation, splitting, and noise utilities
|-- cwru_dataset.py             # CWRU, PU-BS-10, and PU-A2R dataset loaders
|-- export_tsne_features.py     # Optional feature export and t-SNE utility
|-- models/
|   |-- rfe_sfnet.py            # Public RFE-SFNet import wrapper
|   |-- sds_dsfb_transformer.py # Backward-compatible implementation file
|   |-- sds_frontend.py         # RFE-Stem and Haar-Down modules
|   |-- refined_dsfb_modules.py # SFM and Li-FFN modules
|   |-- cnn_transformer.py
|   |-- convformer_nse.py
|   |-- liconvformer.py
|   |-- almformer.py
|   |-- wdcnn.py
|   |-- tslanet.py
|   |-- drsn_cw.py
|   `-- gtfenet.py
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

For Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

The datasets are not distributed with this repository. Download the original CWRU and PU bearing datasets from their official sources, then place or symlink them locally.

Default data roots used by the scripts:

```text
PU_extracted/    # PU .mat files
data/            # CWRU .mat files
```

For PU-BS-10, the loader expects `.mat` files whose names follow this pattern:

```text
N15_M01_F10_K001_1.mat
N09_M07_F10_KA04_2.mat
...
```

For CWRU, the loader recursively scans `.mat` files and uses directory names such as `Normal Baseline Data`, `Ball`, `Inner Race`, and `Outer Race` to infer labels.

You can also pass a custom data directory:

```bash
python train_model.py rfesfnet --data_dir path/to/PU_extracted
python train_model.py rfesfnet --dataset cwru --data_dir path/to/CWRU
```

## Quick Start

Train RFE-SFNet on the default PU-BS-10 setup:

```bash
python train_model.py rfesfnet --data_dir PU_extracted --epochs 100
```

Train under fixed Gaussian noise and evaluate multiple SNR levels:

```bash
python train_model.py rfesfnet ^
  --data_dir PU_extracted ^
  --train_noise --val_noise --snr_per_sample ^
  --noise_type gaussian --train_snr_min -12 --train_snr_max -12 ^
  --test_noise --test_noise_types gaussian ^
  --snr_list -12,-8,-4,0 ^
  --epochs 100
```

Few-shot training:

```bash
python train_model.py rfesfnet --data_dir PU_extracted --train_samples_per_class 50
```

CWRU training:

```bash
python train_model.py rfesfnet --dataset cwru --data_dir data --num_classes 10
```

Alternative command aliases are also supported:

```bash
python train_model.py rfe-sfnet ...
python train_model.py sds_dsfb ...
```

## Supported Models

The unified entry supports:

- `rfesfnet`
- `cnn_transformer`
- `convformer_nse`
- `liconvformer`
- `almformer`
- `wdcnn`
- `tslanet`
- `drsn_cw`
- `gtfenet`
- `mslk`

Show command-line options with:

```bash
python train_model.py -h
python train_model.py rfesfnet -h
```

## Outputs

Training writes checkpoints, curves, confusion matrices, and CSV logs to the configured results directory, usually `results/`. These generated files are ignored by git and should not be committed.

Useful options:

- `--results_dir results`
- `--run_name my_run`
- `--seed 42`
- `--num_workers 0`
- `--batch_size 128`
- `--window_size 2048`
- `--stride 2048`

## Notes

- The public model name is RFE-SFNet. Some implementation files retain the earlier internal `sds_dsfb` name only to preserve compatibility with previous scripts.
- No pretrained weights are included. Train your own checkpoints from the downloaded datasets.
- Reported numbers can vary with GPU, PyTorch version, split settings, seeds, and dataset preprocessing.

## Citation

If this repository is useful for your research, please cite the corresponding RFE-SFNet paper when it becomes available.

## License

This project is released under the MIT License.
