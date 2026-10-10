# EEG Source Imaging Evaluation

Analysis code for *EEG Source Imaging under an Ill-Posed Inverse Problem: Proxy Agreement, Individual Head Models, and Localization Uncertainty*.

The study evaluates three separately fitted estimators:

| Dataset | Evaluation | Main scripts |
| --- | --- | --- |
| ds004752 | Agreement with an EEG-derived LCMV source proxy | `experiments/ds_prior.py`, `experiments/ds_residual.py` |
| Localize-MI | Geometry-assisted coordinate classification | `experiments/localize_nested.py` |
| SPES | Spatial prediction and residual-quantile uncertainty | `experiments/spes_localization.py` |

This repository contains analysis, preprocessing and evaluation code. Original recordings, derived datasets, saved predictions, manuscript files, figures and artwork are not included.

## Installation

The code was checked with Python 3.12. Install the analysis dependencies, including MNE and h5py used by the Localize-MI helpers:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/localize_nested.py --help
```

Neural comparisons also require PyTorch:

```bash
python -m pip install -r requirements-training.txt
```

`requirements-training.txt` specifies the PyTorch version used for validation. Choose a compatible PyTorch build for your CPU or CUDA installation.

## Running the analyses

Download the relevant dataset and prepare its input manifests before running an experiment. [Reproduction instructions](docs/reproduction.md) describe the required inputs, commands and reported settings. Each command-line script also accepts `--help`.

```bash
# ds004752: prepare the common-label cache and its 1–12 Hz version.
python preprocess/ds_cache.py --dataset-root /path/to/ds004752/derivatives
python preprocess/ds_filter.py

# Localize-MI: evaluate an already prepared feature bundle.
python experiments/localize_nested.py \
  --feature-bundle data/derived/localize_features --n-bootstrap 10000

# SPES: evaluate prepared run-level RMS features and electrode coordinates.
python experiments/spes_localization.py \
  --feature-manifest data/derived/spes_raw_feature_manifest.tsv \
  --dataset-root /path/to/spes-hd-eeg-seeg
```

Outputs are written under `results/`. These examples require local inputs; cloning the repository alone does not reproduce the reported scores. Full training from the original recordings has not been rerun as part of this code release.

## Interpretation

The ds004752 target is derived from scalp EEG, so proxy agreement does not measure recovery of independently recorded intracranial signals. Its explored hyperparameter choices are not a nested evaluation. Localize-MI labels use native surface-MRI coordinates, and the BEM-informed predictor has participant-specific anatomical information that the scalp comparator lacks; NLL and Brier score are primary, with pooled AUROC secondary. The SPES procedure reuses inner out-of-fold predictions for penalty selection and radius calibration; report observed coverage together with radius and participant-level variability.

## Data and citation

Obtain data from their original repositories and follow their terms:

- [ds004752, version 1.0.1](https://doi.org/10.18112/openneuro.ds004752.v1.0.1): CC0.
- [Localize-MI](https://doi.org/10.12751/g-node.1cc1ae): CC BY-NC-SA 4.0.
- [SPES](https://osf.io/wsgzp/): CC BY-NC-SA 4.0.

Paper metadata is in [CITATION.cff](CITATION.cff). An arXiv identifier will be added after the preprint is available. The [BSD 3-Clause license](LICENSE) applies to this repository's software; it does not replace the datasets' licenses.
