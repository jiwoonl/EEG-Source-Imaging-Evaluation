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

[Reproduction instructions](docs/reproduction.md) describe dataset preparation, input fields and reported settings. Download each dataset's released derivatives before running these commands:

```bash
# ds004752
python preprocess/ds_cache.py --dataset-root data/raw/ds004752/derivatives
python preprocess/ds_filter.py
python run_reported.py configs/ds004752_neural_reported.json --dry-run

# Localize-MI
python preprocess/localize_manifest.py
python preprocess/localize_features.py
python run_reported.py configs/localize_nested.json

# SPES
python preprocess/spes_manifest.py
python preprocess/spes_primary.py
python run_reported.py configs/spes_primary.json
```

`configs/` fixes the reported model arguments, seeds and training budgets. `--dry-run` prints the command without fitting a model. Outputs go under `results/`. The scripts start from the dataset authors' released derivatives; downloading the repository alone does not reproduce the paper.

The recorded SPES inputs use channel-centered, unit-norm RMS topographies. The [reproduction notes](docs/reproduction.md#spes-epoch-derivatives-to-reported-rms-features) correct the earlier raw-amplitude description. Model training has not been rerun in this update.

## Tests and versions

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

GitHub Actions checks synthetic preprocessing, participant splits, metrics, reported configurations and CLI imports. [Expected aggregate scores](docs/expected_metrics.json) are provided with their rounding precision. The CBraMod pretrained/random diagnostic remains outside this repository's reproduction coverage.

[v1.1](https://github.com/jiwoonl/EEG-Source-Imaging-Evaluation/releases/tag/v1.1) adds input preparation, fixed configurations, automated checks and the SPES preprocessing clarification. The original [v1.0](https://github.com/jiwoonl/EEG-Source-Imaging-Evaluation/releases/tag/v1.0) snapshot is preserved.

## Interpretation

The ds004752 target is derived from scalp EEG, so proxy agreement does not measure recovery of independently recorded intracranial signals. Its explored hyperparameter choices are not a nested evaluation. Localize-MI labels use native surface-MRI coordinates, and the BEM-informed predictor has participant-specific anatomical information that the scalp comparator lacks; NLL and Brier score are primary, with pooled AUROC secondary. The SPES procedure reuses inner out-of-fold predictions for penalty selection and radius calibration; report observed coverage together with radius and participant-level variability.

## Data and citation

Obtain data from their original repositories and follow their terms:

- [ds004752, version 1.0.1](https://doi.org/10.18112/openneuro.ds004752.v1.0.1): CC0.
- [Localize-MI](https://doi.org/10.12751/g-node.1cc1ae): CC BY-NC-SA 4.0.
- [SPES](https://osf.io/wsgzp/): CC BY-NC-SA 4.0.

Paper metadata is in [CITATION.cff](CITATION.cff). An arXiv identifier will be added after the preprint is available. The [BSD 3-Clause license](LICENSE) applies to this repository's software; it does not replace the datasets' licenses.
