# Reproduction

Run commands from the repository root. Download each dataset's released derivatives into the default `data/raw/` directories, or edit the input paths in a copy of the relevant configuration. Dataset links and licenses are in the README. The preparation scripts consume the dataset authors' epoch arrays, beamforming outputs and head models; they do not reconstruct these derivatives from continuous recordings. Downloads are manual.

## Reported experiment settings

`configs/` records the settings used for the reported comparisons. Inspect a command before running it:

```bash
python run_reported.py configs/ds004752_neural_reported.json --dry-run
python run_reported.py configs/ds004752_neural_reported.json
```

The runner passes every non-null configuration argument to its experiment. Each configuration has a separate output directory under `results/`. `device: auto` selects available hardware; recorded neural runs used CUDA. For paper comparisons, use these configurations rather than the experiment scripts' exploratory defaults.

| Configuration | Analysis |
| --- | --- |
| `ds004752_ridge.json` | Low-frequency ridge, alpha 1000 |
| `ds004752_prior_pearson.json`, `ds004752_prior_r2.json` | Pearson- and R²-oriented priors |
| `ds004752_residual_pearson.json`, `ds004752_residual_r2.json` | Corresponding linear residual corrections |
| `ds004752_neural.json` | Low-frequency empirical-operator neural ablation |
| `ds004752_translation.json` | Translation autoencoder |
| `ds004752_neural_reported.json` | EEGNet-style, TCN, masked and FNO comparisons |
| `ds004752_broadband_{mlp,shared,gated}.json` | Separate supplementary broadband ablation |
| `localize_nested.json` | Nested logistic and temperature selection |
| `localize_neural.json`, `localize_ctnet.json` | Scalp neural comparisons |
| `spes_primary.json` | Coordinate ridge and additional tree diagnostic |
| `spes_retrieval.json`, `spes_calibration.json` | Fixed retrieval and its calibration sensitivity |

The ds004752 encoder configuration fixes five supervised epochs, patience two, two masked-pretraining epochs, seed 29, and no trial subsampling. Neural fitting and scaling use fitting participants, with one separate validation participant. The ds004752 validation rule chooses the next participant in cyclic sorted order after the test participant; Localize-MI neural validation uses the outer-fold index modulo the six sorted training participants. Configuration selection for ds004752 remains exploratory rather than nested.

## ds004752: beamforming derivatives to trial cache

The input directory contains `sub-*/beamforming/*-task-verbalWM-LCMVsources.mat` files.

```bash
python preprocess/ds_cache.py --dataset-root data/raw/ds004752/derivatives
python preprocess/ds_filter.py
python run_reported.py configs/ds004752_ridge.json
python run_reported.py configs/ds004752_prior_pearson.json
python run_reported.py configs/ds004752_residual_pearson.json
```

The first command writes trial arrays, channel labels, cache metadata and participant-held-out split manifests. The second writes the 1–12 Hz versions. Both encoding and memory maintenance are evaluated. The targets are EEG-derived LCMV virtual sensors. The neural forward-consistency ablation is separate from the linear residual correction.

## Localize-MI: epoch derivatives to native-coordinate features

Place the released dataset in `data/raw/localize-mi`, retaining these paths:

- `derivatives/epochs/sub-*/eeg/*_epochs.npy` and matching `_epochs.tsv`, `_epochs.json`, `_channels.tsv` files;
- `derivatives/epochs/sub-*/ieeg/*space-{surface,T1w,MNI152NLin2009aSym}_electrodes.tsv`;
- `derivatives/sourcemodelling/sub-*/fwd/sub-*_fwd.fif` and `xfm/sub-*_from-head_to-surface.h5`.

```bash
python preprocess/localize_manifest.py
python preprocess/localize_features.py
python run_reported.py configs/localize_nested.json
python run_reported.py configs/localize_neural.json
python run_reported.py configs/localize_ctnet.json
```

`localize_manifest.py` scans the epoch derivatives, parses stimulation-contact pairs and computes their midpoints in each supplied coordinate frame. The primary feature builder uses the native surface-MRI frame. Its extraction window is 0–10 ms, ridge lambda is 1e-6 and softmax temperature is 3. The scalp vector uses forward-model channel order, zero-filling excluded channels. Labels use the outer-training median native y-coordinate, fixed during inner selection.

| Manifest field | Meaning |
| --- | --- |
| `participant_id`, `run_id`, `trial_type` | Group and stimulation identifiers |
| `eeg_epochs_npy`, `eeg_channels_tsv` | Dataset-relative input paths |
| `sampling_frequency_hz`, `zero_time_s` | Epoch time-axis metadata |
| `surface_mid_x`, `surface_mid_y`, `surface_mid_z` | Native surface-MRI stimulation midpoint |
| `has_surface_midpoint` | Whether both contacts have surface coordinates |

The builder writes `bundle_manifest.json` and `feature_bundle.npz`. Required arrays are `participant_id`, `run_id`, `trial_type`, `true_coord`, `scalp_feature` and `peak_coord`. Coordinate arrays have shape `(n_runs, 3)`. The JSON maps `scalp_lowfreq` to `scalp_feature`, `reconstructed_source_lowfreq` to `peak_coord`, and `oracle_source_lowfreq` to `true_coord`. These retained schema names denote feature groups; the oracle is a true-coordinate diagnostic, not an intracranial signal.

The primary output model is `nested_logistic_temperature`; fixed-head variants are diagnostics. Five held-out participants have single-class labels, so NLL and Brier are primary. Neural comparators use a different 0–50 ms input window with 20 bins and seeds 11, 29 and 47. The CTNet-like variant is a compact comparator inspired by [Zhao et al. (2024)](https://doi.org/10.1038/s41598-024-71118-7).

## SPES: epoch derivatives to reported RMS features

Place the released derivatives in `data/raw/spes-hd-eeg-seeg`. The manifest generator reads each EEG epoch array and its matching epoch/channel TSVs, and looks up stimulation coordinates in the participant's T1w and MNI electrode tables.

```bash
python preprocess/spes_manifest.py
python preprocess/spes_primary.py
python run_reported.py configs/spes_primary.json
python evaluation/spes_intervals.py --predictions results/spes_primary/prediction_rows.tsv
```

For each participant, retain the intersection of good channels across included runs. Average epochs, subtract the −0.2–0 s baseline, retain 5–50 ms, take every fifth sample, and compute per-channel RMS. **The reported analysis then centers each run's RMS vector across channels and normalizes it to unit Euclidean norm when nonzero.** No spatial smoothing is applied.

The archived 323-run, 35-feature matrix matches this normalized-input pathway. An earlier manuscript/repository description omitted centering and normalization; unnormalized RMS does not reproduce those inputs. The legacy filename `spes_raw_feature_manifest.tsv` is retained as an interface name, not as an indication that amplitudes are unnormalized. New feature manifests record `feature_normalization` explicitly. `--preserve-amplitude` in `spes_cache.py` is available for a separate unnormalized diagnostic and is not used by the reported configuration.

| Manifest field | Meaning |
| --- | --- |
| `participant_id`, `run_id`, `trial_type` | Group and stimulation identifiers |
| `eeg_epochs_npy`, `eeg_channels_tsv` | Dataset-relative input paths |
| `eeg_sampling_frequency`, `zero_time_s` | Epoch time-axis metadata |
| `t1w_x`, `t1w_y`, `t1w_z` | Native T1w stimulation coordinates, meters |
| `mni_x`, `mni_y`, `mni_z` | MNI stimulation coordinates, meters |

The generated feature manifest adds `feature_npy`, `common_good_channels_txt` and `feature_normalization`. Feature paths are relative to the repository root; the common-channel file specifies vector order. The estimator converts MNI targets to millimeters and builds 35 field/geometry features from the normalized vector. Its spherical radius is the clipped order statistic at rank `min(n, ceil((n + 1) * coverage))` of the selected inner-OOF errors. Those predictions serve both ridge selection and radius calibration; this is not CV+ or independent split conformal calibration.

### Fixed-retrieval sensitivity

The supplementary predictor performs leave-one-run-out retrieval within each participant, using normalized RMS vectors, soft kNN with k=3 and temperature 5:

```bash
python preprocess/spes_cache.py --feature-mode rms_topography
python run_reported.py configs/spes_retrieval.json
python run_reported.py configs/spes_calibration.json
```

Calibration resplits participants around these fixed predictions over 500 repetitions, with fractions 0.3, 0.5 and 0.7 and seed 34170. It does not refit the primary coordinate-ridge model. Its T1w error and spread use millimeters; it is a supplementary sensitivity, not participant-held-out retrieval training.

## Expected results and tests

[expected_metrics.json](expected_metrics.json) records published aggregate scores. Tolerances describe the rounding precision of the paper, not a guarantee that fresh neural training reproduces a value within that tolerance. Hardware and dependency backends can affect training. No saved per-run predictions are distributed.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Synthetic tests check participant separation, training-only scaling, proper probability scores, conformal order statistics, native-coordinate labels, channel intersection, RMS normalization, all reported argument sets and command-line entry points. GitHub Actions runs them with Python 3.12 and CPU PyTorch.

Input preparation was additionally checked privately against the existing 61-run Localize-MI and 323-run SPES manifests. SPES preprocessing was rerun for seven recorded runs; its vectors matched the archived inputs. The full 323-run fixed feature matrix was numerically compared as well. These checks do not constitute full model retraining from the original recordings.

## Remaining coverage

The CBraMod pretrained/random one-patch diagnostic is not included. Its encoder/checkpoint-loading code and pretrained weights require a separate setup; the reported 0.707/0.708 NLL values are not reproduced by these scripts. All three primary estimators, their main reported neural comparators, the supplementary broadband controls and fixed-retrieval sensitivity have code and configuration entry points here. Figure generation, dataset-author derivative construction and frozen-output verification remain outside the code-only release.

[v1.0](https://github.com/jiwoonl/EEG-Source-Imaging-Evaluation/releases/tag/v1.0) preserves the original code snapshot. The input-preparation tools, reported configurations and CI were added subsequently on `main`; the v1.0 tag is unchanged.
