# Reproduction

Run commands from the repository root. Paths in manifests should refer to your own dataset layout. The code does not download recordings or generate the initial dataset inventory automatically. Derived inputs and saved predictions remain author-held.

## ds004752

`preprocess/ds_cache.py` reads the dataset's `sub-*/beamforming/*-task-verbalWM-LCMVsources.mat` derivatives and writes trial arrays, cache metadata and participant-held-out split manifests. `preprocess/ds_filter.py` produces the 1–12 Hz cache. These are EEG-derived source targets.

```bash
python preprocess/ds_cache.py --dataset-root /path/to/ds004752/derivatives
python preprocess/ds_filter.py
python experiments/ds_prior.py \
  --operator-mode subject_bank_mean --forward-ridge-alpha 100 --inverse-alpha 0.1
python experiments/ds_residual.py \
  --operator-mode subject_bank_mean \
  --base-forward-ridge-alpha 100 --base-inverse-alpha 0.1 \
  --residual-feature-mode scalp_plus_base \
  --residual-alpha 10000 --residual-weight 0.2
```

These prior/residual settings correspond to the primary proxy-correlation comparison. The additional R²-oriented residual configuration uses forward penalty 0.01, inverse penalty 1, residual penalty 10000 and residual weight 0.1. Configuration selection was exploratory across LOSO results; these commands do not turn it into nested selection. Scaling and operator fitting use the training participants within each fold.

| Script | Purpose |
| --- | --- |
| `experiments/ds_ridge.py` | Ridge comparator |
| `experiments/ds_prior.py` | Empirical operator-bank prior |
| `experiments/ds_residual.py` | Linear residual correction |
| `experiments/ds_neural.py` | MLP and forward-consistent neural ablations |
| `experiments/ds_translation.py` | Translation autoencoder |
| `experiments/ds_encoders.py` | EEGNet-style, TCN, masked-pretraining and FNO-style comparisons |

The neural residual/forward-consistency ablation is separate from the linear residual correction above. Use `--help` for each model's arguments; default model choices and training budgets are not necessarily those of every reported comparison.

## Localize-MI

`experiments/localize_common.py` contains source-location scoring, feature extraction and feature-bundle I/O. Feature extraction requires the original EEG epochs, channel metadata, forward solution and head-to-surface transform, together with a stimulation manifest. It is exposed as a Python API; an automatic raw-download-to-manifest pipeline is not included.

The primary bundle contains `bundle_manifest.json` and `feature_bundle.npz`. The NPZ arrays are `participant_id`, `run_id`, `trial_type`, `true_coord`, `scalp_feature` and `peak_coord`. Coordinate arrays have shape `(n_runs, 3)` in the participant's native surface-MRI frame. `scalp_feature` contains one channel-aligned vector per run. The JSON records `n_rows`, `available_feature_keys` and this mapping:

```json
{
  "scalp_lowfreq": "scalp_feature",
  "reconstructed_source_lowfreq": "peak_coord",
  "oracle_source_lowfreq": "true_coord"
}
```

These are retained input-schema names. In particular, `oracle_source_lowfreq` identifies the true-coordinate diagnostic, not measured intracranial activity. `save_feature_bundle()` writes the JSON and NPZ from feature rows. For the reported primary extraction, pass `coordinate_space="surface"`, `post_start_s=0.0`, `post_end_s=0.01`, `ridge_lambda=1e-6` and `softmax_temperature=3.0` to `build_features()`. Do not mix surface-coordinate features with labels from another frame.

```bash
python experiments/localize_nested.py \
  --feature-bundle data/derived/localize_features --label-axis y \
  --n-bootstrap 10000 --seed 34170
```

The outer-training median y-coordinate sets the binary labels and stays fixed during inner selection. Each outer fold selects logistic regularization, class weighting and temperature using training participants. In the output tables, `nested_logistic_temperature` is the primary calibrated comparison; fixed-head variants are diagnostics. Five held-out participants have single-class labels, so use NLL and Brier as primary scores.

`experiments/localize_neural.py` requires a stimulation manifest and epoch files. For the reported TCN and CTNet-like comparisons, use `--models tcn ctnet_like --seeds 11 29 47`, a 0–50 ms window, 20 temporal bins, hidden width 32, dropout 0.25, batch size 16, learning rate 0.0003, weight decay 0.001, up to 200 epochs and patience 25. The script defines these training settings; the input interval differs from the primary logistic comparator.

## SPES

The primary estimator requires per-run RMS topographies prepared from the original epochs: intersect good channels across each participant's runs, average epochs, subtract the −0.2–0 s baseline, retain 0.005–0.05 s, take every fifth sample, and compute RMS across time for each channel. Preserve these amplitudes without centering or unit-norm normalization. A dedicated raw-amplitude cache generator and the initial spatial-validation manifest are not included in this release.

`preprocess/spes_cache.py` is a separate retrieval-feature utility. It centers and normalizes each vector, so its output must not be substituted for the primary raw-RMS features. That utility belongs to the supplementary retrieval pathway.

```bash
python experiments/spes_localization.py \
  --feature-manifest data/derived/spes_raw_feature_manifest.tsv \
  --dataset-root /path/to/spes-hd-eeg-seeg \
  --coverage 0.9 --inner-splits 5 --seed 34170
```

The feature manifest needs `participant_id`, `run_id`, `feature_npy`, `common_good_channels_txt`, and `mni_x`, `mni_y`, `mni_z`; inspect `load_dataset()` for the full interface. MNI target coordinates are stored in meters and converted to millimeters by the estimator. `feature_npy` contains one RMS value per retained channel, ordered as the channel-name file. Manifest paths are resolved relative to `--repo-root`; electrode locations are read from `--dataset-root`.

The primary predictor constructs 35 geometry/amplitude features and selects a ridge penalty from `{0.01, 0.1, 1, 10, 100, 1000}`. Its radius is the clipped order statistic of selected inner-OOF Euclidean residuals, with rank `min(n, ceil((n + 1) * coverage))`. Model selection and radius estimation reuse the inner predictions. The refitted predictor is evaluated in outer participant-held-out folds; this is not CV+ or independent split conformal calibration.

`evaluation/spes_intervals.py` computes the primary ridge coverage and paired mean-error intervals from a locally supplied prediction TSV containing `participant_id`, `run_id`, `model`, `covered`, `error_mm` and `radius_mm`. It matches ridge and coordinate-mean rows by participant and run, retains all runs within each sampled participant, and uses 10,000 cluster bootstrap draws (seeds 34172 and 34270):

```bash
python evaluation/spes_intervals.py --predictions /path/to/prediction_rows.tsv
```

These are descriptive percentile intervals on held-out outcomes; overlapping LOSO training sets remain a limitation.

`experiments/spes_calibration.py` instead resplits participants around a fixed retrieval predictor. It requires a separate TSV with `participant_id`, `t1w_error_mm` and `raw_spatial_spread_t1w_mm`:

```bash
python experiments/spes_calibration.py \
  --run-metrics /path/to/fixed_retrieval_run_metrics.tsv \
  --n-repeats 500 --seed 34170
```

This fixed-predictor sensitivity does not reproduce the primary coordinate-ridge result. Its input predictions are not included, and the original fixed-retrieval predictor is outside this release.

## What was checked

The release was checked locally for Python syntax, command-line imports and numerical agreement with the archived analysis code on synthetic inputs. Those checks do not establish a fresh reproduction of the reported experiments. Exact dataset preprocessing inventories, frozen predictions and figure-generation assets are not part of this code-only release.
