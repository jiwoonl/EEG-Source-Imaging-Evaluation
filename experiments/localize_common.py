#!/usr/bin/env python3
"""Feature extraction and probability metrics for Localize-MI coordinate labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import mne
import numpy as np

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


COORD_FIELDS = {
    "surface": ("surface_mid_x", "surface_mid_y", "surface_mid_z"),
    "t1w": ("t1w_mid_x", "t1w_mid_y", "t1w_mid_z"),
    "mni": ("mni_mid_x", "mni_mid_y", "mni_mid_z"),
}
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
DEFAULT_FEATURE_SETS = {
    "scalp_lowfreq": "scalp_feature",
    "reconstructed_source_lowfreq": "peak_coord",
    "oracle_source_lowfreq": "true_coord",
}


FeatureRow = dict[str, Any]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    positive = logits >= 0
    negative = ~positive
    output = np.empty_like(logits, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[negative])
    output[negative] = exp_logits / (1.0 + exp_logits)
    return output


def load_affine(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["trans"], dtype=np.float64)


def apply_affine(coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    hom = np.concatenate(
        [coords, np.ones((coords.shape[0], 1), dtype=np.float64)], axis=1
    )
    transformed = hom @ affine.T
    return transformed[:, :3]


def build_source_space(
    dataset_root: Path, participant_id: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    fwd_path = (
        dataset_root
        / f"derivatives/sourcemodelling/{participant_id}/fwd/{participant_id}_fwd.fif"
    )
    xfm_path = (
        dataset_root
        / f"derivatives/sourcemodelling/{participant_id}/xfm/{participant_id}_from-head_to-surface.h5"
    )
    fwd = mne.read_forward_solution(fwd_path, verbose="error")
    ch_names = list(fwd["info"]["ch_names"])
    gain = np.asarray(fwd["sol"]["data"], dtype=np.float64)
    source_coords_head = np.concatenate(
        [np.asarray(src["rr"][src["vertno"]], dtype=np.float64) for src in fwd["src"]],
        axis=0,
    )
    coords_surface = apply_affine(source_coords_head, load_affine(xfm_path))
    gain_blocks = gain.reshape(gain.shape[0], coords_surface.shape[0], 3)
    return ch_names, gain_blocks, coords_surface


def localization_scores(
    gain_blocks: np.ndarray, scalp_topography: np.ndarray, ridge_lambda: float
) -> np.ndarray:
    projection = np.einsum("csk,c->sk", gain_blocks, scalp_topography)
    gram = np.einsum("csi,csj->sij", gain_blocks, gain_blocks)
    gram += ridge_lambda * np.eye(3, dtype=np.float64)[None, :, :]
    inv_gram = np.linalg.inv(gram)
    return np.einsum("si,sij,sj->s", projection, inv_gram, projection)


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = temperature * (values - np.max(values))
    weights = np.exp(scaled)
    denom = float(np.sum(weights))
    if denom == 0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return weights / denom


def pooled_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    if np.unique(labels).size < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def binary_f1(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    labels = labels.astype(np.int64)
    pred = (scores >= threshold).astype(np.int64)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def binary_nll(labels: np.ndarray, scores: np.ndarray, epsilon: float = 1e-8) -> float:
    labels = labels.astype(np.float64)
    clipped = np.clip(scores.astype(np.float64), epsilon, 1.0 - epsilon)
    return float(
        -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    )


def binary_brier(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.float64)
    scores = scores.astype(np.float64)
    return float(np.mean((scores - labels) ** 2))


def binary_ece(labels: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    labels = labels.astype(np.float64)
    scores = scores.astype(np.float64)
    if labels.size == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        if bin_index == n_bins - 1:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        if not np.any(mask):
            continue
        confidence = float(np.mean(scores[mask]))
        accuracy = float(np.mean(labels[mask]))
        weight = float(np.mean(mask))
        ece += weight * abs(accuracy - confidence)
    return float(ece)


def fit_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    classifier_c: float,
    head_type: str,
) -> np.ndarray:
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_x)
    test_z = scaler.transform(test_x)
    if head_type == "logistic":
        classifier = LogisticRegression(
            C=classifier_c,
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        )
        classifier.fit(train_z, train_y)
        return classifier.predict_proba(test_z)[:, 1]
    if head_type == "ridge_det":
        classifier = RidgeClassifier(
            alpha=max(1.0 / max(classifier_c, 1e-8), 1e-8),
            class_weight="balanced",
            random_state=0,
        )
        classifier.fit(train_z, train_y)
        decision = classifier.decision_function(test_z)
        return stable_sigmoid(np.asarray(decision, dtype=np.float64))
    raise ValueError(f"Unsupported head type: {head_type}")


def build_features(
    *,
    manifest_rows: list[dict[str, str]],
    dataset_root: Path,
    coordinate_space: str,
    post_start_s: float,
    post_end_s: float,
    ridge_lambda: float,
    softmax_temperature: float,
) -> list[FeatureRow]:
    coord_fields = COORD_FIELDS[coordinate_space]
    feature_rows: list[FeatureRow] = []
    for participant_id in sorted({row["participant_id"] for row in manifest_rows}):
        ch_names, gain_blocks_full, coords_surface = build_source_space(
            dataset_root, participant_id
        )
        fwd_index_by_name = {name: idx for idx, name in enumerate(ch_names)}
        subject_rows = [
            row for row in manifest_rows if row["participant_id"] == participant_id
        ]
        for row in subject_rows:
            channel_rows = read_tsv(dataset_root / row["eeg_channels_tsv"])
            epoch_array = np.load(dataset_root / row["eeg_epochs_npy"], mmap_mode="r")
            evoked = np.asarray(epoch_array.mean(axis=0), dtype=np.float64)

            channel_pairs: list[tuple[int, int]] = []
            for epoch_idx, channel_row in enumerate(channel_rows):
                channel_name = channel_row["name"]
                if channel_row.get("status", "good") != "good":
                    continue
                if channel_name not in fwd_index_by_name:
                    continue
                channel_pairs.append((epoch_idx, fwd_index_by_name[channel_name]))

            epoch_indices = np.asarray(
                [pair[0] for pair in channel_pairs], dtype=np.int64
            )
            fwd_indices = np.asarray(
                [pair[1] for pair in channel_pairs], dtype=np.int64
            )
            fs = float(row["sampling_frequency_hz"])
            zero_time = float(row["zero_time_s"])
            time_axis = np.arange(evoked.shape[1], dtype=np.float64) / fs - zero_time
            time_mask = (time_axis >= post_start_s) & (time_axis <= post_end_s)
            candidate_indices = np.where(time_mask)[0]
            if candidate_indices.size == 0:
                candidate_indices = np.where(time_axis >= max(post_start_s, 0.0))[0]

            evoked_good = evoked[epoch_indices]
            gfp = evoked_good[:, candidate_indices].std(axis=0)
            peak_index = int(candidate_indices[int(np.argmax(gfp))])
            scalp_topography = evoked_good[:, peak_index].astype(np.float64)
            scalp_topography = scalp_topography - scalp_topography.mean()
            scalp_norm = float(np.linalg.norm(scalp_topography))
            if scalp_norm > 0:
                scalp_topography /= scalp_norm

            full_scalp = np.zeros(len(ch_names), dtype=np.float64)
            for local_idx, (_, fwd_idx) in enumerate(channel_pairs):
                full_scalp[fwd_idx] = scalp_topography[local_idx]

            gain_blocks = gain_blocks_full[fwd_indices]
            scores = localization_scores(gain_blocks, scalp_topography, ridge_lambda)
            peak_coord_surface = coords_surface[int(np.argmax(scores))]
            weights = softmax(scores, softmax_temperature)
            expected_coord_surface = np.sum(coords_surface * weights[:, None], axis=0)
            spatial_sq = np.sum(
                (coords_surface - expected_coord_surface[None, :]) ** 2, axis=1
            )
            spatial_spread = float(np.sqrt(np.sum(weights * spatial_sq)))
            weight_entropy = float(
                -(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum()
            )
            top1_weight = float(np.max(weights))
            peak_score = float(np.max(scores))
            peak_expected_gap = peak_coord_surface - expected_coord_surface

            true_coord = np.asarray(
                [float(row[field]) for field in coord_fields], dtype=np.float64
            )
            feature_rows.append(
                {
                    "participant_id": participant_id,
                    "run_id": row["run_id"],
                    "trial_type": row["trial_type"],
                    "scalp_feature": full_scalp,
                    "peak_coord": peak_coord_surface,
                    "expected_coord": expected_coord_surface,
                    "peak_score": np.asarray([peak_score], dtype=np.float64),
                    "top1_weight": np.asarray([top1_weight], dtype=np.float64),
                    "score_entropy": np.asarray([weight_entropy], dtype=np.float64),
                    "spatial_spread": np.asarray([spatial_spread], dtype=np.float64),
                    "peak_expected_gap": peak_expected_gap.astype(np.float64),
                    "true_coord": true_coord,
                }
            )
    return feature_rows


def save_feature_bundle(
    bundle_dir: Path,
    feature_rows: list[FeatureRow],
    feature_sets: dict[str, str],
    metadata: dict[str, object] | None = None,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}
    feature_keys = sorted({"true_coord", *feature_sets.values()})
    arrays: dict[str, np.ndarray] = {
        "participant_id": np.asarray(
            [str(row["participant_id"]) for row in feature_rows]
        ),
        "run_id": np.asarray([str(row["run_id"]) for row in feature_rows]),
        "trial_type": np.asarray([str(row["trial_type"]) for row in feature_rows]),
        "true_coord": np.stack(
            [np.asarray(row["true_coord"], dtype=np.float64) for row in feature_rows],
            axis=0,
        ),
    }
    for key in feature_keys:
        if key == "true_coord":
            continue
        arrays[key] = np.stack(
            [np.asarray(row[key], dtype=np.float64) for row in feature_rows], axis=0
        )
    np.savez_compressed(bundle_dir / "feature_bundle.npz", **arrays)
    manifest = {
        "bundle_name": bundle_dir.name,
        "n_rows": len(feature_rows),
        "feature_sets": feature_sets,
        "available_feature_keys": sorted(arrays.keys()),
        "metadata": metadata,
    }
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def load_feature_bundle(bundle_dir: Path) -> tuple[list[FeatureRow], dict[str, object]]:
    manifest = json.loads(
        (bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    npz = np.load(bundle_dir / "feature_bundle.npz")
    n_rows = int(manifest["n_rows"])
    feature_rows: list[FeatureRow] = []
    numeric_keys = [
        key
        for key in manifest.get("available_feature_keys", [])
        if key not in {"participant_id", "run_id", "trial_type"}
    ]
    for idx in range(n_rows):
        row: FeatureRow = {
            "participant_id": str(npz["participant_id"][idx]),
            "run_id": str(npz["run_id"][idx]),
            "trial_type": str(npz["trial_type"][idx]),
        }
        for key in numeric_keys:
            row[key] = np.asarray(npz[key][idx], dtype=np.float64)
        feature_rows.append(row)
    return feature_rows, manifest


def infer_features(
    manifest: dict[str, object],
    *,
    scalp_key: str | None,
    reconstructed_key: str | None,
    oracle_key: str | None,
) -> dict[str, str]:
    manifest_sets = manifest.get("feature_sets", {})
    inferred = {
        "scalp_lowfreq": scalp_key
        or manifest_sets.get("scalp_lowfreq")
        or "scalp_feature",
        "reconstructed_source_lowfreq": reconstructed_key
        or manifest_sets.get("reconstructed_source_lowfreq"),
        "oracle_source_lowfreq": oracle_key
        or manifest_sets.get("oracle_source_lowfreq")
        or "true_coord",
    }
    if inferred["reconstructed_source_lowfreq"] is None:
        raise ValueError(
            "No reconstructed feature key provided and none found in the bundle manifest."
        )
    return inferred


def evaluate_feature_rows(
    feature_rows: list[FeatureRow],
    *,
    coordinate_space: str,
    label_axis: str,
    feature_mode: str,
    feature_sets: dict[str, str],
    classifier_c: float,
    head_type: str,
    label_threshold_mode: str,
    negative_control: str,
    random_seed: int,
) -> dict[str, object]:
    axis_index = AXIS_INDEX[label_axis]
    subjects = sorted({str(row["participant_id"]) for row in feature_rows})
    prediction_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_seed)

    for holdout_subject in subjects:
        train_rows = [
            row for row in feature_rows if row["participant_id"] != holdout_subject
        ]
        test_rows = [
            row for row in feature_rows if row["participant_id"] == holdout_subject
        ]
        if label_threshold_mode == "train_fold":
            label_threshold = float(
                np.median([row["true_coord"][axis_index] for row in train_rows])
            )
        else:
            label_threshold = float(
                np.median([row["true_coord"][axis_index] for row in feature_rows])
            )
        y_train = np.asarray(
            [
                int(row["true_coord"][axis_index] >= label_threshold)
                for row in train_rows
            ],
            dtype=np.int64,
        )
        y_test = np.asarray(
            [
                int(row["true_coord"][axis_index] >= label_threshold)
                for row in test_rows
            ],
            dtype=np.int64,
        )
        for feature_set, feature_key in feature_sets.items():
            train_x = np.asarray(
                [row[feature_key] for row in train_rows], dtype=np.float64
            )
            test_x = np.asarray(
                [row[feature_key] for row in test_rows], dtype=np.float64
            )
            train_y = y_train.copy()
            if negative_control == "shuffle_labels":
                train_y = rng.permutation(train_y)
            elif negative_control == "shuffle_alignment":
                train_x = train_x[rng.permutation(train_x.shape[0])]

            scores = fit_classifier(
                train_x,
                train_y,
                test_x,
                classifier_c=classifier_c,
                head_type=head_type,
            )
            for row, score, label in zip(test_rows, scores, y_test):
                prediction_rows.append(
                    {
                        "participant_id": holdout_subject,
                        "run_id": row["run_id"],
                        "trial_type": row["trial_type"],
                        "feature_set": feature_set,
                        "feature_key": feature_key,
                        "label": int(label),
                        "score": float(score),
                        "label_threshold": label_threshold,
                        "negative_control": negative_control,
                        "head_type": head_type,
                    }
                )
            fold_metric_rows.append(
                {
                    "holdout_subject": holdout_subject,
                    "feature_set": feature_set,
                    "feature_key": feature_key,
                    "head_type": head_type,
                    "n_test_runs": int(test_x.shape[0]),
                    "positive_prevalence": float(np.mean(y_test)),
                    "label_threshold": label_threshold,
                    "auroc": (
                        pooled_auroc(y_test, scores)
                        if np.unique(y_test).size == 2
                        else ""
                    ),
                    "f1": binary_f1(y_test, scores),
                    "nll": binary_nll(y_test, scores),
                    "brier": binary_brier(y_test, scores),
                    "ece": binary_ece(y_test, scores),
                }
            )

    summary_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "dataset_name": "localize_mi",
        "label_name": f"localize_mi_{coordinate_space}_{label_axis}_high_vs_low",
        "feature_mode": feature_mode,
        "classifier_c": classifier_c,
        "coordinate_space": coordinate_space,
        "label_axis": label_axis,
        "label_threshold_mode": label_threshold_mode,
        "source_feature": feature_sets["reconstructed_source_lowfreq"],
        "head_type": head_type,
        "negative_control": negative_control,
        "random_seed": random_seed,
        "n_subjects": len(subjects),
        "n_runs": len(feature_rows),
        "feature_sets": feature_sets,
    }

    for feature_set, feature_key in feature_sets.items():
        feature_predictions = [
            row for row in prediction_rows if row["feature_set"] == feature_set
        ]
        feature_fold_rows = [
            row for row in fold_metric_rows if row["feature_set"] == feature_set
        ]
        labels = np.asarray(
            [int(row["label"]) for row in feature_predictions], dtype=np.int64
        )
        scores = np.asarray(
            [float(row["score"]) for row in feature_predictions], dtype=np.float64
        )
        fold_aurocs = [
            float(row["auroc"]) for row in feature_fold_rows if row["auroc"] != ""
        ]
        pooled_summary = {
            "feature_key": feature_key,
            "mean_auroc": pooled_auroc(labels, scores),
            "std_auroc": float(np.std(fold_aurocs)) if fold_aurocs else 0.0,
            "mean_f1": binary_f1(labels, scores),
            "std_f1": float(np.std([float(row["f1"]) for row in feature_fold_rows])),
            "mean_nll": binary_nll(labels, scores),
            "std_nll": float(np.std([float(row["nll"]) for row in feature_fold_rows])),
            "mean_brier": binary_brier(labels, scores),
            "std_brier": float(
                np.std([float(row["brier"]) for row in feature_fold_rows])
            ),
            "mean_ece": binary_ece(labels, scores),
            "std_ece": float(np.std([float(row["ece"]) for row in feature_fold_rows])),
            "mean_positive_prevalence": float(np.mean(labels)),
            "n_valid_fold_aurocs": len(fold_aurocs),
        }
        summary_rows.append({"feature_set": feature_set, **pooled_summary})
        summary[feature_set] = pooled_summary

    scalp_summary = summary["scalp_lowfreq"]
    recon_summary = summary["reconstructed_source_lowfreq"]
    oracle_summary = summary["oracle_source_lowfreq"]
    summary["reconstructed_minus_scalp_auroc"] = float(
        recon_summary["mean_auroc"]
    ) - float(scalp_summary["mean_auroc"])
    summary["reconstructed_minus_scalp_f1"] = float(recon_summary["mean_f1"]) - float(
        scalp_summary["mean_f1"]
    )

    comparison_rows = [
        {
            "comparison": "reconstructed_minus_scalp",
            "delta_auroc": summary["reconstructed_minus_scalp_auroc"],
            "delta_f1": summary["reconstructed_minus_scalp_f1"],
            "delta_nll": float(recon_summary["mean_nll"])
            - float(scalp_summary["mean_nll"]),
            "delta_brier": float(recon_summary["mean_brier"])
            - float(scalp_summary["mean_brier"]),
            "delta_ece": float(recon_summary["mean_ece"])
            - float(scalp_summary["mean_ece"]),
        },
        {
            "comparison": "oracle_minus_scalp",
            "delta_auroc": float(oracle_summary["mean_auroc"])
            - float(scalp_summary["mean_auroc"]),
            "delta_f1": float(oracle_summary["mean_f1"])
            - float(scalp_summary["mean_f1"]),
            "delta_nll": float(oracle_summary["mean_nll"])
            - float(scalp_summary["mean_nll"]),
            "delta_brier": float(oracle_summary["mean_brier"])
            - float(scalp_summary["mean_brier"]),
            "delta_ece": float(oracle_summary["mean_ece"])
            - float(scalp_summary["mean_ece"]),
        },
    ]

    return {
        "prediction_rows": prediction_rows,
        "fold_metric_rows": fold_metric_rows,
        "summary_rows": summary_rows,
        "comparison_rows": comparison_rows,
        "summary": summary,
    }


def write_outputs(results_dir: Path, evaluation: dict[str, object]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = evaluation["prediction_rows"]
    fold_metric_rows = evaluation["fold_metric_rows"]
    summary_rows = evaluation["summary_rows"]
    comparison_rows = evaluation["comparison_rows"]
    summary = evaluation["summary"]
    write_tsv(
        results_dir / "prediction_rows.tsv",
        list(prediction_rows[0].keys()),
        prediction_rows,
    )
    write_tsv(
        results_dir / "fold_metrics.tsv",
        list(fold_metric_rows[0].keys()),
        fold_metric_rows,
    )
    write_tsv(results_dir / "summary.tsv", list(summary_rows[0].keys()), summary_rows)
    write_tsv(
        results_dir / "comparison.tsv", list(comparison_rows[0].keys()), comparison_rows
    )
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (results_dir / "threshold_assessment.md").write_text(
        threshold_summary(summary), encoding="utf-8"
    )


def threshold_summary(summary: dict[str, object]) -> str:
    lines = [
        "# Localize-MI Anatomical Downstream Benchmark",
        "",
        f"- label: `{summary['label_name']}`",
        f"- scalp AUROC: `{summary['scalp_lowfreq']['mean_auroc']:.3f}`",
        f"- reconstructed AUROC: `{summary['reconstructed_source_lowfreq']['mean_auroc']:.3f}`",
        f"- oracle AUROC: `{summary['oracle_source_lowfreq']['mean_auroc']:.3f}`",
        f"- reconstructed minus scalp AUROC: `{summary['reconstructed_minus_scalp_auroc']:.3f}`",
        f"- reconstructed minus scalp F1: `{summary['reconstructed_minus_scalp_f1']:.3f}`",
        f"- reconstructed NLL: `{summary['reconstructed_source_lowfreq']['mean_nll']:.3f}` vs scalp `{summary['scalp_lowfreq']['mean_nll']:.3f}`",
        f"- reconstructed ECE: `{summary['reconstructed_source_lowfreq']['mean_ece']:.3f}` vs scalp `{summary['scalp_lowfreq']['mean_ece']:.3f}`",
        "",
        "Interpretation:",
        "",
        "- this is an external anatomical downstream benchmark rather than a behavioral ds004752 task",
        "- AUROC is computed on pooled held-out predictions across LOSO subject folds",
        "- it tests whether source-space localization features carry more useful anatomical information than scalp topography alone",
        f"- label threshold mode: `{summary['label_threshold_mode']}`",
        f"- negative control mode: `{summary['negative_control']}`",
    ]
    return "\n".join(lines) + "\n"
