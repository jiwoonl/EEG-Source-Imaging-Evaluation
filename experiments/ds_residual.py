#!/usr/bin/env python3
"""Run a residual-corrected source-imaging baseline on ds004752 low-frequency beamforming trials."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


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


def zscore_train(
    train: np.ndarray, other: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (other - mean) / std, mean, std


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = y_true.reshape(-1).astype(np.float64)
    b = y_pred.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = y_true.reshape(-1).astype(np.float64)
    pred = y_pred.reshape(-1).astype(np.float64)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = y_true.reshape(-1).astype(np.float64)
    pred = y_pred.reshape(-1).astype(np.float64)
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    denom = float(true.std())
    return 0.0 if denom == 0 else rmse / denom


def parse_float_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def fit_global_affine(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    pred = y_pred_train.reshape(-1).astype(np.float64)
    true = y_true_train.reshape(-1).astype(np.float64)
    pred_mean = float(pred.mean())
    true_mean = float(true.mean())
    pred_centered = pred - pred_mean
    true_centered = true - true_mean
    slope = float(
        np.dot(pred_centered, true_centered)
        / (np.dot(pred_centered, pred_centered) + ridge_alpha)
    )
    intercept = true_mean - slope * pred_mean
    return (slope * y_pred_test + intercept).astype(np.float32), {
        "global_slope": slope,
        "global_intercept": intercept,
    }


def fit_axis_affine(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
    *,
    mode: str,
    n_sources: int,
    n_timepoints: int,
    ridge_alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if mode == "output_affine":
        true = y_true_train.astype(np.float64)
        pred = y_pred_train.astype(np.float64)
        pred_test = y_pred_test.astype(np.float64)
        pred_mean = pred.mean(axis=0, keepdims=True)
        true_mean = true.mean(axis=0, keepdims=True)
        pred_centered = pred - pred_mean
        true_centered = true - true_mean
        denom = (
            np.sum(pred_centered * pred_centered, axis=0, keepdims=True) + ridge_alpha
        )
        slope = np.sum(pred_centered * true_centered, axis=0, keepdims=True) / denom
        intercept = true_mean - slope * pred_mean
        calibrated = (slope * pred_test) + intercept
        return calibrated.astype(np.float32), {
            "mean_slope": float(np.mean(slope)),
            "min_slope": float(np.min(slope)),
            "max_slope": float(np.max(slope)),
        }

    true = y_true_train.reshape(y_true_train.shape[0], n_sources, n_timepoints).astype(
        np.float64
    )
    pred = y_pred_train.reshape(y_pred_train.shape[0], n_sources, n_timepoints).astype(
        np.float64
    )
    pred_test = y_pred_test.reshape(
        y_pred_test.shape[0], n_sources, n_timepoints
    ).astype(np.float64)
    calibrated = np.empty_like(pred_test)
    slopes: list[float] = []
    for source_index in range(n_sources):
        source_pred = pred[:, source_index, :].reshape(-1)
        source_true = true[:, source_index, :].reshape(-1)
        pred_mean = float(source_pred.mean())
        true_mean = float(source_true.mean())
        pred_centered = source_pred - pred_mean
        true_centered = source_true - true_mean
        slope = float(
            np.dot(pred_centered, true_centered)
            / (np.dot(pred_centered, pred_centered) + ridge_alpha)
        )
        intercept = true_mean - slope * pred_mean
        calibrated[:, source_index, :] = (
            slope * pred_test[:, source_index, :]
        ) + intercept
        slopes.append(slope)
    return calibrated.reshape(y_pred_test.shape).astype(np.float32), {
        "mean_slope": float(np.mean(slopes)),
        "min_slope": float(np.min(slopes)),
        "max_slope": float(np.max(slopes)),
    }


def fit_shrinkage_gamma(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mean_ref: np.ndarray,
    gamma_grid: list[float],
) -> float:
    best_gamma = gamma_grid[0]
    best_mse = float("inf")
    for gamma in gamma_grid:
        candidate = mean_ref + gamma * (y_pred - mean_ref)
        mse = float(np.mean((y_true - candidate) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_gamma = gamma
    return float(best_gamma)


def fit_axis_shrinkage(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
    *,
    mode: str,
    n_sources: int,
    n_timepoints: int,
    gamma_grid: list[float],
) -> tuple[np.ndarray, dict[str, float]]:
    if mode == "global_shrinkage":
        mean_ref = np.asarray(y_true_train.mean(), dtype=np.float32)
        gamma = fit_shrinkage_gamma(y_true_train, y_pred_train, mean_ref, gamma_grid)
        return (mean_ref + gamma * (y_pred_test - mean_ref)).astype(np.float32), {
            "global_gamma": gamma
        }

    if mode == "output_shrinkage":
        mean_ref = y_true_train.mean(axis=0, keepdims=True)
        gammas = np.empty((1, y_true_train.shape[1]), dtype=np.float32)
        for output_index in range(y_true_train.shape[1]):
            gammas[0, output_index] = fit_shrinkage_gamma(
                y_true_train[:, output_index : output_index + 1],
                y_pred_train[:, output_index : output_index + 1],
                mean_ref[:, output_index : output_index + 1],
                gamma_grid,
            )
        return (mean_ref + gammas * (y_pred_test - mean_ref)).astype(np.float32), {
            "mean_gamma": float(np.mean(gammas)),
            "min_gamma": float(np.min(gammas)),
            "max_gamma": float(np.max(gammas)),
        }

    true = y_true_train.reshape(y_true_train.shape[0], n_sources, n_timepoints)
    pred = y_pred_train.reshape(y_pred_train.shape[0], n_sources, n_timepoints)
    pred_test = y_pred_test.reshape(y_pred_test.shape[0], n_sources, n_timepoints)
    mean_ref = true.mean(axis=(0, 2), keepdims=True)
    calibrated = np.empty_like(pred_test)
    gammas: list[float] = []
    for source_index in range(n_sources):
        gamma = fit_shrinkage_gamma(
            true[:, source_index, :],
            pred[:, source_index, :],
            mean_ref[:, source_index, :],
            gamma_grid,
        )
        calibrated[:, source_index, :] = mean_ref[:, source_index, :] + gamma * (
            pred_test[:, source_index, :] - mean_ref[:, source_index, :]
        )
        gammas.append(gamma)
    return calibrated.reshape(y_pred_test.shape).astype(np.float32), {
        "mean_gamma": float(np.mean(gammas)),
        "min_gamma": float(np.min(gammas)),
        "max_gamma": float(np.max(gammas)),
    }


def calibrate_prediction(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
    *,
    mode: str,
    n_sources: int,
    n_timepoints: int,
    ridge_alpha: float,
    gamma_grid: list[float],
) -> tuple[np.ndarray, dict[str, float]]:
    if mode == "none":
        return y_pred_test.astype(np.float32), {}
    if mode == "global_affine":
        return fit_global_affine(y_true_train, y_pred_train, y_pred_test, ridge_alpha)
    if mode in {"source_affine", "output_affine"}:
        return fit_axis_affine(
            y_true_train,
            y_pred_train,
            y_pred_test,
            mode=mode,
            n_sources=n_sources,
            n_timepoints=n_timepoints,
            ridge_alpha=ridge_alpha,
        )
    if mode in {"global_shrinkage", "source_shrinkage", "output_shrinkage"}:
        return fit_axis_shrinkage(
            y_true_train,
            y_pred_train,
            y_pred_test,
            mode=mode,
            n_sources=n_sources,
            n_timepoints=n_timepoints,
            gamma_grid=gamma_grid,
        )
    raise ValueError(f"Unknown calibration mode: {mode}")


def fit_dual_ridge(
    x_train: np.ndarray, y_train: np.ndarray, alpha: float
) -> np.ndarray:
    kernel = x_train @ x_train.T
    kernel.flat[:: kernel.shape[0] + 1] += alpha
    return np.linalg.solve(kernel, y_train)


def predict_dual_ridge(
    x_test: np.ndarray, x_train: np.ndarray, dual_coef: np.ndarray
) -> np.ndarray:
    return (x_test @ x_train.T) @ dual_coef


def group_rows_by_cache(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["participant_id"], row["phase"])
        grouped.setdefault(key, []).append(row)
    for key in grouped:
        grouped[key].sort(key=lambda item: int(item["trial_index"]))
    return grouped


def load_examples(
    rows: list[dict[str, str]], cache_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = group_rows_by_cache(rows)
    x_batches: list[np.ndarray] = []
    y_batches: list[np.ndarray] = []
    subject_batches: list[np.ndarray] = []
    for subject_phase in sorted(grouped):
        block_rows = grouped[subject_phase]
        first = block_rows[0]
        scalp = np.load(cache_root / first["scalp_npy"], mmap_mode="r")
        source = np.load(cache_root / first["source_npy"], mmap_mode="r")
        indices = [int(row["trial_index"]) for row in block_rows]
        x_batches.append(
            np.asarray(scalp[indices], dtype=np.float32).reshape(len(indices), -1)
        )
        y_batches.append(
            np.asarray(source[indices], dtype=np.float32).reshape(len(indices), -1)
        )
        subject_batches.append(np.asarray([first["participant_id"]] * len(indices)))
    return (
        np.concatenate(x_batches, axis=0),
        np.concatenate(y_batches, axis=0),
        np.concatenate(subject_batches, axis=0),
    )


def sourcewise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    source_labels: list[str],
    n_timepoints: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_sources = len(source_labels)
    y_true_3d = y_true.reshape(y_true.shape[0], n_sources, n_timepoints)
    y_pred_3d = y_pred.reshape(y_pred.shape[0], n_sources, n_timepoints)
    for source_index, label in enumerate(source_labels):
        true = y_true_3d[:, source_index, :]
        pred = y_pred_3d[:, source_index, :]
        rows.append(
            {
                "source_label": label,
                "pearson_r": pearson_corr(true, pred),
                "r2": r2_score(true, pred),
                "nrmse": nrmse(true, pred),
            }
        )
    return rows


def fit_forward_matrix(
    x_train_z: np.ndarray,
    y_train_z: np.ndarray,
    n_scalp_channels: int,
    n_source_channels: int,
    n_timepoints: int,
    alpha: float,
) -> np.ndarray:
    y_flat = (
        y_train_z.reshape(-1, n_source_channels, n_timepoints)
        .transpose(0, 2, 1)
        .reshape(-1, n_source_channels)
    )
    x_flat = (
        x_train_z.reshape(-1, n_scalp_channels, n_timepoints)
        .transpose(0, 2, 1)
        .reshape(-1, n_scalp_channels)
    )
    gram = y_flat.T @ y_flat
    gram.flat[:: gram.shape[0] + 1] += alpha
    weights = np.linalg.solve(gram, y_flat.T @ x_flat)
    return weights.T.astype(np.float32)


def fit_operator_bank(
    x_train_z: np.ndarray,
    y_train_z: np.ndarray,
    subject_ids: np.ndarray,
    n_scalp_channels: int,
    n_source_channels: int,
    n_timepoints: int,
    alpha: float,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    operator_subjects = sorted({str(subject_id) for subject_id in subject_ids.tolist()})
    operators: list[np.ndarray] = []
    subject_counts: dict[str, int] = {}
    for subject_id in operator_subjects:
        mask = subject_ids == subject_id
        subject_counts[subject_id] = int(mask.sum())
        operators.append(
            fit_forward_matrix(
                x_train_z=x_train_z[mask],
                y_train_z=y_train_z[mask],
                n_scalp_channels=n_scalp_channels,
                n_source_channels=n_source_channels,
                n_timepoints=n_timepoints,
                alpha=alpha,
            )
        )
    return (
        np.stack(operators, axis=0).astype(np.float32),
        operator_subjects,
        subject_counts,
    )


def fit_inverse_operator(
    forward_matrix: np.ndarray, inverse_alpha: float
) -> np.ndarray:
    n_scalp = forward_matrix.shape[0]
    gram = forward_matrix @ forward_matrix.T
    gram.flat[:: gram.shape[0] + 1] += inverse_alpha
    identity = np.eye(n_scalp, dtype=np.float32)
    return (forward_matrix.T @ np.linalg.solve(gram, identity)).astype(np.float32)


def apply_inverse_operator(
    inverse_operator: np.ndarray,
    scalp_flat: np.ndarray,
    n_scalp_channels: int,
    n_source_channels: int,
    n_timepoints: int,
) -> np.ndarray:
    scalp_3d = scalp_flat.reshape(-1, n_scalp_channels, n_timepoints)
    source_3d = np.einsum("sc,bct->bst", inverse_operator, scalp_3d)
    return source_3d.reshape(-1, n_source_channels * n_timepoints).astype(np.float32)


def apply_forward_operator(
    forward_matrix: np.ndarray,
    source_flat: np.ndarray,
    n_scalp_channels: int,
    n_source_channels: int,
    n_timepoints: int,
) -> np.ndarray:
    source_3d = source_flat.reshape(-1, n_source_channels, n_timepoints)
    scalp_3d = np.einsum("cs,bst->bct", forward_matrix, source_3d)
    return scalp_3d.reshape(-1, n_scalp_channels * n_timepoints).astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=repo_root
        / "data/derived/ds004752_beamforming_lowfreq_1to12hz/ds004752_beamforming_loso_trial_split.tsv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=repo_root / "data/cache/ds004752/beamforming_common_lowfreq_1to12hz",
    )
    parser.add_argument("--base-forward-ridge-alpha", type=float, default=100.0)
    parser.add_argument("--base-inverse-alpha", type=float, default=0.1)
    parser.add_argument(
        "--operator-mode",
        choices=("pooled", "subject_mean", "subject_bank_mean"),
        default="subject_bank_mean",
    )
    parser.add_argument(
        "--residual-feature-mode",
        choices=("scalp", "scalp_plus_base"),
        default="scalp_plus_base",
    )
    parser.add_argument("--residual-alpha", type=float, default=10.0)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument(
        "--residual-rank",
        type=int,
        default=0,
        help="If > 0, fit residual correction in a low-rank PCA basis of this size.",
    )
    parser.add_argument(
        "--prediction-calibration",
        choices=(
            "none",
            "global_affine",
            "source_affine",
            "output_affine",
            "global_shrinkage",
            "source_shrinkage",
            "output_shrinkage",
        ),
        default="none",
        help="Train-fold-only post-hoc calibration of predictions before test metrics.",
    )
    parser.add_argument("--calibration-ridge-alpha", type=float, default=1e-6)
    parser.add_argument(
        "--calibration-gamma-grid",
        default="0,0.1,0.2,0.35,0.5,0.75,1,1.25",
        help="Comma-separated gamma values for shrinkage calibration modes.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results/ds004752_beamforming_source_imaging_residual",
    )
    return parser


def fit_residual_basis(
    residual_train_centered: np.ndarray, residual_rank: int
) -> tuple[np.ndarray, np.ndarray]:
    if residual_rank <= 0:
        identity = np.eye(residual_train_centered.shape[1], dtype=np.float32)
        return residual_train_centered.astype(np.float32), identity

    max_rank = min(
        residual_rank,
        residual_train_centered.shape[0],
        residual_train_centered.shape[1],
    )
    _, _, vt = np.linalg.svd(residual_train_centered, full_matrices=False)
    basis = vt[:max_rank].T.astype(np.float32)
    coeff = residual_train_centered @ basis
    return coeff.astype(np.float32), basis


def main() -> None:
    args = build_parser().parse_args()
    calibration_gamma_grid = parse_float_grid(args.calibration_gamma_grid)
    split_manifest = args.split_manifest.resolve()
    cache_root = args.cache_root.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    split_rows = read_tsv(split_manifest)
    derived_dir = split_manifest.parent
    source_labels = [
        line.strip()
        for line in (derived_dir / "ds004752_beamforming_common_source_labels.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    scalp_labels = [
        line.strip()
        for line in (derived_dir / "ds004752_beamforming_common_scalp_labels.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    phase_metadata = {
        row["phase"]: row
        for row in read_tsv(derived_dir / "ds004752_beamforming_cache_manifest.tsv")
    }

    metric_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    source_summary_rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}

    folds = sorted({row["fold_id"] for row in split_rows})
    phases = sorted({row["phase"] for row in split_rows})

    for phase in phases:
        phase_fold_rows: list[dict[str, object]] = []
        n_timepoints = int(phase_metadata[phase]["n_timepoints"])
        n_scalp_channels = len(scalp_labels)
        n_source_channels = len(source_labels)
        for fold_id in folds:
            fold_phase_rows = [
                row
                for row in split_rows
                if row["fold_id"] == fold_id and row["phase"] == phase
            ]
            train_rows = [row for row in fold_phase_rows if row["split"] == "train"]
            test_rows = [row for row in fold_phase_rows if row["split"] == "test"]
            holdout_subject = fold_phase_rows[0]["holdout_subject"]

            x_train, y_train, train_subject_ids = load_examples(train_rows, cache_root)
            x_test, y_test, _ = load_examples(test_rows, cache_root)

            x_train_z, x_test_z, x_mean, x_std = zscore_train(x_train, x_test)
            y_train_z, _, y_mean, y_std = zscore_train(y_train, y_test)

            forward_bank, operator_subjects, operator_trial_counts = fit_operator_bank(
                x_train_z=x_train_z,
                y_train_z=y_train_z,
                subject_ids=train_subject_ids,
                n_scalp_channels=n_scalp_channels,
                n_source_channels=n_source_channels,
                n_timepoints=n_timepoints,
                alpha=args.base_forward_ridge_alpha,
            )

            if args.operator_mode == "pooled":
                forward_operator = fit_forward_matrix(
                    x_train_z=x_train_z,
                    y_train_z=y_train_z,
                    n_scalp_channels=n_scalp_channels,
                    n_source_channels=n_source_channels,
                    n_timepoints=n_timepoints,
                    alpha=args.base_forward_ridge_alpha,
                )
                inverse_operator = fit_inverse_operator(
                    forward_operator, args.base_inverse_alpha
                )
                y_base_train_z = apply_inverse_operator(
                    inverse_operator,
                    x_train_z,
                    n_scalp_channels,
                    n_source_channels,
                    n_timepoints,
                )
                y_base_test_z = apply_inverse_operator(
                    inverse_operator,
                    x_test_z,
                    n_scalp_channels,
                    n_source_channels,
                    n_timepoints,
                )
                repro_forward = forward_operator
                repro_mode = "single"
                n_operator_subjects = 1
            elif args.operator_mode == "subject_mean":
                forward_operator = forward_bank.mean(axis=0)
                inverse_operator = fit_inverse_operator(
                    forward_operator, args.base_inverse_alpha
                )
                y_base_train_z = apply_inverse_operator(
                    inverse_operator,
                    x_train_z,
                    n_scalp_channels,
                    n_source_channels,
                    n_timepoints,
                )
                y_base_test_z = apply_inverse_operator(
                    inverse_operator,
                    x_test_z,
                    n_scalp_channels,
                    n_source_channels,
                    n_timepoints,
                )
                repro_forward = forward_operator
                repro_mode = "single"
                n_operator_subjects = len(operator_subjects)
            else:
                inverse_bank = np.stack(
                    [
                        fit_inverse_operator(forward_operator, args.base_inverse_alpha)
                        for forward_operator in forward_bank
                    ],
                    axis=0,
                )
                y_train_bank = np.stack(
                    [
                        apply_inverse_operator(
                            inverse_operator,
                            x_train_z,
                            n_scalp_channels,
                            n_source_channels,
                            n_timepoints,
                        )
                        for inverse_operator in inverse_bank
                    ],
                    axis=0,
                )
                y_test_bank = np.stack(
                    [
                        apply_inverse_operator(
                            inverse_operator,
                            x_test_z,
                            n_scalp_channels,
                            n_source_channels,
                            n_timepoints,
                        )
                        for inverse_operator in inverse_bank
                    ],
                    axis=0,
                )
                y_base_train_z = y_train_bank.mean(axis=0)
                y_base_test_z = y_test_bank.mean(axis=0)
                repro_forward = forward_bank
                repro_mode = "bank_mean"
                n_operator_subjects = len(operator_subjects)

            y_base_train = (y_base_train_z * y_std) + y_mean
            y_base_test = (y_base_test_z * y_std) + y_mean
            residual_train = y_train - y_base_train
            residual_mean = residual_train.mean(axis=0, keepdims=True)
            residual_train_centered = residual_train - residual_mean

            if args.residual_feature_mode == "scalp":
                residual_train_features = x_train_z
                residual_test_features = x_test_z
            else:
                residual_train_features = np.concatenate(
                    [x_train_z, y_base_train_z], axis=1
                )
                residual_test_features = np.concatenate(
                    [x_test_z, y_base_test_z], axis=1
                )

            residual_train_target, residual_basis = fit_residual_basis(
                residual_train_centered=residual_train_centered,
                residual_rank=args.residual_rank,
            )
            residual_dual = fit_dual_ridge(
                residual_train_features,
                residual_train_target,
                alpha=args.residual_alpha,
            )
            train_residual_pred = predict_dual_ridge(
                residual_train_features,
                residual_train_features,
                residual_dual,
            )
            test_residual_pred = predict_dual_ridge(
                residual_test_features,
                residual_train_features,
                residual_dual,
            )
            if args.residual_rank > 0:
                residual_pred_train = train_residual_pred @ residual_basis.T
                residual_pred_test = test_residual_pred @ residual_basis.T
            else:
                residual_pred_train = train_residual_pred
                residual_pred_test = test_residual_pred
            residual_pred_train = residual_pred_train + residual_mean
            residual_pred_test = residual_pred_test + residual_mean

            y_pred_train = y_base_train + args.residual_weight * residual_pred_train
            y_pred_test_raw = y_base_test + args.residual_weight * residual_pred_test
            y_pred_test, calibration_info = calibrate_prediction(
                y_true_train=y_train,
                y_pred_train=y_pred_train,
                y_pred_test=y_pred_test_raw,
                mode=args.prediction_calibration,
                n_sources=n_source_channels,
                n_timepoints=n_timepoints,
                ridge_alpha=args.calibration_ridge_alpha,
                gamma_grid=calibration_gamma_grid,
            )
            y_pred_test_z = (y_pred_test - y_mean) / y_std

            if repro_mode == "single":
                scalp_pred_z = apply_forward_operator(
                    repro_forward,
                    y_pred_test_z,
                    n_scalp_channels,
                    n_source_channels,
                    n_timepoints,
                )
            else:
                scalp_pred_bank = np.stack(
                    [
                        apply_forward_operator(
                            repro_forward[index],
                            y_pred_test_z,
                            n_scalp_channels,
                            n_source_channels,
                            n_timepoints,
                        )
                        for index in range(repro_forward.shape[0])
                    ],
                    axis=0,
                )
                scalp_pred_z = scalp_pred_bank.mean(axis=0)
            scalp_pred = (scalp_pred_z * x_std) + x_mean

            metrics = {
                "fold_id": fold_id,
                "holdout_subject": holdout_subject,
                "phase": phase,
                "model": "source_imaging_residual",
                "operator_mode": args.operator_mode,
                "base_forward_ridge_alpha": args.base_forward_ridge_alpha,
                "base_inverse_alpha": args.base_inverse_alpha,
                "residual_feature_mode": args.residual_feature_mode,
                "residual_alpha": args.residual_alpha,
                "residual_weight": args.residual_weight,
                "residual_rank": args.residual_rank,
                "prediction_calibration": args.prediction_calibration,
                "calibration_ridge_alpha": args.calibration_ridge_alpha,
                "calibration_gamma_grid": args.calibration_gamma_grid,
                "n_train_trials": x_train.shape[0],
                "n_test_trials": x_test.shape[0],
                "n_operator_subjects": n_operator_subjects,
                "pearson_r": pearson_corr(y_test, y_pred_test),
                "r2": r2_score(y_test, y_pred_test),
                "nrmse": nrmse(y_test, y_pred_test),
                "raw_pearson_r": pearson_corr(y_test, y_pred_test_raw),
                "raw_r2": r2_score(y_test, y_pred_test_raw),
                "raw_nrmse": nrmse(y_test, y_pred_test_raw),
                "scalp_pearson_r": pearson_corr(x_test, scalp_pred),
                "scalp_r2": r2_score(x_test, scalp_pred),
                "scalp_nrmse": nrmse(x_test, scalp_pred),
                "base_test_pearson_r": pearson_corr(y_test, y_base_test),
                "base_test_r2": r2_score(y_test, y_base_test),
                "base_test_nrmse": nrmse(y_test, y_base_test),
                **calibration_info,
            }
            metric_rows.append(metrics)
            phase_fold_rows.append(metrics)

            for source_metric in sourcewise_metrics(
                y_test, y_pred_test, source_labels, n_timepoints
            ):
                source_rows.append(
                    {
                        "fold_id": fold_id,
                        "holdout_subject": holdout_subject,
                        "phase": phase,
                        "operator_mode": args.operator_mode,
                        "base_forward_ridge_alpha": args.base_forward_ridge_alpha,
                        "base_inverse_alpha": args.base_inverse_alpha,
                        "residual_feature_mode": args.residual_feature_mode,
                        "residual_alpha": args.residual_alpha,
                        "residual_weight": args.residual_weight,
                        "residual_rank": args.residual_rank,
                        "prediction_calibration": args.prediction_calibration,
                        "calibration_ridge_alpha": args.calibration_ridge_alpha,
                        "calibration_gamma_grid": args.calibration_gamma_grid,
                        **source_metric,
                    }
                )

        summary[phase] = {
            "mean_pearson_r": float(
                np.mean([row["pearson_r"] for row in phase_fold_rows])
            ),
            "std_pearson_r": float(
                np.std([row["pearson_r"] for row in phase_fold_rows])
            ),
            "mean_r2": float(np.mean([row["r2"] for row in phase_fold_rows])),
            "std_r2": float(np.std([row["r2"] for row in phase_fold_rows])),
            "mean_nrmse": float(np.mean([row["nrmse"] for row in phase_fold_rows])),
            "std_nrmse": float(np.std([row["nrmse"] for row in phase_fold_rows])),
            "mean_scalp_pearson_r": float(
                np.mean([row["scalp_pearson_r"] for row in phase_fold_rows])
            ),
            "std_scalp_pearson_r": float(
                np.std([row["scalp_pearson_r"] for row in phase_fold_rows])
            ),
            "mean_scalp_r2": float(
                np.mean([row["scalp_r2"] for row in phase_fold_rows])
            ),
            "std_scalp_r2": float(np.std([row["scalp_r2"] for row in phase_fold_rows])),
            "mean_scalp_nrmse": float(
                np.mean([row["scalp_nrmse"] for row in phase_fold_rows])
            ),
            "std_scalp_nrmse": float(
                np.std([row["scalp_nrmse"] for row in phase_fold_rows])
            ),
            "mean_base_pearson_r": float(
                np.mean([row["base_test_pearson_r"] for row in phase_fold_rows])
            ),
            "mean_base_r2": float(
                np.mean([row["base_test_r2"] for row in phase_fold_rows])
            ),
            "mean_base_nrmse": float(
                np.mean([row["base_test_nrmse"] for row in phase_fold_rows])
            ),
        }

    for phase in phases:
        for source_label in source_labels:
            rows = [
                row
                for row in source_rows
                if row["phase"] == phase and row["source_label"] == source_label
            ]
            source_summary_rows.append(
                {
                    "phase": phase,
                    "source_label": source_label,
                    "mean_pearson_r": float(
                        np.mean([row["pearson_r"] for row in rows])
                    ),
                    "std_pearson_r": float(np.std([row["pearson_r"] for row in rows])),
                    "mean_r2": float(np.mean([row["r2"] for row in rows])),
                    "std_r2": float(np.std([row["r2"] for row in rows])),
                    "mean_nrmse": float(np.mean([row["nrmse"] for row in rows])),
                    "std_nrmse": float(np.std([row["nrmse"] for row in rows])),
                }
            )

    write_tsv(
        results_dir / "fold_metrics.tsv", list(metric_rows[0].keys()), metric_rows
    )
    write_tsv(
        results_dir / "source_metrics.tsv", list(source_rows[0].keys()), source_rows
    )
    write_tsv(
        results_dir / "source_summary.tsv",
        list(source_summary_rows[0].keys()),
        source_summary_rows,
    )
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (results_dir / "config.json").write_text(
        json.dumps(
            {
                "split_manifest": str(split_manifest),
                "cache_root": str(cache_root),
                "operator_mode": args.operator_mode,
                "base_forward_ridge_alpha": args.base_forward_ridge_alpha,
                "base_inverse_alpha": args.base_inverse_alpha,
                "residual_feature_mode": args.residual_feature_mode,
                "residual_alpha": args.residual_alpha,
                "residual_weight": args.residual_weight,
                "residual_rank": args.residual_rank,
                "prediction_calibration": args.prediction_calibration,
                "calibration_ridge_alpha": args.calibration_ridge_alpha,
                "calibration_gamma_grid": args.calibration_gamma_grid,
                "n_scalp_channels": len(scalp_labels),
                "n_source_channels": len(source_labels),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(results_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
