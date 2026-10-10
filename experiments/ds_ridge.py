#!/usr/bin/env python3
"""Run a LOSO ridge baseline on cached ds004752 beamforming trials."""

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
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std, mean, std


def center_train(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    return train - mean, test - mean, mean


def fit_dual_ridge(
    x_train: np.ndarray, y_train: np.ndarray, alpha: float
) -> np.ndarray:
    kernel = x_train @ x_train.T
    kernel.flat[:: kernel.shape[0] + 1] += alpha
    dual_coef = np.linalg.solve(kernel, y_train)
    return dual_coef


def predict_dual_ridge(
    x_test: np.ndarray, x_train: np.ndarray, dual_coef: np.ndarray
) -> np.ndarray:
    return (x_test @ x_train.T) @ dual_coef


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
    rows: list[dict[str, str]],
    cache_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    grouped = group_rows_by_cache(rows)
    x_batches: list[np.ndarray] = []
    y_batches: list[np.ndarray] = []
    for subject_phase in sorted(grouped):
        block_rows = grouped[subject_phase]
        first = block_rows[0]
        scalp = np.load(cache_root / first["scalp_npy"], mmap_mode="r")
        source = np.load(cache_root / first["source_npy"], mmap_mode="r")
        indices = [int(row["trial_index"]) for row in block_rows]
        x_batches.append(np.asarray(scalp[indices]).reshape(len(indices), -1))
        y_batches.append(np.asarray(source[indices]).reshape(len(indices), -1))
    return np.concatenate(x_batches, axis=0), np.concatenate(y_batches, axis=0)


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


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=repo_root / "data/derived/ds004752_beamforming_loso_trial_split.tsv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=repo_root / "data/cache/ds004752/beamforming_common",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results/ds004752_beamforming_ridge_alpha10",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split_rows = read_tsv(args.split_manifest.resolve())
    cache_root = args.cache_root.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    source_labels = [
        line.strip()
        for line in (
            args.split_manifest.resolve().parents[0]
            / "ds004752_beamforming_common_source_labels.txt"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    phase_metadata = {
        row["phase"]: row
        for row in read_tsv(
            args.split_manifest.resolve().parents[0]
            / "ds004752_beamforming_cache_manifest.tsv"
        )
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
        for fold_id in folds:
            fold_phase_rows = [
                row
                for row in split_rows
                if row["fold_id"] == fold_id and row["phase"] == phase
            ]
            train_rows = [row for row in fold_phase_rows if row["split"] == "train"]
            test_rows = [row for row in fold_phase_rows if row["split"] == "test"]
            x_train, y_train = load_examples(train_rows, cache_root)
            x_test, y_test = load_examples(test_rows, cache_root)

            x_train_z, x_test_z, _, _ = zscore_train(x_train, x_test)
            y_train_c, _, y_mean = center_train(y_train, y_test)
            dual_coef = fit_dual_ridge(x_train_z, y_train_c, alpha=args.alpha)
            y_pred = predict_dual_ridge(x_test_z, x_train_z, dual_coef) + y_mean

            metrics = {
                "fold_id": fold_id,
                "holdout_subject": fold_phase_rows[0]["holdout_subject"],
                "phase": phase,
                "alpha": args.alpha,
                "n_train_trials": x_train.shape[0],
                "n_test_trials": x_test.shape[0],
                "pearson_r": pearson_corr(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "nrmse": nrmse(y_test, y_pred),
            }
            metric_rows.append(metrics)
            phase_fold_rows.append(metrics)

            for source_metric in sourcewise_metrics(
                y_test, y_pred, source_labels, n_timepoints
            ):
                source_rows.append(
                    {
                        "fold_id": fold_id,
                        "holdout_subject": fold_phase_rows[0]["holdout_subject"],
                        "phase": phase,
                        "alpha": args.alpha,
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
        }

    write_tsv(
        results_dir / "fold_metrics.tsv", list(metric_rows[0].keys()), metric_rows
    )
    write_tsv(
        results_dir / "source_metrics.tsv", list(source_rows[0].keys()), source_rows
    )
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
        results_dir / "source_summary.tsv",
        list(source_summary_rows[0].keys()),
        source_summary_rows,
    )
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Wrote ridge baseline results:")
    print(results_dir / "fold_metrics.tsv")
    print(results_dir / "source_metrics.tsv")
    print(results_dir / "source_summary.tsv")
    print(results_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
