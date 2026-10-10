#!/usr/bin/env python3
"""Evaluate SPES localization and cross-validated residual-quantile prediction balls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    index = int(np.ceil(coverage * (ordered.size + 1))) - 1
    return float(ordered[max(0, min(index, ordered.size - 1))])


def normalize_coords(
    dataset_root: Path,
    participant_id: str,
    channel_names: list[str],
) -> np.ndarray:
    electrode_path = (
        dataset_root
        / f"derivatives/epochs/{participant_id}/eeg/{participant_id}_task-ccepcoreg_electrodes.tsv"
    )
    coords_by_name = {
        row["name"]: np.asarray(
            [float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64
        )
        for row in read_tsv(electrode_path)
    }
    coords = np.stack([coords_by_name[name] for name in channel_names], axis=0)
    coords -= np.median(coords, axis=0, keepdims=True)
    radius = float(np.median(np.linalg.norm(coords, axis=1)))
    if radius <= 0:
        radius = 1.0
    return coords / radius


def scalp_basis(coords: np.ndarray) -> np.ndarray:
    x, y, z = coords.T
    return np.stack(
        [
            np.ones_like(x),
            x,
            y,
            z,
            x * x,
            y * y,
            z * z,
            x * y,
            x * z,
            y * z,
            x * x * x,
            y * y * y,
            z * z * z,
        ],
        axis=1,
    )


def fixed_scalp_features(
    values: np.ndarray, coords: np.ndarray, basis_ridge: float
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    basis = scalp_basis(coords)
    gram = basis.T @ basis + basis_ridge * np.eye(basis.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(gram, basis.T @ values)
    fitted = basis @ coefficients
    residual_rms = float(np.sqrt(np.mean((values - fitted) ** 2)))

    weights = np.abs(values) + 1e-12
    weights /= weights.sum()
    centroid = np.sum(coords * weights[:, None], axis=0)
    centered = coords - centroid[None, :]
    covariance = np.einsum("n,ni,nj->ij", weights, centered, centered)
    covariance_upper = covariance[np.triu_indices(3)]
    peak_coord = coords[int(np.argmax(np.abs(values)))]
    quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    summaries = np.asarray(
        [
            float(values.mean()),
            float(values.std()),
            float(np.mean(np.abs(values))),
            residual_rms,
            float(values.size / 256.0),
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        [coefficients, centroid, covariance_upper, peak_coord, quantiles, summaries],
        axis=0,
    )


def load_dataset(
    manifest_path: Path,
    repo_root: Path,
    dataset_root: Path,
    basis_ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    rows = read_tsv(manifest_path)
    coords_cache: dict[str, np.ndarray] = {}
    names_cache: dict[str, list[str]] = {}
    features: list[np.ndarray] = []
    audit_rows: list[dict[str, object]] = []
    for row in rows:
        subject = row["participant_id"]
        if subject not in coords_cache:
            names = [
                line.strip()
                for line in (repo_root / row["common_good_channels_txt"])
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            names_cache[subject] = names
            coords_cache[subject] = normalize_coords(dataset_root, subject, names)
        values = np.load(repo_root / row["feature_npy"]).astype(np.float64)
        if values.size != coords_cache[subject].shape[0]:
            raise RuntimeError(
                f"Feature/channel mismatch for {subject} {row['run_id']}: "
                f"{values.size} vs {coords_cache[subject].shape[0]}"
            )
        features.append(
            fixed_scalp_features(values, coords_cache[subject], basis_ridge)
        )
        audit_rows.append(
            {
                "participant_id": subject,
                "run_id": row["run_id"],
                "n_channels": len(names_cache[subject]),
                "feature_length": int(features[-1].size),
                "target_mni_x_mm": 1000.0 * float(row["mni_x"]),
                "target_mni_y_mm": 1000.0 * float(row["mni_y"]),
                "target_mni_z_mm": 1000.0 * float(row["mni_z"]),
            }
        )
    x = np.stack(features, axis=0)
    y = np.asarray(
        [
            [
                1000.0 * float(row["mni_x"]),
                1000.0 * float(row["mni_y"]),
                1000.0 * float(row["mni_z"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    subjects = np.asarray([row["participant_id"] for row in rows])
    run_ids = np.asarray([row["run_id"] for row in rows])
    return x, y, subjects, run_ids, audit_rows


def model_candidates(args: argparse.Namespace) -> dict[str, list[dict[str, object]]]:
    return {
        "coordinate_mean": [{}],
        "ridge": [{"alpha": alpha} for alpha in args.ridge_alphas],
        "extra_trees": [
            {"max_features": max_features, "min_samples_leaf": min_leaf}
            for max_features in args.extra_max_features
            for min_leaf in args.extra_min_samples_leaf
        ],
    }


def fit_predict(
    model_name: str,
    params: dict[str, object],
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    n_trees: int,
) -> np.ndarray:
    if model_name == "coordinate_mean":
        return np.repeat(train_y.mean(axis=0, keepdims=True), test_x.shape[0], axis=0)
    if model_name == "ridge":
        scaler = StandardScaler()
        train_z = scaler.fit_transform(train_x)
        test_z = scaler.transform(test_x)
        model = Ridge(alpha=float(params["alpha"]))
        model.fit(train_z, train_y)
        return np.asarray(model.predict(test_z), dtype=np.float64)
    if model_name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=n_trees,
            max_features=float(params["max_features"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(train_x, train_y)
        return np.asarray(model.predict(test_x), dtype=np.float64)
    raise ValueError(model_name)


def grouped_oof_predictions(
    model_name: str,
    params: dict[str, object],
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_trees: int,
    n_splits: int,
) -> np.ndarray:
    splitter = GroupKFold(n_splits=min(n_splits, len(set(groups.tolist()))))
    predictions = np.full_like(y, np.nan, dtype=np.float64)
    for fold_index, (fit_index, val_index) in enumerate(splitter.split(x, y, groups)):
        predictions[val_index] = fit_predict(
            model_name,
            params,
            x[fit_index],
            y[fit_index],
            x[val_index],
            seed + fold_index,
            n_trees,
        )
    if np.isnan(predictions).any():
        raise RuntimeError("Grouped OOF predictions are incomplete.")
    return predictions


def select_candidate(
    model_name: str,
    candidates: list[dict[str, object]],
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, object], np.ndarray, float]:
    scored: list[tuple[float, str, dict[str, object], np.ndarray]] = []
    for candidate_index, params in enumerate(candidates):
        predictions = grouped_oof_predictions(
            model_name,
            params,
            x,
            y,
            groups,
            args.seed + 1000 * candidate_index,
            args.n_trees,
            args.inner_splits,
        )
        errors = np.linalg.norm(predictions - y, axis=1)
        score = float(np.mean(errors))
        scored.append((score, json.dumps(params, sort_keys=True), params, predictions))
    best = min(scored, key=lambda item: (item[0], item[1]))
    return best[2], best[3], best[0]


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    repo_root = args.repo_root.resolve()
    x, y, subjects, run_ids, feature_audit = load_dataset(
        args.feature_manifest.resolve(),
        repo_root,
        args.dataset_root.resolve(),
        args.basis_ridge,
    )
    unique_subjects = sorted(set(subjects.tolist()))
    candidates_by_model = model_candidates(args)
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []

    for outer_index, holdout_subject in enumerate(unique_subjects):
        test_mask = subjects == holdout_subject
        train_mask = ~test_mask
        train_subjects = set(subjects[train_mask].tolist())
        test_subjects = set(subjects[test_mask].tolist())
        overlap = train_subjects & test_subjects
        if overlap:
            raise RuntimeError(
                f"Subject leakage in outer fold {holdout_subject}: {sorted(overlap)}"
            )
        leakage_rows.append(
            {
                "holdout_subject": holdout_subject,
                "n_train_subjects": len(train_subjects),
                "n_test_subjects": len(test_subjects),
                "subject_overlap": 0,
                "n_train_runs": int(train_mask.sum()),
                "n_test_runs": int(test_mask.sum()),
            }
        )

        for model_name, candidates in candidates_by_model.items():
            best_params, inner_predictions, inner_mean_error = select_candidate(
                model_name,
                candidates,
                x[train_mask],
                y[train_mask],
                subjects[train_mask],
                args,
            )
            calibration_errors = np.linalg.norm(
                inner_predictions - y[train_mask], axis=1
            )
            radius = conformal_quantile(calibration_errors, args.coverage)
            test_predictions = fit_predict(
                model_name,
                best_params,
                x[train_mask],
                y[train_mask],
                x[test_mask],
                args.seed + 100000 + outer_index,
                args.n_trees,
            )
            test_errors = np.linalg.norm(test_predictions - y[test_mask], axis=1)
            selection_rows.append(
                {
                    "holdout_subject": holdout_subject,
                    "model": model_name,
                    "selected_params": json.dumps(best_params, sort_keys=True),
                    "inner_mean_error_mm": inner_mean_error,
                    "conformal_radius_mm": radius,
                    "n_calibration_runs": int(train_mask.sum()),
                    "n_calibration_subjects": len(train_subjects),
                }
            )
            for run_id, truth, prediction, error in zip(
                run_ids[test_mask], y[test_mask], test_predictions, test_errors
            ):
                prediction_rows.append(
                    {
                        "participant_id": holdout_subject,
                        "run_id": str(run_id),
                        "model": model_name,
                        "true_mni_x_mm": truth[0],
                        "true_mni_y_mm": truth[1],
                        "true_mni_z_mm": truth[2],
                        "pred_mni_x_mm": prediction[0],
                        "pred_mni_y_mm": prediction[1],
                        "pred_mni_z_mm": prediction[2],
                        "error_mm": error,
                        "radius_mm": radius,
                        "covered": int(error <= radius),
                    }
                )
        print(f"completed outer subject {holdout_subject}", flush=True)

    metric_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    for model_name in candidates_by_model:
        selected = [row for row in prediction_rows if row["model"] == model_name]
        errors = np.asarray(
            [float(row["error_mm"]) for row in selected], dtype=np.float64
        )
        radii = np.asarray(
            [float(row["radius_mm"]) for row in selected], dtype=np.float64
        )
        covered = np.asarray(
            [int(row["covered"]) for row in selected], dtype=np.float64
        )
        subject_coverages: list[float] = []
        for subject in unique_subjects:
            subject_selected = [
                row for row in selected if row["participant_id"] == subject
            ]
            subject_error = np.asarray(
                [float(row["error_mm"]) for row in subject_selected]
            )
            subject_covered = np.asarray(
                [int(row["covered"]) for row in subject_selected]
            )
            coverage = float(np.mean(subject_covered))
            subject_coverages.append(coverage)
            subject_rows.append(
                {
                    "participant_id": subject,
                    "model": model_name,
                    "n_runs": len(subject_selected),
                    "median_error_mm": float(np.median(subject_error)),
                    "p75_error_mm": float(np.percentile(subject_error, 75)),
                    "coverage": coverage,
                }
            )
        metric_rows.append(
            {
                "model": model_name,
                "n_runs": len(selected),
                "n_subjects": len(unique_subjects),
                "median_error_mm": float(np.median(errors)),
                "p75_error_mm": float(np.percentile(errors, 75)),
                "mean_error_mm": float(np.mean(errors)),
                "nominal_coverage": args.coverage,
                "pooled_coverage": float(np.mean(covered)),
                "mean_subject_coverage": float(np.mean(subject_coverages)),
                "sd_subject_coverage": float(np.std(subject_coverages)),
                "min_subject_coverage": float(np.min(subject_coverages)),
                "max_subject_coverage": float(np.max(subject_coverages)),
                "median_radius_mm": float(np.median(radii)),
                "p75_radius_mm": float(np.percentile(radii, 75)),
            }
        )

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(results_dir / "feature_audit.tsv", feature_audit)
    write_tsv(results_dir / "leakage_audit.tsv", leakage_rows)
    write_tsv(results_dir / "nested_selection.tsv", selection_rows)
    write_tsv(results_dir / "prediction_rows.tsv", prediction_rows)
    write_tsv(results_dir / "pooled_metrics.tsv", metric_rows)
    write_tsv(results_dir / "subject_metrics.tsv", subject_rows)
    np.savez_compressed(
        results_dir / "fixed_feature_matrix.npz",
        x=x,
        y=y,
        subjects=subjects,
        run_ids=run_ids,
    )
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config |= {
        "n_runs": int(x.shape[0]),
        "n_subjects": len(unique_subjects),
        "fixed_feature_dimension": int(x.shape[1]),
    }
    (results_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    summary = {
        "scope": "end-to-end held-out-subject MNI localization with group-derived conformal radii",
        "feature_definition": "low-order Cartesian scalp-field coefficients and geometry-invariant moments",
        "metrics": metric_rows,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = ["# SPES participant-held-out localization", ""]
    for row in metric_rows:
        lines.append(
            f"- {row['model']}: median/p75 error `{float(row['median_error_mm']):.2f}`/"
            f"`{float(row['p75_error_mm']):.2f}` mm; pooled 90% coverage "
            f"`{float(row['pooled_coverage']):.3f}`; median radius `{float(row['median_radius_mm']):.2f}` mm."
        )
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=repo_root / "data/derived/spes_raw_feature_manifest.tsv",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=repo_root / "data/raw/spes-hd-eeg-seeg"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results/spes_localization",
    )
    parser.add_argument("--basis-ridge", type=float, default=1e-3)
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument(
        "--extra-max-features", type=float, nargs="+", default=[0.5, 1.0]
    )
    parser.add_argument("--extra-min-samples-leaf", type=int, nargs="+", default=[2, 5])
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=34170)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
