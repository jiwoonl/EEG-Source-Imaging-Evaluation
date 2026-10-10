#!/usr/bin/env python3
"""Evaluate the fixed SPES spatial-retrieval predictor from normalized features."""

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


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denom == 0:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = temperature * (values - np.max(values))
    weights = np.exp(scaled)
    weight_sum = float(np.sum(weights))
    if weight_sum == 0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return weights / weight_sum


def lle_weights(
    query_feature: np.ndarray,
    neighbor_features: np.ndarray,
    regularization: float,
) -> np.ndarray:
    diffs = neighbor_features - query_feature[None, :]
    gram = diffs @ diffs.T
    trace = float(np.trace(gram))
    diagonal_reg = regularization * (trace / max(1, gram.shape[0]))
    if diagonal_reg <= 0:
        diagonal_reg = 1e-6
    gram.flat[:: gram.shape[0] + 1] += diagonal_reg
    ones = np.ones(gram.shape[0], dtype=np.float64)
    weights = np.linalg.solve(gram, ones)
    weight_sum = float(np.sum(weights))
    if abs(weight_sum) < 1e-12:
        return np.full(weights.shape, 1.0 / weights.size, dtype=np.float64)
    return weights / weight_sum


def load_subject_common_eeg_positions_mm(
    dataset_root: Path,
    participant_id: str,
    common_good_channels_txt: Path,
) -> np.ndarray:
    electrode_path = (
        dataset_root
        / f"derivatives/epochs/{participant_id}/eeg/{participant_id}_task-ccepcoreg_electrodes.tsv"
    )
    electrode_rows = read_tsv(electrode_path)
    wanted = {
        line.strip()
        for line in common_good_channels_txt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    coords_mm: list[np.ndarray] = []
    for row in electrode_rows:
        if row["name"] not in wanted:
            continue
        coords_mm.append(
            np.asarray(
                [float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64
            )
        )
    if not coords_mm:
        raise RuntimeError(f"No EEG electrode coordinates found for {participant_id}.")
    return np.stack(coords_mm, axis=0)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=repo_root / "data/derived/spes_spatial_retrieval_feature_manifest.tsv",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results/spes_spatial_retrieval_baseline",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "data/raw/spes-hd-eeg-seeg",
    )
    parser.add_argument(
        "--method",
        choices=("hard_nn", "soft_knn", "lle_knn"),
        default="hard_nn",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--lle-regularization", type=float, default=1e-2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    dataset_root = args.dataset_root.resolve()
    feature_manifest_path = args.feature_manifest.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_tsv(feature_manifest_path)
    rows_by_subject: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        rows_by_subject.setdefault(row["participant_id"], []).append(row)

    run_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []

    for participant_id, subject_manifest_rows in sorted(rows_by_subject.items()):
        features: list[np.ndarray] = []
        labels: list[str] = []
        currents: list[float] = []
        coords_t1w: list[np.ndarray] = []
        coords_mni: list[np.ndarray] = []
        run_ids: list[str] = []
        n_common_channels = int(
            subject_manifest_rows[0]["subject_common_good_channel_count"]
        )
        feature_length = int(subject_manifest_rows[0]["feature_length"])
        eeg_positions_mm = load_subject_common_eeg_positions_mm(
            dataset_root=dataset_root,
            participant_id=participant_id,
            common_good_channels_txt=repo_root
            / subject_manifest_rows[0]["common_good_channels_txt"],
        )

        for row in subject_manifest_rows:
            feature = np.load(repo_root / row["feature_npy"]).astype(np.float64)
            features.append(feature)
            labels.append(row["stim_label"])
            currents.append(safe_float(row["stim_current_ma"]))
            coords_t1w.append(
                np.asarray(
                    [float(row["t1w_x"]), float(row["t1w_y"]), float(row["t1w_z"])],
                    dtype=np.float64,
                )
            )
            coords_mni.append(
                np.asarray(
                    [float(row["mni_x"]), float(row["mni_y"]), float(row["mni_z"])],
                    dtype=np.float64,
                )
            )
            run_ids.append(row["run_id"])

        label_counts: dict[str, int] = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        subject_errors_mm: list[float] = []
        subject_random_mm: list[float] = []
        repeated_correct = 0
        repeated_total = 0

        for index, (row, feature, true_coord_t1w, true_coord_mni) in enumerate(
            zip(subject_manifest_rows, features, coords_t1w, coords_mni)
        ):
            similarity_scores = np.asarray(
                [
                    pearson(feature, candidate) if candidate_index != index else -np.inf
                    for candidate_index, candidate in enumerate(features)
                ],
                dtype=np.float64,
            )
            best_index = int(np.argmax(similarity_scores))
            best_similarity = float(similarity_scores[best_index])
            sorted_candidate_indices = np.argsort(similarity_scores)[::-1]
            sorted_candidate_indices = sorted_candidate_indices[
                np.isfinite(similarity_scores[sorted_candidate_indices])
            ]

            if args.method == "hard_nn":
                predicted_coord_t1w = coords_t1w[best_index]
                predicted_coord_mni = coords_mni[best_index]
                neighbor_weights = np.asarray([1.0], dtype=np.float64)
                neighbor_indices = np.asarray([best_index], dtype=np.int64)
            elif args.method == "soft_knn":
                top_k = max(1, min(args.top_k, sorted_candidate_indices.size))
                neighbor_indices = sorted_candidate_indices[:top_k]
                neighbor_scores = similarity_scores[neighbor_indices]
                neighbor_weights = softmax(neighbor_scores, args.temperature)
                predicted_coord_t1w = np.sum(
                    np.stack([coords_t1w[idx] for idx in neighbor_indices], axis=0)
                    * neighbor_weights[:, None],
                    axis=0,
                )
                predicted_coord_mni = np.sum(
                    np.stack([coords_mni[idx] for idx in neighbor_indices], axis=0)
                    * neighbor_weights[:, None],
                    axis=0,
                )
            else:
                top_k = max(2, min(args.top_k, sorted_candidate_indices.size))
                neighbor_indices = sorted_candidate_indices[:top_k]
                neighbor_features = np.stack(
                    [features[idx] for idx in neighbor_indices], axis=0
                )
                neighbor_weights = lle_weights(
                    query_feature=feature,
                    neighbor_features=neighbor_features,
                    regularization=args.lle_regularization,
                )
                predicted_coord_t1w = np.sum(
                    np.stack([coords_t1w[idx] for idx in neighbor_indices], axis=0)
                    * neighbor_weights[:, None],
                    axis=0,
                )
                predicted_coord_mni = np.sum(
                    np.stack([coords_mni[idx] for idx in neighbor_indices], axis=0)
                    * neighbor_weights[:, None],
                    axis=0,
                )

            neighbor_coords_t1w = np.stack(
                [coords_t1w[idx] for idx in neighbor_indices], axis=0
            )
            neighbor_coords_mni = np.stack(
                [coords_mni[idx] for idx in neighbor_indices], axis=0
            )
            spread_weights = np.abs(neighbor_weights).astype(np.float64)
            spread_weights_sum = float(np.sum(spread_weights))
            if spread_weights_sum == 0:
                spread_weights = np.full(
                    spread_weights.shape, 1.0 / spread_weights.size, dtype=np.float64
                )
            else:
                spread_weights /= spread_weights_sum
            raw_spatial_spread_t1w_mm = float(
                np.sqrt(
                    np.sum(
                        spread_weights
                        * np.sum(
                            (neighbor_coords_t1w - predicted_coord_t1w[None, :]) ** 2,
                            axis=1,
                        )
                    )
                )
                * 1000.0
            )
            raw_spatial_spread_mni_mm = float(
                np.sqrt(
                    np.sum(
                        spread_weights
                        * np.sum(
                            (neighbor_coords_mni - predicted_coord_mni[None, :]) ** 2,
                            axis=1,
                        )
                    )
                )
                * 1000.0
            )

            t1w_distances_mm = np.asarray(
                [
                    float(np.linalg.norm(true_coord_t1w - candidate_coord) * 1000.0)
                    for candidate_coord in coords_t1w
                ],
                dtype=np.float64,
            )
            mni_distances_mm = np.asarray(
                [
                    float(np.linalg.norm(true_coord_mni - candidate_coord) * 1000.0)
                    for candidate_coord in coords_mni
                ],
                dtype=np.float64,
            )

            candidate_t1w_distances = np.delete(t1w_distances_mm, index)
            candidate_mni_distances = np.delete(mni_distances_mm, index)
            oracle_t1w_error_mm = float(np.min(candidate_t1w_distances))
            oracle_mni_error_mm = float(np.min(candidate_mni_distances))
            expected_random_t1w_error_mm = float(np.mean(candidate_t1w_distances))
            expected_random_mni_error_mm = float(np.mean(candidate_mni_distances))
            predicted_t1w_error_mm = float(
                np.linalg.norm(predicted_coord_t1w - true_coord_t1w) * 1000.0
            )
            predicted_mni_error_mm = float(
                np.linalg.norm(predicted_coord_mni - true_coord_mni) * 1000.0
            )
            true_depth_proxy_mm = float(
                np.min(
                    np.linalg.norm(
                        eeg_positions_mm - (true_coord_t1w[None, :] * 1000.0), axis=1
                    )
                )
            )
            predicted_depth_proxy_mm = float(
                np.min(
                    np.linalg.norm(
                        eeg_positions_mm - (predicted_coord_t1w[None, :] * 1000.0),
                        axis=1,
                    )
                )
            )

            repeated_label_available = label_counts[row["stim_label"]] > 1
            repeated_label_correct = (
                repeated_label_available and labels[best_index] == row["stim_label"]
            )
            if repeated_label_available:
                repeated_total += 1
                repeated_correct += int(repeated_label_correct)

            subject_errors_mm.append(predicted_t1w_error_mm)
            subject_random_mm.append(expected_random_t1w_error_mm)

            run_rows.append(
                {
                    "participant_id": participant_id,
                    "run_id": row["run_id"],
                    "stim_label": row["stim_label"],
                    "stim_current_ma": row["stim_current_ma"],
                    "method": args.method,
                    "top_k": args.top_k if args.method != "hard_nn" else 1,
                    "temperature": (
                        args.temperature if args.method == "soft_knn" else ""
                    ),
                    "lle_regularization": (
                        args.lle_regularization if args.method == "lle_knn" else ""
                    ),
                    "n_common_good_channels": n_common_channels,
                    "feature_length": feature_length,
                    "predicted_run_id": run_ids[best_index],
                    "predicted_stim_label": labels[best_index],
                    "predicted_stim_current_ma": currents[best_index],
                    "top1_similarity": best_similarity,
                    "neighbor_run_ids": ",".join(
                        run_ids[idx] for idx in neighbor_indices
                    ),
                    "neighbor_weights": ",".join(
                        f"{weight:.6f}" for weight in neighbor_weights
                    ),
                    "true_t1w_x": true_coord_t1w[0],
                    "true_t1w_y": true_coord_t1w[1],
                    "true_t1w_z": true_coord_t1w[2],
                    "predicted_t1w_x": predicted_coord_t1w[0],
                    "predicted_t1w_y": predicted_coord_t1w[1],
                    "predicted_t1w_z": predicted_coord_t1w[2],
                    "raw_spatial_spread_t1w_mm": raw_spatial_spread_t1w_mm,
                    "raw_spatial_spread_mni_mm": raw_spatial_spread_mni_mm,
                    "t1w_error_mm": predicted_t1w_error_mm,
                    "mni_error_mm": predicted_mni_error_mm,
                    "oracle_t1w_error_mm": oracle_t1w_error_mm,
                    "oracle_mni_error_mm": oracle_mni_error_mm,
                    "expected_random_t1w_error_mm": expected_random_t1w_error_mm,
                    "expected_random_mni_error_mm": expected_random_mni_error_mm,
                    "true_depth_proxy_mm": true_depth_proxy_mm,
                    "predicted_depth_proxy_mm": predicted_depth_proxy_mm,
                    "repeated_label_available": int(repeated_label_available),
                    "repeated_label_top1_correct": (
                        int(repeated_label_correct) if repeated_label_available else ""
                    ),
                    "t1w_improvement_over_random_mm": expected_random_t1w_error_mm
                    - predicted_t1w_error_mm,
                    "mni_improvement_over_random_mm": expected_random_mni_error_mm
                    - predicted_mni_error_mm,
                }
            )

        subject_rows.append(
            {
                "participant_id": participant_id,
                "n_runs": len(subject_manifest_rows),
                "n_common_good_channels": n_common_channels,
                "feature_length": feature_length,
                "median_t1w_error_mm": percentile(subject_errors_mm, 50),
                "p75_t1w_error_mm": percentile(subject_errors_mm, 75),
                "median_expected_random_t1w_error_mm": percentile(
                    subject_random_mm, 50
                ),
                "repeated_label_top1_accuracy": (
                    (repeated_correct / repeated_total) if repeated_total else ""
                ),
                "repeated_label_n_runs": repeated_total,
            }
        )

    depth_values = np.asarray(
        [float(row["true_depth_proxy_mm"]) for row in run_rows], dtype=np.float64
    )
    depth_q33 = float(np.percentile(depth_values, 33.3333333333))
    depth_q67 = float(np.percentile(depth_values, 66.6666666667))
    for row in run_rows:
        depth_value = float(row["true_depth_proxy_mm"])
        if depth_value <= depth_q33:
            row["depth_proxy_bin"] = "shallow_tertile"
        elif depth_value <= depth_q67:
            row["depth_proxy_bin"] = "middle_tertile"
        else:
            row["depth_proxy_bin"] = "deep_tertile"

    subset_definitions: list[tuple[str, list[dict[str, object]]]] = [
        ("all_runs", run_rows),
        (
            "five_ma_only",
            [
                row
                for row in run_rows
                if safe_float(str(row["stim_current_ma"])) is not None
                and safe_float(str(row["stim_current_ma"])) >= 5.0
            ],
        ),
        (
            "below_five_ma",
            [
                row
                for row in run_rows
                if safe_float(str(row["stim_current_ma"])) is not None
                and safe_float(str(row["stim_current_ma"])) < 5.0
            ],
        ),
        (
            "repeated_label_subset",
            [row for row in run_rows if int(row["repeated_label_available"]) == 1],
        ),
        (
            "unique_label_subset",
            [row for row in run_rows if int(row["repeated_label_available"]) == 0],
        ),
    ]

    subset_rows: list[dict[str, object]] = []
    for subset_name, subset in subset_definitions:
        if not subset:
            continue
        subset_rows.append(
            {
                "subset": subset_name,
                "n_runs": len(subset),
                "median_t1w_error_mm": percentile(
                    [float(row["t1w_error_mm"]) for row in subset], 50
                ),
                "p75_t1w_error_mm": percentile(
                    [float(row["t1w_error_mm"]) for row in subset], 75
                ),
                "median_mni_error_mm": percentile(
                    [float(row["mni_error_mm"]) for row in subset], 50
                ),
                "p75_mni_error_mm": percentile(
                    [float(row["mni_error_mm"]) for row in subset], 75
                ),
                "median_expected_random_t1w_error_mm": percentile(
                    [float(row["expected_random_t1w_error_mm"]) for row in subset], 50
                ),
                "median_oracle_t1w_error_mm": percentile(
                    [float(row["oracle_t1w_error_mm"]) for row in subset], 50
                ),
                "repeated_label_top1_accuracy": (
                    sum(
                        int(row["repeated_label_top1_correct"])
                        for row in subset
                        if row["repeated_label_top1_correct"] != ""
                    )
                    / max(
                        1, sum(int(row["repeated_label_available"]) for row in subset)
                    )
                ),
            }
        )

    depth_rows: list[dict[str, object]] = []
    for depth_bin in ("shallow_tertile", "middle_tertile", "deep_tertile"):
        subset = [row for row in run_rows if row["depth_proxy_bin"] == depth_bin]
        depth_rows.append(
            {
                "depth_proxy_bin": depth_bin,
                "n_runs": len(subset),
                "median_true_depth_proxy_mm": percentile(
                    [float(row["true_depth_proxy_mm"]) for row in subset], 50
                ),
                "median_t1w_error_mm": percentile(
                    [float(row["t1w_error_mm"]) for row in subset], 50
                ),
                "p75_t1w_error_mm": percentile(
                    [float(row["t1w_error_mm"]) for row in subset], 75
                ),
                "median_expected_random_t1w_error_mm": percentile(
                    [float(row["expected_random_t1w_error_mm"]) for row in subset], 50
                ),
                "median_improvement_over_random_mm": percentile(
                    [float(row["t1w_improvement_over_random_mm"]) for row in subset], 50
                ),
            }
        )

    summary = {
        "n_subjects": len(rows_by_subject),
        "n_runs": len(run_rows),
        "method": args.method,
        "top_k": args.top_k if args.method != "hard_nn" else 1,
        "temperature": args.temperature if args.method == "soft_knn" else None,
        "lle_regularization": (
            args.lle_regularization if args.method == "lle_knn" else None
        ),
        "primary_metric": "within_subject_topographic_retrieval_t1w_error_mm",
        "median_t1w_error_mm": percentile(
            [float(row["t1w_error_mm"]) for row in run_rows], 50
        ),
        "p75_t1w_error_mm": percentile(
            [float(row["t1w_error_mm"]) for row in run_rows], 75
        ),
        "median_mni_error_mm": percentile(
            [float(row["mni_error_mm"]) for row in run_rows], 50
        ),
        "p75_mni_error_mm": percentile(
            [float(row["mni_error_mm"]) for row in run_rows], 75
        ),
        "median_expected_random_t1w_error_mm": percentile(
            [float(row["expected_random_t1w_error_mm"]) for row in run_rows], 50
        ),
        "median_oracle_t1w_error_mm": percentile(
            [float(row["oracle_t1w_error_mm"]) for row in run_rows], 50
        ),
        "repeated_label_top1_accuracy": (
            sum(
                int(row["repeated_label_top1_correct"])
                for row in run_rows
                if row["repeated_label_top1_correct"] != ""
            )
            / max(1, sum(int(row["repeated_label_available"]) for row in run_rows))
        ),
        "repeated_label_n_runs": sum(
            int(row["repeated_label_available"]) for row in run_rows
        ),
        "median_improvement_over_random_mm": percentile(
            [float(row["t1w_improvement_over_random_mm"]) for row in run_rows], 50
        ),
        "depth_proxy_q33_mm": depth_q33,
        "depth_proxy_q67_mm": depth_q67,
        "feature_manifest": str(feature_manifest_path),
        "results_dir": str(results_dir),
    }

    summary_lines = [
        "# SPES fixed retrieval results",
        "",
        f"Method: {args.method}, k={args.top_k}, temperature={args.temperature}.",
        "Within-participant retrieval; each query run is excluded from its candidate set.",
        f"Median T1w error: {summary['median_t1w_error_mm']:.2f} mm.",
        f"75th-percentile T1w error: {summary['p75_t1w_error_mm']:.2f} mm.",
    ]

    write_tsv(results_dir / "run_metrics.tsv", list(run_rows[0].keys()), run_rows)
    write_tsv(
        results_dir / "subject_summary.tsv", list(subject_rows[0].keys()), subject_rows
    )
    write_tsv(
        results_dir / "subset_summary.tsv", list(subset_rows[0].keys()), subset_rows
    )
    write_tsv(results_dir / "depth_summary.tsv", list(depth_rows[0].keys()), depth_rows)
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (results_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
