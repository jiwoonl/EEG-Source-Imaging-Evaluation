#!/usr/bin/env python3
"""Evaluate spatial coverage with participant-disjoint calibration splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


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
    sorted_scores = np.sort(np.asarray(scores, dtype=np.float64))
    index = int(np.ceil(coverage * (sorted_scores.size + 1))) - 1
    index = max(0, min(index, sorted_scores.size - 1))
    return float(sorted_scores[index])


def evaluate_split(
    errors: np.ndarray,
    sigmas: np.ndarray,
    calibration_mask: np.ndarray,
    test_mask: np.ndarray,
    coverage: float,
) -> dict[str, float]:
    calibration_scores = errors[calibration_mask] / sigmas[calibration_mask]
    quantile = conformal_quantile(calibration_scores, coverage)
    radii = quantile * sigmas[test_mask]
    covered = errors[test_mask] <= radii
    return {
        "quantile": quantile,
        "empirical_coverage": float(np.mean(covered)),
        "absolute_coverage_error": abs(float(np.mean(covered)) - coverage),
        "median_radius_mm": float(np.median(radii)),
        "p75_radius_mm": float(np.percentile(radii, 75)),
        "mean_radius_mm": float(np.mean(radii)),
        "n_calibration_runs": int(calibration_mask.sum()),
        "n_test_runs": int(test_mask.sum()),
    }


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-metrics",
        type=Path,
        default=repo_root / "data/derived/spes_fixed_retrieval_run_metrics.tsv",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results/spes_calibration",
    )
    parser.add_argument(
        "--coverage-levels",
        type=float,
        nargs="+",
        default=[0.5, 0.8, 0.9, 0.95],
    )
    parser.add_argument(
        "--calibration-subject-fractions",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
    )
    parser.add_argument("--n-repeats", type=int, default=500)
    parser.add_argument("--epsilon-mm", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=34170)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_tsv(args.run_metrics.resolve())
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    subjects = np.asarray([row["participant_id"] for row in rows])
    unique_subjects = np.asarray(sorted(set(subjects.tolist())))
    errors = np.asarray([float(row["t1w_error_mm"]) for row in rows], dtype=np.float64)
    sigmas = np.asarray(
        [max(float(row["raw_spatial_spread_t1w_mm"]), args.epsilon_mm) for row in rows],
        dtype=np.float64,
    )

    loso_rows: list[dict[str, object]] = []
    loso_prediction_rows: list[dict[str, object]] = []
    for holdout_subject in unique_subjects:
        calibration_mask = subjects != holdout_subject
        test_mask = subjects == holdout_subject
        for coverage in args.coverage_levels:
            result = evaluate_split(
                errors, sigmas, calibration_mask, test_mask, coverage
            )
            loso_rows.append(
                {
                    "holdout_subject": holdout_subject,
                    "nominal_coverage": coverage,
                    **result,
                }
            )
            quantile = float(result["quantile"])
            for index in np.flatnonzero(test_mask):
                radius = quantile * sigmas[index]
                loso_prediction_rows.append(
                    {
                        "participant_id": subjects[index],
                        "run_id": rows[index]["run_id"],
                        "nominal_coverage": coverage,
                        "error_mm": errors[index],
                        "raw_spread_mm": sigmas[index],
                        "quantile": quantile,
                        "radius_mm": radius,
                        "covered": int(errors[index] <= radius),
                    }
                )

    loso_summary_rows: list[dict[str, object]] = []
    for coverage in args.coverage_levels:
        selected = [
            row
            for row in loso_prediction_rows
            if float(row["nominal_coverage"]) == coverage
        ]
        covered = np.asarray(
            [int(row["covered"]) for row in selected], dtype=np.float64
        )
        radii = np.asarray(
            [float(row["radius_mm"]) for row in selected], dtype=np.float64
        )
        subject_coverages = np.asarray(
            [
                float(row["empirical_coverage"])
                for row in loso_rows
                if float(row["nominal_coverage"]) == coverage
            ],
            dtype=np.float64,
        )
        loso_summary_rows.append(
            {
                "protocol": "leave_one_subject_out_cross_conformal",
                "nominal_coverage": coverage,
                "pooled_empirical_coverage": float(np.mean(covered)),
                "pooled_absolute_error": abs(float(np.mean(covered)) - coverage),
                "mean_subject_coverage": float(np.mean(subject_coverages)),
                "sd_subject_coverage": float(np.std(subject_coverages)),
                "min_subject_coverage": float(np.min(subject_coverages)),
                "max_subject_coverage": float(np.max(subject_coverages)),
                "median_radius_mm": float(np.median(radii)),
                "p75_radius_mm": float(np.percentile(radii, 75)),
                "n_test_runs": int(covered.size),
                "n_subjects": len(unique_subjects),
            }
        )

    rng = np.random.default_rng(args.seed)
    repeated_rows: list[dict[str, object]] = []
    for fraction in args.calibration_subject_fractions:
        n_calibration = int(round(fraction * len(unique_subjects)))
        n_calibration = max(2, min(n_calibration, len(unique_subjects) - 1))
        for repeat in range(args.n_repeats):
            shuffled = rng.permutation(unique_subjects)
            calibration_subjects = set(shuffled[:n_calibration].tolist())
            calibration_mask = np.asarray(
                [subject in calibration_subjects for subject in subjects],
                dtype=bool,
            )
            test_mask = ~calibration_mask
            for coverage in args.coverage_levels:
                result = evaluate_split(
                    errors, sigmas, calibration_mask, test_mask, coverage
                )
                repeated_rows.append(
                    {
                        "repeat": repeat,
                        "calibration_subject_fraction": fraction,
                        "n_calibration_subjects": n_calibration,
                        "n_test_subjects": len(unique_subjects) - n_calibration,
                        "nominal_coverage": coverage,
                        **result,
                    }
                )

    repeated_summary_rows: list[dict[str, object]] = []
    for fraction in args.calibration_subject_fractions:
        for coverage in args.coverage_levels:
            selected = [
                row
                for row in repeated_rows
                if float(row["calibration_subject_fraction"]) == fraction
                and float(row["nominal_coverage"]) == coverage
            ]
            empirical = np.asarray(
                [float(row["empirical_coverage"]) for row in selected],
                dtype=np.float64,
            )
            radii = np.asarray(
                [float(row["median_radius_mm"]) for row in selected],
                dtype=np.float64,
            )
            repeated_summary_rows.append(
                {
                    "protocol": "repeated_subject_disjoint_split",
                    "calibration_subject_fraction": fraction,
                    "nominal_coverage": coverage,
                    "mean_empirical_coverage": float(np.mean(empirical)),
                    "sd_empirical_coverage": float(np.std(empirical)),
                    "p05_empirical_coverage": float(np.percentile(empirical, 5)),
                    "p95_empirical_coverage": float(np.percentile(empirical, 95)),
                    "mean_absolute_coverage_error": float(
                        np.mean(np.abs(empirical - coverage))
                    ),
                    "median_of_median_radius_mm": float(np.median(radii)),
                    "p05_median_radius_mm": float(np.percentile(radii, 5)),
                    "p95_median_radius_mm": float(np.percentile(radii, 95)),
                    "n_repeats": len(selected),
                }
            )

    write_tsv(results_dir / "loso_subject_folds.tsv", loso_rows)
    write_tsv(results_dir / "loso_run_predictions.tsv", loso_prediction_rows)
    write_tsv(results_dir / "loso_summary.tsv", loso_summary_rows)
    write_tsv(results_dir / "repeated_subject_splits.tsv", repeated_rows)
    write_tsv(results_dir / "repeated_subject_summary.tsv", repeated_summary_rows)

    loso_90 = next(
        row for row in loso_summary_rows if float(row["nominal_coverage"]) == 0.9
    )
    repeated_90 = [
        row for row in repeated_summary_rows if float(row["nominal_coverage"]) == 0.9
    ]
    summary = {
        "source_run_metrics": str(args.run_metrics.resolve()),
        "n_runs": len(rows),
        "n_subjects": len(unique_subjects),
        "coverage_levels": args.coverage_levels,
        "loso_90": loso_90,
        "repeated_90": repeated_90,
        "scope": (
            "Sensitivity of the conformal wrapper with subject-disjoint calibration; "
            "the underlying spatial retrieval predictor is held fixed."
        ),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# SPES calibration sensitivity",
        "",
        f"- Runs / subjects: `{len(rows)}` / `{len(unique_subjects)}`.",
        f"- LOSO-subject 90% pooled coverage: `{float(loso_90['pooled_empirical_coverage']):.3f}`.",
        f"- LOSO-subject 90% median / p75 radius: `{float(loso_90['median_radius_mm']):.2f}` / "
        f"`{float(loso_90['p75_radius_mm']):.2f}` mm.",
        "- Repeated subject-disjoint 90% results:",
    ]
    for row in repeated_90:
        lines.append(
            f"  - calibration fraction `{float(row['calibration_subject_fraction']):.1f}`: "
            f"coverage `{float(row['mean_empirical_coverage']):.3f} +/- "
            f"{float(row['sd_empirical_coverage']):.3f}`, 5--95% "
            f"`[{float(row['p05_empirical_coverage']):.3f}, "
            f"{float(row['p95_empirical_coverage']):.3f}]`, median radius "
            f"`{float(row['median_of_median_radius_mm']):.2f}` mm."
        )
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
