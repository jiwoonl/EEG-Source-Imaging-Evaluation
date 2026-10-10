#!/usr/bin/env python3
"""Evaluate Localize-MI coordinate-label probabilities with nested participant splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from localize_common import (
    AXIS_INDEX,
    binary_brier,
    binary_ece,
    binary_nll,
    load_feature_bundle,
    pooled_auroc,
    stable_sigmoid,
)


MODEL_ORDER = (
    "constant_0p5",
    "train_prevalence",
    "fixed_unweighted_c1",
    "fixed_balanced_c1",
    "nested_logistic",
    "nested_logistic_temperature",
)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def fit_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    c_value: float,
    class_weight: str | None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_x)
    test_z = scaler.transform(test_x)
    model = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        max_iter=5000,
        random_state=0,
    )
    model.fit(train_z, train_y)
    classes = [int(value) for value in model.classes_.tolist()]
    positive_index = classes.index(1)
    probabilities = model.predict_proba(test_z)[:, positive_index]
    decision = np.asarray(model.decision_function(test_z), dtype=np.float64)
    return probabilities.astype(np.float64), decision, classes


def inner_oof_logits(
    train_rows: list[dict[str, Any]],
    train_y: np.ndarray,
    feature_key: str,
    *,
    c_value: float,
    class_weight: str | None,
) -> np.ndarray:
    subjects = np.asarray([str(row["participant_id"]) for row in train_rows])
    x_all = np.asarray([row[feature_key] for row in train_rows], dtype=np.float64)
    oof_logits = np.full(train_y.shape, np.nan, dtype=np.float64)
    for subject in sorted(set(subjects.tolist())):
        val_mask = subjects == subject
        fit_mask = ~val_mask
        if np.unique(train_y[fit_mask]).size < 2:
            prior = float(np.mean(train_y[fit_mask]))
            prior = float(np.clip(prior, 1e-6, 1.0 - 1e-6))
            oof_logits[val_mask] = np.log(prior / (1.0 - prior))
            continue
        _, logits, _ = fit_logistic(
            x_all[fit_mask],
            train_y[fit_mask],
            x_all[val_mask],
            c_value=c_value,
            class_weight=class_weight,
        )
        oof_logits[val_mask] = logits
    if np.isnan(oof_logits).any():
        raise RuntimeError("Nested LOSO did not produce all training-fold logits.")
    return oof_logits


def select_nested_model(
    train_rows: list[dict[str, Any]],
    train_y: np.ndarray,
    feature_key: str,
    c_grid: list[float],
    temperature_grid: list[float],
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for class_weight in (None, "balanced"):
        for c_value in c_grid:
            logits = inner_oof_logits(
                train_rows,
                train_y,
                feature_key,
                c_value=c_value,
                class_weight=class_weight,
            )
            probabilities = stable_sigmoid(logits)
            candidates.append(
                {
                    "c_value": c_value,
                    "class_weight": class_weight,
                    "oof_nll": binary_nll(train_y, probabilities),
                    "oof_logits": logits,
                }
            )
    best = min(
        candidates,
        key=lambda row: (
            float(row["oof_nll"]),
            1 if row["class_weight"] == "balanced" else 0,
            abs(np.log10(float(row["c_value"]))),
        ),
    )
    temperature_rows = [
        {
            "temperature": temperature,
            "oof_nll": binary_nll(
                train_y,
                stable_sigmoid(
                    np.asarray(best["oof_logits"], dtype=np.float64) / temperature
                ),
            ),
        }
        for temperature in temperature_grid
    ]
    best_temperature = min(
        temperature_rows,
        key=lambda row: (float(row["oof_nll"]), float(row["temperature"])),
    )
    return {
        "c_value": float(best["c_value"]),
        "class_weight": best["class_weight"],
        "base_oof_nll": float(best["oof_nll"]),
        "temperature": float(best_temperature["temperature"]),
        "calibrated_oof_nll": float(best_temperature["oof_nll"]),
    }


def metric_row(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    feature_set: str,
    model: str,
    level: str,
    subject: str = "pooled",
) -> dict[str, object]:
    return {
        "level": level,
        "holdout_subject": subject,
        "feature_set": feature_set,
        "model": model,
        "n": int(labels.size),
        "positive_prevalence": float(np.mean(labels)),
        "auroc": pooled_auroc(labels, scores) if np.unique(labels).size == 2 else "",
        "nll": binary_nll(labels, scores),
        "brier": binary_brier(labels, scores),
        "ece": binary_ece(labels, scores),
        "min_probability": float(np.min(scores)),
        "max_probability": float(np.max(scores)),
    }


def bootstrap_nll_delta(
    prediction_rows: list[dict[str, object]],
    *,
    model: str,
    reference_model: str,
    feature_set: str,
    reference_feature_set: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    rows_a = [
        row
        for row in prediction_rows
        if row["model"] == model and row["feature_set"] == feature_set
    ]
    rows_b = [
        row
        for row in prediction_rows
        if row["model"] == reference_model
        and row["feature_set"] == reference_feature_set
    ]
    key = lambda row: (str(row["participant_id"]), str(row["run_id"]))
    map_a = {key(row): row for row in rows_a}
    map_b = {key(row): row for row in rows_b}
    keys = sorted(set(map_a) & set(map_b))
    subjects = np.asarray([item[0] for item in keys])
    labels = np.asarray([int(map_a[item]["label"]) for item in keys], dtype=np.int64)
    scores_a = np.asarray(
        [float(map_a[item]["score"]) for item in keys], dtype=np.float64
    )
    scores_b = np.asarray(
        [float(map_b[item]["score"]) for item in keys], dtype=np.float64
    )
    observed = binary_nll(labels, scores_a) - binary_nll(labels, scores_b)
    unique_subjects = np.asarray(sorted(set(subjects.tolist())))
    draws: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_subjects, size=unique_subjects.size, replace=True)
        indices = np.concatenate(
            [np.flatnonzero(subjects == subject) for subject in sampled]
        )
        draws.append(
            binary_nll(labels[indices], scores_a[indices])
            - binary_nll(labels[indices], scores_b[indices])
        )
    low, high = np.percentile(np.asarray(draws), [2.5, 97.5])
    return {
        "comparison": f"{feature_set}/{model} minus {reference_feature_set}/{reference_model}",
        "delta_nll": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_bootstrap": n_bootstrap,
        "unit": "subject_cluster",
    }


def bridge_analysis(
    prediction_rows: list[dict[str, object]],
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = [
        row
        for row in prediction_rows
        if row["model"] == "nested_logistic_temperature"
        and row["feature_set"] == "source_peak"
    ]
    errors = np.asarray(
        [float(row["spatial_error_mm"]) for row in rows], dtype=np.float64
    )
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.clip(
        np.asarray([float(row["score"]) for row in rows], dtype=np.float64),
        1e-8,
        1.0 - 1e-8,
    )
    losses = -(labels * np.log(scores) + (1 - labels) * np.log(1 - scores))
    subjects = np.asarray([str(row["participant_id"]) for row in rows])
    observed = float(spearmanr(errors, losses).statistic)
    unique_subjects = np.asarray(sorted(set(subjects.tolist())))
    draws: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_subjects, size=unique_subjects.size, replace=True)
        indices = np.concatenate(
            [np.flatnonzero(subjects == subject) for subject in sampled]
        )
        value = float(spearmanr(errors[indices], losses[indices]).statistic)
        if np.isfinite(value):
            draws.append(value)
    low, high = np.percentile(np.asarray(draws), [2.5, 97.5])
    edges = np.quantile(errors, [0.0, 0.25, 0.5, 0.75, 1.0])
    quartile_rows: list[dict[str, object]] = []
    for index in range(4):
        if index == 3:
            mask = (errors >= edges[index]) & (errors <= edges[index + 1])
        else:
            mask = (errors >= edges[index]) & (errors < edges[index + 1])
        quartile_rows.append(
            {
                "spatial_error_quartile": index + 1,
                "n": int(mask.sum()),
                "error_low_mm": float(edges[index]),
                "error_high_mm": float(edges[index + 1]),
                "mean_spatial_error_mm": float(np.mean(errors[mask])),
                "mean_nll": float(np.mean(losses[mask])),
            }
        )
    summary = {
        "analysis": "heldout_spatial_error_vs_log_loss",
        "model": "nested_logistic_temperature",
        "n_runs": len(rows),
        "n_subjects": len(unique_subjects),
        "spearman_rho": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_bootstrap": n_bootstrap,
        "interpretation_scope": "within-Localize-MI spatial diagnostic; not a waveform-recovery bridge",
    }
    return summary, quartile_rows


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-bundle",
        type=Path,
        default=repo_root / "data/derived/localize_features",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results/localize_nested",
    )
    parser.add_argument("--label-axis", choices=("x", "y", "z"), default="y")
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument(
        "--temperature-grid",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=34170)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    feature_rows, manifest = load_feature_bundle(args.feature_bundle.resolve())
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    axis_index = AXIS_INDEX[args.label_axis]
    feature_sets = {"scalp": "scalp_feature", "source_peak": "peak_coord"}
    subjects = sorted({str(row["participant_id"]) for row in feature_rows})
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for holdout_subject in subjects:
        train_rows = [
            row for row in feature_rows if row["participant_id"] != holdout_subject
        ]
        test_rows = [
            row for row in feature_rows if row["participant_id"] == holdout_subject
        ]
        threshold = float(
            np.median([row["true_coord"][axis_index] for row in train_rows])
        )
        train_y = np.asarray(
            [int(row["true_coord"][axis_index] >= threshold) for row in train_rows],
            dtype=np.int64,
        )
        test_y = np.asarray(
            [int(row["true_coord"][axis_index] >= threshold) for row in test_rows],
            dtype=np.int64,
        )
        prevalence = float(np.mean(train_y))
        for feature_set, feature_key in feature_sets.items():
            train_x = np.asarray(
                [row[feature_key] for row in train_rows], dtype=np.float64
            )
            test_x = np.asarray(
                [row[feature_key] for row in test_rows], dtype=np.float64
            )
            nested = select_nested_model(
                train_rows,
                train_y,
                feature_key,
                c_grid=args.c_grid,
                temperature_grid=args.temperature_grid,
            )
            nested_prob, nested_logits, nested_classes = fit_logistic(
                train_x,
                train_y,
                test_x,
                c_value=float(nested["c_value"]),
                class_weight=nested["class_weight"],
            )
            fixed_unweighted, _, unweighted_classes = fit_logistic(
                train_x,
                train_y,
                test_x,
                c_value=1.0,
                class_weight=None,
            )
            fixed_balanced, _, balanced_classes = fit_logistic(
                train_x,
                train_y,
                test_x,
                c_value=1.0,
                class_weight="balanced",
            )
            model_scores = {
                "constant_0p5": np.full(test_y.shape, 0.5, dtype=np.float64),
                "train_prevalence": np.full(test_y.shape, prevalence, dtype=np.float64),
                "fixed_unweighted_c1": fixed_unweighted,
                "fixed_balanced_c1": fixed_balanced,
                "nested_logistic": nested_prob,
                "nested_logistic_temperature": stable_sigmoid(
                    nested_logits / float(nested["temperature"])
                ),
            }
            selection_rows.append(
                {
                    "holdout_subject": holdout_subject,
                    "feature_set": feature_set,
                    "feature_key": feature_key,
                    "n_train": len(train_rows),
                    "train_prevalence": prevalence,
                    "selected_c": nested["c_value"],
                    "selected_class_weight": nested["class_weight"] or "none",
                    "base_oof_nll": nested["base_oof_nll"],
                    "selected_temperature": nested["temperature"],
                    "calibrated_oof_nll": nested["calibrated_oof_nll"],
                    "nested_classes": ",".join(map(str, nested_classes)),
                    "fixed_unweighted_classes": ",".join(map(str, unweighted_classes)),
                    "fixed_balanced_classes": ",".join(map(str, balanced_classes)),
                }
            )
            for test_row, label, row_index in zip(
                test_rows, test_y, range(len(test_rows))
            ):
                spatial_error_mm = 1000.0 * float(
                    np.linalg.norm(
                        np.asarray(test_row["peak_coord"], dtype=np.float64)
                        - np.asarray(test_row["true_coord"], dtype=np.float64)
                    )
                )
                for model, scores in model_scores.items():
                    prediction_rows.append(
                        {
                            "participant_id": holdout_subject,
                            "run_id": str(test_row["run_id"]),
                            "feature_set": feature_set,
                            "feature_key": feature_key,
                            "model": model,
                            "label": int(label),
                            "score": float(scores[row_index]),
                            "label_threshold": threshold,
                            "train_prevalence": prevalence,
                            "spatial_error_mm": spatial_error_mm,
                        }
                    )

    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for feature_set in feature_sets:
        for model in MODEL_ORDER:
            selected = [
                row
                for row in prediction_rows
                if row["feature_set"] == feature_set and row["model"] == model
            ]
            labels = np.asarray([int(row["label"]) for row in selected], dtype=np.int64)
            scores = np.asarray(
                [float(row["score"]) for row in selected], dtype=np.float64
            )
            metric_rows.append(
                metric_row(
                    labels, scores, feature_set=feature_set, model=model, level="pooled"
                )
            )
            for subject in subjects:
                subject_rows = [
                    row for row in selected if row["participant_id"] == subject
                ]
                subject_labels = np.asarray(
                    [int(row["label"]) for row in subject_rows], dtype=np.int64
                )
                subject_scores = np.asarray(
                    [float(row["score"]) for row in subject_rows], dtype=np.float64
                )
                fold_rows.append(
                    metric_row(
                        subject_labels,
                        subject_scores,
                        feature_set=feature_set,
                        model=model,
                        level="subject",
                        subject=subject,
                    )
                )

    rng = np.random.default_rng(args.seed)
    bootstrap_rows = [
        bootstrap_nll_delta(
            prediction_rows,
            model="nested_logistic_temperature",
            reference_model="nested_logistic_temperature",
            feature_set="source_peak",
            reference_feature_set="scalp",
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        ),
        bootstrap_nll_delta(
            prediction_rows,
            model="nested_logistic_temperature",
            reference_model="constant_0p5",
            feature_set="source_peak",
            reference_feature_set="source_peak",
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        ),
        bootstrap_nll_delta(
            prediction_rows,
            model="nested_logistic_temperature",
            reference_model="train_prevalence",
            feature_set="source_peak",
            reference_feature_set="source_peak",
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        ),
    ]

    paired_rows: list[dict[str, object]] = []
    for subject in subjects:
        source = next(
            row
            for row in fold_rows
            if row["holdout_subject"] == subject
            and row["feature_set"] == "source_peak"
            and row["model"] == "nested_logistic_temperature"
        )
        scalp = next(
            row
            for row in fold_rows
            if row["holdout_subject"] == subject
            and row["feature_set"] == "scalp"
            and row["model"] == "nested_logistic_temperature"
        )
        paired_rows.append(
            {
                "holdout_subject": subject,
                "scalp_nll": scalp["nll"],
                "source_nll": source["nll"],
                "delta_nll": float(source["nll"]) - float(scalp["nll"]),
                "source_improved": float(source["nll"]) < float(scalp["nll"]),
            }
        )
    n_wins = sum(bool(row["source_improved"]) for row in paired_rows)
    sign_p = float(
        binomtest(n_wins, len(paired_rows), p=0.5, alternative="greater").pvalue
    )

    bridge_summary, bridge_quartiles = bridge_analysis(
        prediction_rows,
        n_bootstrap=args.n_bootstrap,
        rng=rng,
    )
    write_tsv(results_dir / "prediction_rows.tsv", prediction_rows)
    write_tsv(results_dir / "pooled_metrics.tsv", metric_rows)
    write_tsv(results_dir / "fold_metrics.tsv", fold_rows)
    write_tsv(results_dir / "nested_selection.tsv", selection_rows)
    write_tsv(results_dir / "bootstrap_delta_nll.tsv", bootstrap_rows)
    write_tsv(results_dir / "paired_subject_nll.tsv", paired_rows)
    write_tsv(results_dir / "spatial_error_quartiles.tsv", bridge_quartiles)
    (results_dir / "spatial_error_bridge.json").write_text(
        json.dumps(bridge_summary, indent=2),
        encoding="utf-8",
    )

    pooled = {(row["feature_set"], row["model"]): row for row in metric_rows}
    original = pooled[("scalp", "fixed_balanced_c1")]
    source = pooled[("source_peak", "nested_logistic_temperature")]
    scalp = pooled[("scalp", "nested_logistic_temperature")]
    summary = {
        "bundle": str(args.feature_bundle.resolve()),
        "bundle_manifest": manifest,
        "n_subjects": len(subjects),
        "n_runs": len(feature_rows),
        "original_scalp_reproduced": {
            "auroc": original["auroc"],
            "nll": original["nll"],
            "brier": original["brier"],
            "ece": original["ece"],
        },
        "nested_calibrated_scalp": {
            "auroc": scalp["auroc"],
            "nll": scalp["nll"],
            "brier": scalp["brier"],
            "ece": scalp["ece"],
        },
        "nested_calibrated_source": {
            "auroc": source["auroc"],
            "nll": source["nll"],
            "brier": source["brier"],
            "ece": source["ece"],
        },
        "source_nll_subject_wins": n_wins,
        "source_nll_subject_total": len(paired_rows),
        "source_nll_sign_test_p": sign_p,
        "bootstrap_comparisons": bootstrap_rows,
        "spatial_error_bridge": bridge_summary,
        "audit_findings": [
            "NLL is the sample mean of binary log losses, not a fold sum.",
            "The positive probability column is selected from classifier.classes_ for label 1.",
            "Fixed C=1 balanced logistic regression is included as a diagnostic.",
            "All scaling, hyperparameter selection, and temperature selection use outer-training subjects only.",
        ],
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Localize-MI probability scores",
        "",
        f"- Runs / subjects: `{len(feature_rows)}` / `{len(subjects)}`.",
        f"- Fixed-head scalp diagnostic: AUROC `{float(original['auroc']):.3f}`, NLL `{float(original['nll']):.3f}`.",
        f"- Nested calibrated scalp: NLL `{float(scalp['nll']):.3f}`, Brier `{float(scalp['brier']):.3f}`, pooled AUROC `{float(scalp['auroc']):.3f}`.",
        f"- Nested calibrated source: NLL `{float(source['nll']):.3f}`, Brier `{float(source['brier']):.3f}`, pooled AUROC `{float(source['auroc']):.3f}`.",
        f"- Source NLL wins: `{n_wins}/{len(paired_rows)}` subjects; sign-test `p={sign_p:.5f}`.",
        f"- Spatial error / held-out log-loss Spearman rho: `{bridge_summary['spearman_rho']:.3f}` "
        f"[`{bridge_summary['ci_low']:.3f}`, `{bridge_summary['ci_high']:.3f}`].",
        "",
        "Scores are in `pooled_metrics.tsv`; fold settings and paired losses are saved separately.",
    ]
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
