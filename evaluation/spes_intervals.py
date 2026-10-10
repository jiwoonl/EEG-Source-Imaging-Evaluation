"""Compute participant-cluster intervals from SPES held-out prediction rows."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def clustered_interval(groups, seed, n_bootstrap=10000):
    subjects = sorted(groups)
    sums = np.asarray([np.sum(groups[s]) for s in subjects])
    sizes = np.asarray([len(groups[s]) for s in subjects])
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(subjects), size=(n_bootstrap, len(subjects)), replace=True)
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def summarize(rows, n_bootstrap=10000):
    models = {}
    for row in rows:
        key = (row["participant_id"], row["run_id"])
        model = models.setdefault(row["model"], {})
        if key in model:
            raise ValueError(
                f"Duplicate participant/run for model {row['model']}: {key}"
            )
        model[key] = row
    ridge = models["ridge"]
    prior = models["coordinate_mean"]
    if ridge.keys() != prior.keys():
        raise ValueError("Ridge and coordinate-prior participant/run pairs differ.")
    coverage, differences = {}, {}
    errors, radii = [], []
    for key in sorted(ridge):
        row = ridge[key]
        subject = key[0]
        coverage.setdefault(subject, []).append(float(row["covered"]))
        differences.setdefault(subject, []).append(
            float(prior[key]["error_mm"]) - float(row["error_mm"])
        )
        errors.append(float(row["error_mm"]))
        radii.append(float(row["radius_mm"]))
    return {
        "n_participants": len(coverage),
        "n_runs": len(ridge),
        "coverage": float(np.mean([x for values in coverage.values() for x in values])),
        "coverage_ci": clustered_interval(coverage, 34172, n_bootstrap),
        "median_error_mm": float(np.median(errors)),
        "median_radius_mm": float(np.median(radii)),
        "mean_error_reduction_mm": float(
            np.mean([x for values in differences.values() for x in values])
        ),
        "error_reduction_ci_mm": clustered_interval(differences, 34270, n_bootstrap),
        "n_bootstrap": n_bootstrap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be positive")
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    print(json.dumps(summarize(rows, args.n_bootstrap), indent=2))


if __name__ == "__main__":
    main()
