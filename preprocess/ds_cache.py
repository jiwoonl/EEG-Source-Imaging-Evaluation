#!/usr/bin/env python3
"""Cache common-label beamforming trials from ds004752 derivatives."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.io as sio


SOURCE_LABEL_ORDER = ["DLPFC", "OFC", "PPC", "AC", "V1"]


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def stack_trials(trials: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(trial, dtype=np.float32) for trial in trials], axis=0)


def resample_trials(
    trials: np.ndarray, source_time: np.ndarray, target_hz: int
) -> tuple[np.ndarray, np.ndarray]:
    duration_s = float(
        source_time[-1] - source_time[0] + (source_time[1] - source_time[0])
    )
    target_samples = int(round(duration_s * target_hz))
    target_time = np.linspace(
        float(source_time[0]),
        float(source_time[-1]),
        target_samples,
        endpoint=True,
        dtype=np.float64,
    )
    out = np.empty((trials.shape[0], trials.shape[1], target_samples), dtype=np.float32)
    for trial_idx in range(trials.shape[0]):
        for channel_idx in range(trials.shape[1]):
            out[trial_idx, channel_idx] = np.interp(
                target_time, source_time, trials[trial_idx, channel_idx]
            ).astype(np.float32)
    return out, target_time


def common_label_order(dataset_root: Path) -> list[str]:
    label_sets: list[set[str]] = []
    first_labels: list[str] | None = None
    for path in sorted(
        dataset_root.glob("sub-*/beamforming/*-task-verbalWM-LCMVsources.mat")
    ):
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        labels = [str(x) for x in mat["LCMV_encoding"].label.tolist()]
        if first_labels is None:
            first_labels = labels
        label_sets.append(set(labels))
    if not label_sets or first_labels is None:
        raise RuntimeError(f"No beamforming .mat files found under {dataset_root}")
    common = set.intersection(*label_sets)
    return [label for label in first_labels if label in common]


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "data/raw/ds004752/derivatives",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "data/cache/ds004752/beamforming_common",
    )
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=repo_root / "data/derived",
    )
    parser.add_argument(
        "--target-hz",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    derived_dir = args.derived_dir.resolve()

    common_labels = common_label_order(dataset_root)
    source_labels = [label for label in SOURCE_LABEL_ORDER if label in common_labels]
    scalp_labels = [label for label in common_labels if label not in source_labels]
    if len(source_labels) != 5:
        raise RuntimeError(f"Expected 5 common source labels, found {source_labels}")

    session_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    loso_rows: list[dict[str, object]] = []

    subjects = sorted(path.name for path in dataset_root.glob("sub-*") if path.is_dir())
    for participant_id in subjects:
        mat_path = (
            dataset_root
            / participant_id
            / "beamforming"
            / f"{participant_id}-task-verbalWM-LCMVsources.mat"
        )
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        for phase_name, phase_key in (
            ("encoding", "LCMV_encoding"),
            ("maintenance", "LCMV_maintenance"),
        ):
            obj = mat[phase_key]
            labels = [str(x) for x in obj.label.tolist()]
            label_to_index = {label: idx for idx, label in enumerate(labels)}
            missing = [label for label in common_labels if label not in label_to_index]
            if missing:
                raise RuntimeError(
                    f"Missing labels for {participant_id} {phase_name}: {missing}"
                )

            cache_dir = output_root / participant_id / phase_name
            scalp_path = cache_dir / "scalp_common_trials.npy"
            source_path = cache_dir / "source_common_trials.npy"
            metadata_path = cache_dir / "metadata.json"
            n_trials = len(obj.trial)

            if args.overwrite or not (
                scalp_path.exists() and source_path.exists() and metadata_path.exists()
            ):
                print(f"building beamforming cache for {participant_id} {phase_name}")
                trial_stack = stack_trials(obj.trial)
                source_time = np.asarray(obj.time[0], dtype=np.float64)
                trial_stack, target_time = resample_trials(
                    trial_stack, source_time, args.target_hz
                )
                scalp_idx = [label_to_index[label] for label in scalp_labels]
                source_idx = [label_to_index[label] for label in source_labels]
                scalp_trials = trial_stack[:, scalp_idx, :]
                source_trials = trial_stack[:, source_idx, :]
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(scalp_path, scalp_trials.astype(np.float32))
                np.save(source_path, source_trials.astype(np.float32))
                metadata = {
                    "participant_id": participant_id,
                    "phase": phase_name,
                    "n_trials": int(n_trials),
                    "original_fsample": int(obj.fsample),
                    "target_fsample": int(args.target_hz),
                    "time_start_s": float(target_time[0]),
                    "time_end_s": float(target_time[-1]),
                    "n_timepoints": int(target_time.shape[0]),
                    "all_labels": labels,
                    "common_labels": common_labels,
                    "scalp_labels": scalp_labels,
                    "source_labels": source_labels,
                }
                metadata_path.write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
            else:
                print(f"reusing beamforming cache for {participant_id} {phase_name}")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            session_rows.append(
                {
                    "participant_id": participant_id,
                    "phase": phase_name,
                    "n_trials": metadata["n_trials"],
                    "original_fsample": metadata["original_fsample"],
                    "target_fsample": metadata["target_fsample"],
                    "time_start_s": metadata["time_start_s"],
                    "time_end_s": metadata["time_end_s"],
                    "n_timepoints": metadata["n_timepoints"],
                    "n_scalp_channels": len(metadata["scalp_labels"]),
                    "n_source_channels": len(metadata["source_labels"]),
                    "scalp_labels": ",".join(metadata["scalp_labels"]),
                    "source_labels": ",".join(metadata["source_labels"]),
                    "scalp_npy": relpath(scalp_path, output_root),
                    "source_npy": relpath(source_path, output_root),
                    "metadata_json": relpath(metadata_path, output_root),
                }
            )

            for trial_index in range(int(metadata["n_trials"])):
                trial_row = {
                    "participant_id": participant_id,
                    "phase": phase_name,
                    "trial_index": trial_index,
                    "original_fsample": metadata["original_fsample"],
                    "target_fsample": metadata["target_fsample"],
                    "n_timepoints": metadata["n_timepoints"],
                    "scalp_npy": relpath(scalp_path, output_root),
                    "source_npy": relpath(source_path, output_root),
                    "metadata_json": relpath(metadata_path, output_root),
                    "scalp_labels": ",".join(metadata["scalp_labels"]),
                    "source_labels": ",".join(metadata["source_labels"]),
                }
                trial_rows.append(trial_row)
                for holdout_subject in subjects:
                    loso_row = dict(trial_row)
                    loso_row["fold_id"] = f"loso_{holdout_subject}"
                    loso_row["holdout_subject"] = holdout_subject
                    loso_row["split"] = (
                        "test" if participant_id == holdout_subject else "train"
                    )
                    loso_rows.append(loso_row)

    write_tsv(
        derived_dir / "ds004752_beamforming_cache_manifest.tsv",
        list(session_rows[0].keys()) if session_rows else [],
        session_rows,
    )
    write_tsv(
        derived_dir / "ds004752_beamforming_trial_manifest.tsv",
        list(trial_rows[0].keys()) if trial_rows else [],
        trial_rows,
    )
    write_tsv(
        derived_dir / "ds004752_beamforming_loso_trial_split.tsv",
        list(loso_rows[0].keys()) if loso_rows else [],
        loso_rows,
    )
    (derived_dir / "ds004752_beamforming_common_scalp_labels.txt").write_text(
        "\n".join(scalp_labels) + "\n", encoding="utf-8"
    )
    (derived_dir / "ds004752_beamforming_common_source_labels.txt").write_text(
        "\n".join(source_labels) + "\n", encoding="utf-8"
    )

    print("Wrote beamforming cache assets:")
    for name in [
        "ds004752_beamforming_cache_manifest.tsv",
        "ds004752_beamforming_trial_manifest.tsv",
        "ds004752_beamforming_loso_trial_split.tsv",
        "ds004752_beamforming_common_scalp_labels.txt",
        "ds004752_beamforming_common_source_labels.txt",
    ]:
        print(derived_dir / name)


if __name__ == "__main__":
    main()
