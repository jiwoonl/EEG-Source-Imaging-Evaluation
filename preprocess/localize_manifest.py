#!/usr/bin/env python3
"""Build a Localize-MI stimulation manifest from the released epoch derivatives."""

from __future__ import annotations

import argparse
import csv
import json
import re
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


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def electrode_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for row in rows:
        coords = {}
        for axis in ["x", "y", "z"]:
            value = safe_float(row.get(axis, ""))
            if value is not None:
                coords[axis] = value
        if len(coords) == 3:
            lookup[row["name"]] = coords
    return lookup


def parse_stim_pair(trial_type: str) -> tuple[str, str, float | None]:
    match = re.match(r"^(?P<pair>.+?)\s+(?P<current>[0-9.]+)mA$", trial_type.strip())
    if not match:
        return "", "", None
    pair_text = match.group("pair")
    current_ma = float(match.group("current"))
    if "-" not in pair_text:
        return pair_text, "", current_ma

    first, second = pair_text.split("-", 1)
    if re.fullmatch(r"[A-Za-z']+[0-9]+", first) and re.fullmatch(r"[0-9]+", second):
        prefix = re.match(r"([A-Za-z']+)", first).group(1)
        second = f"{prefix}{second}"
    return first, second, current_ma


def midpoint(
    lookup: dict[str, dict[str, float]],
    contact_1: str,
    contact_2: str,
) -> tuple[float | None, float | None, float | None]:
    if contact_1 not in lookup or contact_2 not in lookup:
        return None, None, None
    a = lookup[contact_1]
    b = lookup[contact_2]
    return ((a["x"] + b["x"]) / 2.0, (a["y"] + b["y"]) / 2.0, (a["z"] + b["z"]) / 2.0)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "data/raw/localize-mi",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=repo_root / "data/derived/localize_mi_stimulation_manifest.tsv",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    output_path = args.output_path.resolve()

    rows: list[dict[str, object]] = []
    eeg_epoch_paths = sorted(
        dataset_root.glob("derivatives/epochs/sub-*/eeg/*_epochs.npy")
    )
    for eeg_epochs_npy in eeg_epoch_paths:
        participant_id = eeg_epochs_npy.parts[-3]
        stem = eeg_epochs_npy.name.replace("_epochs.npy", "")
        run_match = re.search(r"(run-[0-9]+)", stem)
        run_id = run_match.group(1) if run_match else ""

        eeg_dir = eeg_epochs_npy.parent
        subject_epochs_root = eeg_dir.parent
        ieeg_dir = subject_epochs_root / "ieeg"

        eeg_epochs_tsv = eeg_dir / f"{stem}_epochs.tsv"
        eeg_epochs_json = eeg_dir / f"{stem}_epochs.json"
        eeg_channels_tsv = eeg_dir / f"{stem.replace('_epochs', '')}_channels.tsv"
        if not eeg_channels_tsv.exists():
            eeg_channels_tsv = eeg_dir / f"{stem}_channels.tsv"

        eeg_electrodes_tsv = eeg_dir / f"{participant_id}_task-seegstim_electrodes.tsv"
        t1w_electrodes_tsv = (
            ieeg_dir / f"{participant_id}_task-seegstim_space-T1w_electrodes.tsv"
        )
        mni_electrodes_tsv = (
            ieeg_dir
            / f"{participant_id}_task-seegstim_space-MNI152NLin2009aSym_electrodes.tsv"
        )
        surface_electrodes_tsv = (
            ieeg_dir / f"{participant_id}_task-seegstim_space-surface_electrodes.tsv"
        )

        epoch_rows = read_tsv(eeg_epochs_tsv)
        channels_rows = read_tsv(eeg_channels_tsv)
        epochs_meta = json.loads(eeg_epochs_json.read_text(encoding="utf-8"))
        t1w_lookup = electrode_lookup(read_tsv(t1w_electrodes_tsv))
        mni_lookup = electrode_lookup(read_tsv(mni_electrodes_tsv))
        surface_lookup = electrode_lookup(read_tsv(surface_electrodes_tsv))

        trial_types = sorted({row["trial_type"] for row in epoch_rows})
        trial_type = trial_types[0] if len(trial_types) == 1 else "; ".join(trial_types)
        contact_1, contact_2, current_ma = parse_stim_pair(trial_type)
        t1w_mid_x, t1w_mid_y, t1w_mid_z = midpoint(t1w_lookup, contact_1, contact_2)
        mni_mid_x, mni_mid_y, mni_mid_z = midpoint(mni_lookup, contact_1, contact_2)
        surface_mid_x, surface_mid_y, surface_mid_z = midpoint(
            surface_lookup, contact_1, contact_2
        )

        epochs = np.load(eeg_epochs_npy, mmap_mode="r")
        n_trials, n_channels, n_timepoints = epochs.shape
        good_channels = sum(
            1 for row in channels_rows if row.get("status", "good") == "good"
        )
        bad_channels = sum(
            1 for row in channels_rows if row.get("status", "") != "good"
        )
        sampling_freqs = sorted(
            {
                safe_float(row.get("sampling_frequency", ""))
                for row in channels_rows
                if safe_float(row.get("sampling_frequency", "")) is not None
            }
        )

        first_epoch = epoch_rows[0] if epoch_rows else {}
        rows.append(
            {
                "participant_id": participant_id,
                "run_id": run_id,
                "trial_type": trial_type,
                "stim_contact_1": contact_1,
                "stim_contact_2": contact_2,
                "stim_current_ma": current_ma,
                "n_trials": n_trials,
                "n_eeg_channels": n_channels,
                "n_good_channels": good_channels,
                "n_bad_channels": bad_channels,
                "n_timepoints": n_timepoints,
                "sampling_frequency_hz": (
                    sampling_freqs[0]
                    if len(sampling_freqs) == 1
                    else ";".join(str(value) for value in sampling_freqs)
                ),
                "epoch_duration_s": first_epoch.get("duration", ""),
                "zero_time_s": first_epoch.get("zero_time", ""),
                "baseline_corrected": epochs_meta.get("BaselineCorrection", ""),
                "baseline_period_start_s": (
                    epochs_meta.get("BaselinePeriod", ["", ""])[0]
                    if epochs_meta.get("BaselinePeriod")
                    else ""
                ),
                "baseline_period_end_s": (
                    epochs_meta.get("BaselinePeriod", ["", ""])[1]
                    if epochs_meta.get("BaselinePeriod")
                    else ""
                ),
                "raw_source_path": epochs_meta.get("Sources", ""),
                "t1w_mid_x": t1w_mid_x,
                "t1w_mid_y": t1w_mid_y,
                "t1w_mid_z": t1w_mid_z,
                "mni_mid_x": mni_mid_x,
                "mni_mid_y": mni_mid_y,
                "mni_mid_z": mni_mid_z,
                "surface_mid_x": surface_mid_x,
                "surface_mid_y": surface_mid_y,
                "surface_mid_z": surface_mid_z,
                "eeg_epochs_npy": relpath(eeg_epochs_npy, dataset_root),
                "eeg_epochs_tsv": relpath(eeg_epochs_tsv, dataset_root),
                "eeg_epochs_json": relpath(eeg_epochs_json, dataset_root),
                "eeg_channels_tsv": relpath(eeg_channels_tsv, dataset_root),
                "eeg_electrodes_tsv": relpath(eeg_electrodes_tsv, dataset_root),
                "t1w_electrodes_tsv": relpath(t1w_electrodes_tsv, dataset_root),
                "mni_electrodes_tsv": relpath(mni_electrodes_tsv, dataset_root),
                "surface_electrodes_tsv": relpath(surface_electrodes_tsv, dataset_root),
                "has_t1w_midpoint": int(t1w_mid_x is not None),
                "has_mni_midpoint": int(mni_mid_x is not None),
                "has_surface_midpoint": int(surface_mid_x is not None),
            }
        )

    if not rows:
        raise RuntimeError(f"No Localize-MI epoch files found under {dataset_root}")

    fieldnames = [
        "participant_id",
        "run_id",
        "trial_type",
        "stim_contact_1",
        "stim_contact_2",
        "stim_current_ma",
        "n_trials",
        "n_eeg_channels",
        "n_good_channels",
        "n_bad_channels",
        "n_timepoints",
        "sampling_frequency_hz",
        "epoch_duration_s",
        "zero_time_s",
        "baseline_corrected",
        "baseline_period_start_s",
        "baseline_period_end_s",
        "raw_source_path",
        "t1w_mid_x",
        "t1w_mid_y",
        "t1w_mid_z",
        "mni_mid_x",
        "mni_mid_y",
        "mni_mid_z",
        "surface_mid_x",
        "surface_mid_y",
        "surface_mid_z",
        "has_t1w_midpoint",
        "has_mni_midpoint",
        "has_surface_midpoint",
        "eeg_epochs_npy",
        "eeg_epochs_tsv",
        "eeg_epochs_json",
        "eeg_channels_tsv",
        "eeg_electrodes_tsv",
        "t1w_electrodes_tsv",
        "mni_electrodes_tsv",
        "surface_electrodes_tsv",
    ]
    write_tsv(output_path, fieldnames, rows)

    summary = {
        "n_runs": len(rows),
        "n_subjects": len({row["participant_id"] for row in rows}),
        "n_runs_with_t1w_midpoint": sum(int(row["has_t1w_midpoint"]) for row in rows),
        "n_runs_with_mni_midpoint": sum(int(row["has_mni_midpoint"]) for row in rows),
        "n_runs_with_surface_midpoint": sum(
            int(row["has_surface_midpoint"]) for row in rows
        ),
    }
    print(json.dumps(summary, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
