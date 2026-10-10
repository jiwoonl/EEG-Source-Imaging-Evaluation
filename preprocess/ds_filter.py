#!/usr/bin/env python3
"""Create a low-frequency filtered ds004752 beamforming cache from existing common-label tensors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt


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


def build_filter(
    sample_rate_hz: float, low_hz: float, high_hz: float, order: int
) -> tuple[np.ndarray, str]:
    nyquist = sample_rate_hz / 2.0
    if low_hz <= 0 and high_hz >= nyquist:
        raise ValueError("Filter must limit at least one side of the band.")
    if low_hz <= 0:
        return (
            butter(order, high_hz, btype="lowpass", fs=sample_rate_hz, output="sos"),
            "lowpass",
        )
    if high_hz >= nyquist:
        return (
            butter(order, low_hz, btype="highpass", fs=sample_rate_hz, output="sos"),
            "highpass",
        )
    return (
        butter(
            order, [low_hz, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos"
        ),
        "bandpass",
    )


def filter_trials(trials: np.ndarray, sos: np.ndarray) -> np.ndarray:
    filtered = sosfiltfilt(sos, trials.astype(np.float64), axis=-1)
    return filtered.astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-cache-root",
        type=Path,
        default=repo_root / "data/cache/ds004752/beamforming_common",
    )
    parser.add_argument(
        "--input-derived-dir",
        type=Path,
        default=repo_root / "data/derived",
    )
    parser.add_argument(
        "--output-cache-root",
        type=Path,
        default=repo_root / "data/cache/ds004752/beamforming_common_lowfreq_1to12hz",
    )
    parser.add_argument(
        "--output-derived-dir",
        type=Path,
        default=repo_root / "data/derived/ds004752_beamforming_lowfreq_1to12hz",
    )
    parser.add_argument("--low-hz", type=float, default=1.0)
    parser.add_argument("--high-hz", type=float, default=12.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_cache_root = args.input_cache_root.resolve()
    input_derived_dir = args.input_derived_dir.resolve()
    output_cache_root = args.output_cache_root.resolve()
    output_derived_dir = args.output_derived_dir.resolve()

    cache_manifest = read_tsv(
        input_derived_dir / "ds004752_beamforming_cache_manifest.tsv"
    )
    trial_manifest = read_tsv(
        input_derived_dir / "ds004752_beamforming_trial_manifest.tsv"
    )
    split_manifest = read_tsv(
        input_derived_dir / "ds004752_beamforming_loso_trial_split.tsv"
    )

    session_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for session_row in cache_manifest:
        scalp_in = input_cache_root / session_row["scalp_npy"]
        source_in = input_cache_root / session_row["source_npy"]
        metadata_in = input_cache_root / session_row["metadata_json"]

        scalp_out = output_cache_root / session_row["scalp_npy"]
        source_out = output_cache_root / session_row["source_npy"]
        metadata_out = output_cache_root / session_row["metadata_json"]

        if args.overwrite or not (
            scalp_out.exists() and source_out.exists() and metadata_out.exists()
        ):
            metadata = json.loads(metadata_in.read_text(encoding="utf-8"))
            sample_rate_hz = float(metadata["target_fsample"])
            sos, filter_type = build_filter(
                sample_rate_hz, args.low_hz, args.high_hz, args.order
            )
            scalp_trials = np.load(scalp_in)
            source_trials = np.load(source_in)

            scalp_filtered = filter_trials(scalp_trials, sos)
            source_filtered = filter_trials(source_trials, sos)

            metadata["filter"] = {
                "type": filter_type,
                "low_hz": args.low_hz,
                "high_hz": args.high_hz,
                "order": args.order,
                "library": "scipy.signal.sosfiltfilt",
            }

            scalp_out.parent.mkdir(parents=True, exist_ok=True)
            source_out.parent.mkdir(parents=True, exist_ok=True)
            metadata_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(scalp_out, scalp_filtered.astype(np.float32))
            np.save(source_out, source_filtered.astype(np.float32))
            metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"filtered {session_row['participant_id']} {session_row['phase']}")
        else:
            print(
                f"reusing filtered cache for {session_row['participant_id']} {session_row['phase']}"
            )

        session_copy = dict(session_row)
        session_copy["scalp_npy"] = relpath(scalp_out, output_cache_root)
        session_copy["source_npy"] = relpath(source_out, output_cache_root)
        session_copy["metadata_json"] = relpath(metadata_out, output_cache_root)
        session_copy["filter_low_hz"] = args.low_hz
        session_copy["filter_high_hz"] = args.high_hz
        session_copy["filter_order"] = args.order
        session_rows.append(session_copy)

    for row in trial_manifest:
        row_copy = dict(row)
        row_copy["scalp_npy"] = row["scalp_npy"]
        row_copy["source_npy"] = row["source_npy"]
        row_copy["metadata_json"] = row["metadata_json"]
        row_copy["filter_low_hz"] = args.low_hz
        row_copy["filter_high_hz"] = args.high_hz
        row_copy["filter_order"] = args.order
        trial_rows.append(row_copy)

    for row in split_manifest:
        row_copy = dict(row)
        row_copy["scalp_npy"] = row["scalp_npy"]
        row_copy["source_npy"] = row["source_npy"]
        row_copy["metadata_json"] = row["metadata_json"]
        row_copy["filter_low_hz"] = args.low_hz
        row_copy["filter_high_hz"] = args.high_hz
        row_copy["filter_order"] = args.order
        split_rows.append(row_copy)

    write_tsv(
        output_derived_dir / "ds004752_beamforming_cache_manifest.tsv",
        list(session_rows[0].keys()),
        session_rows,
    )
    write_tsv(
        output_derived_dir / "ds004752_beamforming_trial_manifest.tsv",
        list(trial_rows[0].keys()),
        trial_rows,
    )
    write_tsv(
        output_derived_dir / "ds004752_beamforming_loso_trial_split.tsv",
        list(split_rows[0].keys()),
        split_rows,
    )

    for name in [
        "ds004752_beamforming_common_scalp_labels.txt",
        "ds004752_beamforming_common_source_labels.txt",
    ]:
        (output_derived_dir / name).write_text(
            (input_derived_dir / name).read_text(encoding="utf-8"), encoding="utf-8"
        )

    band_spec = {
        "input_cache_root": str(input_cache_root),
        "output_cache_root": str(output_cache_root),
        "low_hz": args.low_hz,
        "high_hz": args.high_hz,
        "order": args.order,
    }
    (output_derived_dir / "band_spec.json").write_text(
        json.dumps(band_spec, indent=2), encoding="utf-8"
    )

    print("Wrote low-frequency beamforming cache assets:")
    for name in [
        "ds004752_beamforming_cache_manifest.tsv",
        "ds004752_beamforming_trial_manifest.tsv",
        "ds004752_beamforming_loso_trial_split.tsv",
        "ds004752_beamforming_common_scalp_labels.txt",
        "ds004752_beamforming_common_source_labels.txt",
        "band_spec.json",
    ]:
        print(output_derived_dir / name)


if __name__ == "__main__":
    main()
