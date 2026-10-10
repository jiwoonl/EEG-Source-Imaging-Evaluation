#!/usr/bin/env python3
"""Cache SPES run-level scalp EEG features for spatial retrieval benchmarks."""

from __future__ import annotations

import argparse
import csv
import os
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
    return Path(os.path.relpath(path, root)).as_posix()


def absolute_no_resolve(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "data/derived/spes_spatial_validation_manifest.tsv",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "data/raw/spes-hd-eeg-seeg",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=repo_root / "data/cache/spes/spatial_retrieval_v0",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=repo_root / "data/derived/spes_spatial_retrieval_feature_manifest.tsv",
    )
    parser.add_argument("--baseline-start-s", type=float, default=-0.2)
    parser.add_argument("--baseline-end-s", type=float, default=0.0)
    parser.add_argument("--post-start-s", type=float, default=0.005)
    parser.add_argument("--post-end-s", type=float, default=0.05)
    parser.add_argument("--temporal-downsample", type=int, default=5)
    parser.add_argument("--spatial-smoothing-sigma-mm", type=float, default=0.0)
    parser.add_argument(
        "--feature-mode",
        choices=(
            "flatten",
            "mean_topography",
            "rms_topography",
            "peak_topography",
            "rms_peak_topography",
            "rms_timebins_topography",
            "mean_timebins_topography",
        ),
        default="flatten",
    )
    parser.add_argument("--n-post-bins", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preserve-amplitude",
        action="store_true",
        help="Keep amplitudes without centering or unit-norm scaling.",
    )
    return parser


def cache_features(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest.resolve()
    dataset_root = args.dataset_root.resolve()
    cache_root = absolute_no_resolve(args.cache_root)
    output_manifest = args.output_manifest.resolve()

    manifest_rows = read_tsv(manifest_path)
    rows_by_subject: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        if not row.get("t1w_x"):
            continue
        rows_by_subject.setdefault(row["participant_id"], []).append(row)

    output_rows: list[dict[str, object]] = []

    for participant_id, subject_rows in sorted(rows_by_subject.items()):
        channel_rows_by_run: dict[str, list[dict[str, str]]] = {}
        common_good_channels: set[str] | None = None
        for row in subject_rows:
            channel_rows = read_tsv(dataset_root / row["eeg_channels_tsv"])
            channel_rows_by_run[row["run_id"]] = channel_rows
            good_channels = {
                entry["name"]
                for entry in channel_rows
                if entry.get("status", "good") == "good"
            }
            common_good_channels = (
                good_channels
                if common_good_channels is None
                else common_good_channels & good_channels
            )

        if common_good_channels is None or not common_good_channels:
            continue

        ordered_common_channels = sorted(
            common_good_channels, key=lambda name: int(name[1:])
        )
        smoothing_weights: np.ndarray | None = None
        if args.spatial_smoothing_sigma_mm > 0:
            electrode_rows = read_tsv(
                dataset_root
                / f"derivatives/epochs/{participant_id}/eeg/{participant_id}_task-ccepcoreg_electrodes.tsv"
            )
            coord_by_name = {
                row["name"]: np.asarray(
                    [float(row["x"]), float(row["y"]), float(row["z"])],
                    dtype=np.float64,
                )
                for row in electrode_rows
            }
            coords_mm = np.stack(
                [coord_by_name[name] for name in ordered_common_channels], axis=0
            )
            distances_mm = np.linalg.norm(
                coords_mm[:, None, :] - coords_mm[None, :, :], axis=2
            )
            sigma = float(args.spatial_smoothing_sigma_mm)
            smoothing_weights = np.exp(-(distances_mm**2) / (2.0 * sigma * sigma))
            smoothing_weights /= smoothing_weights.sum(axis=1, keepdims=True) + 1e-12
        channel_txt_path = (
            cache_root / participant_id / f"{participant_id}_common_good_channels.txt"
        )
        channel_txt_path.parent.mkdir(parents=True, exist_ok=True)
        channel_txt_path.write_text(
            "\n".join(ordered_common_channels) + "\n", encoding="utf-8"
        )

        for row in subject_rows:
            channel_rows = channel_rows_by_run[row["run_id"]]
            index_by_name = {
                entry["name"]: idx for idx, entry in enumerate(channel_rows)
            }
            selected_indices = np.asarray(
                [index_by_name[name] for name in ordered_common_channels],
                dtype=np.int64,
            )

            epoch_array = np.load(dataset_root / row["eeg_epochs_npy"], mmap_mode="r")
            evoked = np.asarray(epoch_array.mean(axis=0), dtype=np.float64)[
                selected_indices
            ]

            fs = float(row["eeg_sampling_frequency"])
            zero_time = float(row["zero_time_s"])
            time_axis = np.arange(evoked.shape[1], dtype=np.float64) / fs - zero_time

            baseline_mask = (time_axis >= args.baseline_start_s) & (
                time_axis < args.baseline_end_s
            )
            if np.any(baseline_mask):
                evoked = evoked - evoked[:, baseline_mask].mean(axis=1, keepdims=True)

            post_mask = (time_axis >= args.post_start_s) & (
                time_axis <= args.post_end_s
            )
            post_indices = np.where(post_mask)[0]
            if post_indices.size == 0:
                raise RuntimeError(
                    f"No post-stimulation samples selected for {participant_id} {row['run_id']} "
                    f"with window {args.post_start_s} to {args.post_end_s} s."
                )

            post_window = evoked[:, post_indices]
            windowed = post_window[:, :: args.temporal_downsample]
            if args.feature_mode == "flatten":
                feature = windowed.reshape(-1).astype(np.float32)
            elif args.feature_mode == "mean_topography":
                feature = windowed.mean(axis=1).astype(np.float32)
            elif args.feature_mode == "rms_topography":
                feature = np.sqrt((windowed**2).mean(axis=1)).astype(np.float32)
            elif args.feature_mode == "peak_topography":
                gfp = np.sqrt((windowed**2).mean(axis=0))
                peak_index = int(np.argmax(gfp))
                feature = windowed[:, peak_index].astype(np.float32)
            elif args.feature_mode == "rms_peak_topography":
                gfp = np.sqrt((windowed**2).mean(axis=0))
                peak_index = int(np.argmax(gfp))
                feature = np.concatenate(
                    [
                        np.sqrt((windowed**2).mean(axis=1)).astype(np.float32),
                        windowed[:, peak_index].astype(np.float32),
                    ],
                    axis=0,
                )
            elif args.feature_mode == "rms_timebins_topography":
                feature = np.concatenate(
                    [
                        np.sqrt((post_window[:, bin_indices] ** 2).mean(axis=1)).astype(
                            np.float32
                        )
                        for bin_indices in np.array_split(
                            np.arange(post_window.shape[1]), args.n_post_bins
                        )
                        if bin_indices.size > 0
                    ],
                    axis=0,
                )
            elif args.feature_mode == "mean_timebins_topography":
                feature = np.concatenate(
                    [
                        post_window[:, bin_indices].mean(axis=1).astype(np.float32)
                        for bin_indices in np.array_split(
                            np.arange(post_window.shape[1]), args.n_post_bins
                        )
                        if bin_indices.size > 0
                    ],
                    axis=0,
                )
            else:
                raise ValueError(f"Unsupported feature mode: {args.feature_mode}")
            if smoothing_weights is not None:
                n_channels = len(ordered_common_channels)
                if feature.shape[0] % n_channels != 0:
                    raise RuntimeError(
                        f"Feature length {feature.shape[0]} is not divisible by channel count {n_channels} "
                        f"for {participant_id} {row['run_id']}."
                    )
                feature_blocks = feature.reshape(-1, n_channels).astype(np.float64)
                feature = (
                    (feature_blocks @ smoothing_weights.T)
                    .reshape(-1)
                    .astype(np.float32)
                )
            if not args.preserve_amplitude:
                feature = feature - feature.mean()
                feature_norm = float(np.linalg.norm(feature))
                if feature_norm > 0:
                    feature /= feature_norm

            feature_path = (
                cache_root
                / participant_id
                / f"{participant_id}_{row['run_id']}_feature.npy"
            )
            if args.overwrite or not feature_path.exists():
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(feature_path, feature)

            output_rows.append(
                {
                    "participant_id": participant_id,
                    "run_id": row["run_id"],
                    "task": row["task"],
                    "trial_type": row["trial_type"],
                    "stim_label": row["stim_label"],
                    "stim_current_ma": row["stim_current_ma"],
                    "pulse_width_ms": row["pulse_width_ms"],
                    "stim_frequency_hz": row["stim_frequency_hz"],
                    "angle_label": row["angle_label"],
                    "tissue_label": row["tissue_label"],
                    "n_eeg_epochs": row["n_eeg_epochs"],
                    "usable_eeg_epochs": row["usable_eeg_epochs"],
                    "eeg_sampling_frequency": row["eeg_sampling_frequency"],
                    "subject_common_good_channel_count": len(ordered_common_channels),
                    "feature_length": feature.shape[0],
                    "baseline_start_s": args.baseline_start_s,
                    "baseline_end_s": args.baseline_end_s,
                    "post_start_s": args.post_start_s,
                    "post_end_s": args.post_end_s,
                    "temporal_downsample": args.temporal_downsample,
                    "spatial_smoothing_sigma_mm": args.spatial_smoothing_sigma_mm,
                    "feature_mode": args.feature_mode,
                    "feature_normalization": (
                        "none" if args.preserve_amplitude else "center_unit_norm"
                    ),
                    "n_post_bins": args.n_post_bins,
                    "t1w_x": row["t1w_x"],
                    "t1w_y": row["t1w_y"],
                    "t1w_z": row["t1w_z"],
                    "mni_x": row["mni_x"],
                    "mni_y": row["mni_y"],
                    "mni_z": row["mni_z"],
                    "feature_npy": relpath(feature_path, repo_root),
                    "common_good_channels_txt": relpath(channel_txt_path, repo_root),
                    "source_manifest_row": relpath(manifest_path, repo_root),
                }
            )

    fieldnames = [
        "participant_id",
        "run_id",
        "task",
        "trial_type",
        "stim_label",
        "stim_current_ma",
        "pulse_width_ms",
        "stim_frequency_hz",
        "angle_label",
        "tissue_label",
        "n_eeg_epochs",
        "usable_eeg_epochs",
        "eeg_sampling_frequency",
        "subject_common_good_channel_count",
        "feature_length",
        "baseline_start_s",
        "baseline_end_s",
        "post_start_s",
        "post_end_s",
        "temporal_downsample",
        "spatial_smoothing_sigma_mm",
        "feature_mode",
        "feature_normalization",
        "n_post_bins",
        "t1w_x",
        "t1w_y",
        "t1w_z",
        "mni_x",
        "mni_y",
        "mni_z",
        "feature_npy",
        "common_good_channels_txt",
        "source_manifest_row",
    ]
    write_tsv(output_manifest, fieldnames, output_rows)

    summary = {
        "n_subjects": len({row["participant_id"] for row in output_rows}),
        "n_runs": len(output_rows),
        "cache_root": str(cache_root),
        "output_manifest": str(output_manifest),
        "post_window_s": [args.post_start_s, args.post_end_s],
        "temporal_downsample": args.temporal_downsample,
        "spatial_smoothing_sigma_mm": args.spatial_smoothing_sigma_mm,
        "feature_mode": args.feature_mode,
        "n_post_bins": args.n_post_bins,
        "min_subject_common_channels": min(
            int(row["subject_common_good_channel_count"]) for row in output_rows
        ),
        "max_subject_common_channels": max(
            int(row["subject_common_good_channel_count"]) for row in output_rows
        ),
    }
    print(summary)


def main() -> None:
    cache_features(build_parser().parse_args())


if __name__ == "__main__":
    main()
