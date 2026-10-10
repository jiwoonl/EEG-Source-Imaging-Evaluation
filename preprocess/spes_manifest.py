"""Build SPES run metadata and stimulation coordinates from epoch derivatives."""

import argparse
import csv
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


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_trial_type(text: str) -> dict[str, object]:
    match = re.match(
        r"^(?P<stim_label>.+?)\s+(?P<current_ma>[0-9.]+)ma\s+(?P<pulse_width_ms>[0-9.]+)ms\s+(?P<frequency_hz>[0-9.]+)hz\s+(?P<angle_label>\S+)\s+(?P<tissue_label>\S+)$",
        text.strip(),
    )
    if match:
        return {
            "stim_label": match.group("stim_label"),
            "stim_current_ma": float(match.group("current_ma")),
            "pulse_width_ms": float(match.group("pulse_width_ms")),
            "stim_frequency_hz": float(match.group("frequency_hz")),
            "angle_label": match.group("angle_label"),
            "tissue_label": match.group("tissue_label"),
        }

    tokens = text.strip().split()
    stim_label = tokens[0] if tokens else text.strip()
    current_ma = ""
    pulse_width_ms = ""
    stim_frequency_hz = ""
    angle_label = ""
    tissue_label = ""
    trailing_tokens: list[str] = []

    for token in tokens[1:]:
        if current_ma == "":
            current_match = re.search(r"([0-9.]+)(?:ms)?ma$", token)
            if current_match:
                current_ma = float(current_match.group(1))
                continue
        if pulse_width_ms == "":
            pulse_match = re.search(r"([0-9.]+)ms$", token)
            if pulse_match:
                pulse_width_ms = float(pulse_match.group(1))
                continue
        if stim_frequency_hz == "":
            freq_match = re.search(r"([0-9.]+)hz$", token)
            if freq_match:
                stim_frequency_hz = float(freq_match.group(1))
                continue
        trailing_tokens.append(token)

    if trailing_tokens:
        angle_label = trailing_tokens[0]
    if len(trailing_tokens) > 1:
        tissue_label = trailing_tokens[1]

    return {
        "stim_label": stim_label,
        "stim_current_ma": current_ma,
        "pulse_width_ms": pulse_width_ms,
        "stim_frequency_hz": stim_frequency_hz,
        "angle_label": angle_label,
        "tissue_label": tissue_label,
    }


def electrode_lookup(
    rows: list[dict[str, str]],
) -> dict[str, tuple[float, float, float]]:
    lookup: dict[str, tuple[float, float, float]] = {}
    for row in rows:
        x = safe_float(row.get("x", ""))
        y = safe_float(row.get("y", ""))
        z = safe_float(row.get("z", ""))
        if x is None or y is None or z is None:
            continue
        lookup[row["name"]] = (x, y, z)
    return lookup


def build_manifest(dataset_root):
    rows = []
    for path in sorted(dataset_root.glob("derivatives/epochs/sub-*/eeg/*_epochs.npy")):
        subject = path.parts[-3]
        stem = path.name.removesuffix("_epochs.npy")
        run = re.search(r"run-[0-9]+", stem)
        if run is None:
            raise ValueError(f"Missing run identifier: {path}")
        epochs = read_tsv(path.with_suffix(".tsv"))
        channels = read_tsv(path.parent / f"{stem}_channels.tsv")
        trial_types = sorted(
            {row["trial_type"].strip() for row in epochs if row["trial_type"].strip()}
        )
        if len(trial_types) != 1:
            raise ValueError(f"Expected one stimulation type in {path}")
        parsed = parse_trial_type(trial_types[0])
        ieeg = path.parent.parent / "ieeg"
        coords = {}
        for space, pattern in [
            ("t1w", "*space-T1w_electrodes.tsv"),
            ("mni", "*space-MNI152NLin2009aSym_electrodes.tsv"),
        ]:
            matches = sorted(ieeg.glob(pattern))
            if len(matches) != 1:
                raise ValueError(f"Expected one {space} electrode table in {ieeg}")
            lookup = electrode_lookup(read_tsv(matches[0]))
            value = lookup.get(parsed["stim_label"])
            for axis, v in zip("xyz", value or ("", "", "")):
                coords[f"{space}_{axis}"] = v
        frequencies = {row["sampling_frequency"] for row in channels}
        if len(frequencies) != 1 or not epochs:
            raise ValueError(f"Missing or inconsistent epoch metadata: {path}")
        shape = np.load(path, mmap_mode="r").shape
        if len(shape) != 3 or shape[0] != len(epochs) or shape[1] != len(channels):
            raise ValueError(f"Epoch/channel metadata does not match {path}")
        rows.append(
            {
                "participant_id": subject,
                "run_id": run.group(),
                "task": "ccepcoreg",
                "trial_type": trial_types[0],
                **parsed,
                "n_eeg_epochs": shape[0],
                "usable_eeg_epochs": shape[0],
                "eeg_sampling_frequency": next(iter(frequencies)),
                "zero_time_s": epochs[0]["zero_time"],
                **coords,
                "eeg_epochs_npy": path.relative_to(dataset_root).as_posix(),
                "eeg_channels_tsv": (path.parent / f"{stem}_channels.tsv")
                .relative_to(dataset_root)
                .as_posix(),
            }
        )
    if not rows:
        raise ValueError(f"No SPES epoch files found under {dataset_root}")
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=root / "data/raw/spes-hd-eeg-seeg"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=root / "data/derived/spes_spatial_validation_manifest.tsv",
    )
    args = parser.parse_args()
    rows = build_manifest(args.dataset_root.resolve())
    write_tsv(args.output_path, list(rows[0]), rows)
    print(f"Wrote {len(rows)} runs to {args.output_path}")


if __name__ == "__main__":
    main()
