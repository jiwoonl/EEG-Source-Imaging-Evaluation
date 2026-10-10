"""Build native surface-MRI feature bundles for Localize-MI."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.localize_common import build_features, read_tsv, save_feature_bundle


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "data/derived/localize_mi_stimulation_manifest.tsv",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=root / "data/raw/localize-mi"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "data/derived/localize_features"
    )
    args = parser.parse_args()
    rows = read_tsv(args.manifest)
    if not rows or any(row.get("has_surface_midpoint") == "0" for row in rows):
        parser.error(
            "The manifest must contain runs with native surface-MRI midpoints."
        )
    settings = dict(
        coordinate_space="surface",
        post_start_s=0.0,
        post_end_s=0.01,
        ridge_lambda=1e-6,
        softmax_temperature=3.0,
    )
    features = build_features(
        manifest_rows=rows, dataset_root=args.dataset_root, **settings
    )
    save_feature_bundle(
        args.output_dir,
        features,
        {
            "scalp_lowfreq": "scalp_feature",
            "reconstructed_source_lowfreq": "peak_coord",
            "oracle_source_lowfreq": "true_coord",
        },
        metadata=settings | {"label_axis": "y", "source_feature": "peak_coord"},
    )
    print(f"Wrote {len(features)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
