"""Build the RMS topographies used in the reported primary SPES evaluation."""

from pathlib import Path

try:
    from .spes_cache import build_parser, cache_features
except ImportError:
    from spes_cache import build_parser, cache_features


def main():
    root = Path(__file__).resolve().parents[1]
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(
        feature_mode="rms_topography",
        preserve_amplitude=False,
        spatial_smoothing_sigma_mm=0.0,
        cache_root=root / "data/cache/spes/primary_rms",
        output_manifest=root / "data/derived/spes_raw_feature_manifest.tsv",
    )
    args = parser.parse_args()
    if args.feature_mode != "rms_topography" or args.spatial_smoothing_sigma_mm != 0:
        parser.error(
            "Primary features require RMS topography without spatial smoothing."
        )
    if args.preserve_amplitude:
        parser.error("The reported primary inputs use centering and unit-norm scaling.")
    cache_features(args)


if __name__ == "__main__":
    main()
