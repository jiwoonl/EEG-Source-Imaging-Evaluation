"""Synthetic checks for input preparation, participant splits and reported settings."""

import argparse
import ast
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from experiments import ds_ridge, localize_common, localize_nested, spes_localization
from preprocess import spes_cache, spes_manifest
from run_reported import command


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parser_for(path):
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_parser"
    )
    namespace = {
        "__file__": str(path),
        "__doc__": "Test parser",
        "Path": Path,
        "argparse": argparse,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["build_parser"]()


@pytest.mark.parametrize(
    "path", sorted((ROOT / "configs").glob("*.json")), ids=lambda p: p.stem
)
def test_reported_config(path):
    config = json.loads(path.read_text())
    argv = command(config, ROOT)
    parsed = vars(parser_for(Path(argv[1])).parse_args(argv[2:]))
    for key, expected in config["args"].items():
        actual = parsed[key]
        if isinstance(actual, Path):
            actual = actual.as_posix()
        assert actual == expected


def test_reported_budget():
    args = json.loads((ROOT / "configs/ds004752_neural_reported.json").read_text())[
        "args"
    ]
    assert (args["max_epochs"], args["patience"], args["ssl_pretrain_epochs"]) == (
        5,
        2,
        2,
    )
    assert args["seed"] == 29
    assert args["max_train_trials_per_fold"] == args["max_val_trials_per_fold"] == 0


def test_training_scaling():
    train = np.array([[1.0, 5.0], [3.0, 5.0]])
    test = np.array([[1001.0, -10.0]])
    train_z, test_z, mean, std = ds_ridge.zscore_train(train, test)
    np.testing.assert_array_equal(mean, [[2.0, 5.0]])
    np.testing.assert_array_equal(std, [[1.0, 1.0]])
    np.testing.assert_array_equal(train_z.mean(axis=0), [0.0, 0.0])
    np.testing.assert_array_equal(test_z, [[999.0, -15.0]])


def test_participant_splits():
    from experiments import ds_encoders

    subjects = [f"sub-{i:02}" for i in range(15)]
    for held in subjects:
        rows = [{"participant_id": s} for s in subjects if s != held]
        fit, validation, selected = ds_encoders.split_train_rows(rows, subjects, held)
        fit_ids = {r["participant_id"] for r in fit}
        val_ids = {r["participant_id"] for r in validation}
        assert len(fit_ids) == 13 and val_ids == {selected}
        assert fit_ids.isdisjoint(val_ids | {held})
        assert held not in val_ids


def test_inner_participant_exclusion(monkeypatch):
    rows = [
        {"participant_id": str(s), "feature": np.array([s, i])}
        for s in range(4)
        for i in range(4)
    ]
    y = np.tile([0, 1, 0, 1], 4)
    calls = []

    def fit(train_x, train_y, test_x, **kwargs):
        assert set(train_x[:, 0]).isdisjoint(test_x[:, 0])
        calls.append(1)
        return np.full(len(test_x), 0.5), np.zeros(len(test_x)), [0, 1]

    monkeypatch.setattr(localize_nested, "fit_logistic", fit)
    logits = localize_nested.inner_oof_logits(
        rows, y, "feature", c_value=1.0, class_weight=None
    )
    assert len(calls) == 4
    np.testing.assert_array_equal(logits, np.zeros(16))


def test_probability_scores():
    labels = np.array([0, 1])
    scores = np.array([0.25, 0.75])
    assert localize_common.binary_nll(labels, scores) == pytest.approx(-np.log(0.75))
    assert localize_common.binary_brier(labels, scores) == pytest.approx(0.0625)
    assert localize_common.binary_ece(labels, scores) == pytest.approx(0.25)
    assert np.isfinite(localize_common.binary_nll(np.zeros(2), scores))


def test_source_feature_shapes():
    rng = np.random.default_rng(7)
    scores = localize_common.localization_scores(
        rng.normal(size=(8, 11, 3)), np.zeros(8), 1e-6
    )
    np.testing.assert_array_equal(scores, np.zeros(11))
    features = spes_localization.fixed_scalp_features(
        rng.normal(size=32), rng.normal(size=(32, 3)), 1e-3
    )
    assert features.shape == (35,)
    assert np.isfinite(features).all()


def test_quantile_rank():
    scores = np.arange(1.0, 10.0)
    assert spes_localization.conformal_quantile(scores, 0.5) == 5
    assert spes_localization.conformal_quantile(scores, 0.9) == 9
    assert spes_localization.conformal_quantile(scores, 0.95) == 9


@pytest.fixture
def spes_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    eeg = dataset / "derivatives/epochs/sub-01/eeg"
    ieeg = eeg.parent / "ieeg"
    eeg.mkdir(parents=True)
    for space in ["T1w", "MNI152NLin2009aSym"]:
        write_tsv(
            ieeg / f"sub-01_space-{space}_electrodes.tsv",
            [{"name": "A1-2", "x": 0.01, "y": 0.02, "z": 0.03}],
        )
    for run, multiplier in [(1, 1), (2, 2)]:
        stem = f"sub-01_task-ccepcoreg_run-{run:02}"
        epochs = np.full((2, 3, 500), 5.0, dtype=np.float32)
        epochs[:, :, 205:251] += np.array([2.0, 6.0, 100.0])[None, :, None] * multiplier
        np.save(eeg / f"{stem}_epochs.npy", epochs)
        write_tsv(
            eeg / f"{stem}_epochs.tsv",
            [dict(trial_type="A1-2 5ma 0.5ms 0.5hz parallel wh_wh", zero_time=0.2)] * 2,
        )
        write_tsv(
            eeg / f"{stem}_channels.tsv",
            [
                dict(
                    name=f"E{i+1}",
                    status="bad" if run == 2 and i == 2 else "good",
                    sampling_frequency=1000,
                )
                for i in range(3)
            ],
        )
    return dataset


@pytest.mark.parametrize(
    "raw", [False, True], ids=["reported-primary", "unnormalized-diagnostic"]
)
def test_spes_preprocessing(spes_dataset, tmp_path, raw):
    rows = spes_manifest.build_manifest(spes_dataset)
    assert len(rows) == 2
    assert float(rows[0]["mni_x"]) == 0.01
    manifest = tmp_path / "manifest.tsv"
    write_tsv(manifest, rows)
    args = spes_cache.build_parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(spes_dataset),
            "--cache-root",
            str(tmp_path / "cache"),
            "--output-manifest",
            str(tmp_path / "features.tsv"),
            "--feature-mode",
            "rms_topography",
        ]
        + (["--preserve-amplitude"] if raw else [])
    )
    spes_cache.cache_features(args)
    with (tmp_path / "features.tsv").open() as handle:
        features = list(csv.DictReader(handle, delimiter="\t"))
    for run, row in enumerate(features, 1):
        vector = np.load(ROOT / row["feature_npy"])
        expected = np.array([2.0, 6.0], dtype=np.float32) * run
        if not raw:
            expected -= expected.mean()
            expected /= np.linalg.norm(expected)
        np.testing.assert_array_equal(vector, expected)
        assert (ROOT / row["common_good_channels_txt"]).read_text().splitlines() == [
            "E1",
            "E2",
        ]
        assert row["feature_normalization"] == ("none" if raw else "center_unit_norm")


def test_localize_manifest(tmp_path):
    dataset = tmp_path / "localize"
    eeg = dataset / "derivatives/epochs/sub-01/eeg"
    ieeg = eeg.parent / "ieeg"
    eeg.mkdir(parents=True)
    stem = "sub-01_task-seegstim_run-01"
    np.save(eeg / f"{stem}_epochs.npy", np.zeros((2, 2, 20), dtype=np.float32))
    write_tsv(
        eeg / f"{stem}_epochs.tsv",
        [dict(trial_type="A1-2 5mA", duration=0.02, zero_time=0)] * 2,
    )
    write_tsv(
        eeg / f"{stem}_channels.tsv",
        [dict(name=f"E{i}", status="good", sampling_frequency=1000) for i in [1, 2]],
    )
    (eeg / f"{stem}_epochs.json").write_text("{}")
    for space, factor in [("surface", 1), ("T1w", 10), ("MNI152NLin2009aSym", 100)]:
        write_tsv(
            ieeg / f"sub-01_task-seegstim_space-{space}_electrodes.tsv",
            [
                dict(name="A1", x=1 * factor, y=2 * factor, z=3 * factor),
                dict(name="A2", x=3 * factor, y=4 * factor, z=5 * factor),
            ],
        )
    output = tmp_path / "manifest.tsv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "preprocess/localize_manifest.py"),
            "--dataset-root",
            str(dataset),
            "--output-path",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    with output.open() as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert float(row["surface_mid_y"]) == 3
    assert float(row["mni_mid_y"]) == 300
    assert row["n_trials"] == "2"


CLI = sorted(
    p
    for folder in ["experiments", "preprocess", "evaluation"]
    for p in (ROOT / folder).glob("*.py")
    if p.name != "localize_common.py"
) + [ROOT / "run_reported.py"]


@pytest.mark.parametrize("path", CLI, ids=lambda p: p.stem)
def test_cli_help(path):
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_localize_feature_bundle(tmp_path, monkeypatch):
    from preprocess import localize_features

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    epochs = np.zeros((2, 2, 20), dtype=np.float32)
    epochs[:, 1, 5] = 1
    np.save(dataset / "epochs.npy", epochs)
    write_tsv(
        dataset / "channels.tsv",
        [dict(name="E1", status="good"), dict(name="E2", status="good")],
    )
    row = dict(
        participant_id="sub-01",
        run_id="run-01",
        trial_type="A1-2 5mA",
        eeg_epochs_npy="epochs.npy",
        eeg_channels_tsv="channels.tsv",
        sampling_frequency_hz=1000,
        zero_time_s=0,
        surface_mid_x=2,
        surface_mid_y=3,
        surface_mid_z=4,
        mni_mid_x=200,
        mni_mid_y=300,
        mni_mid_z=400,
        has_surface_midpoint=1,
    )
    manifest = tmp_path / "manifest.tsv"
    write_tsv(manifest, [row])
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    gain = np.repeat(np.eye(2, 3)[:, None, :], 2, axis=1)
    monkeypatch.setattr(
        localize_common, "build_source_space", lambda *_: (["E1", "E2"], gain, coords)
    )
    output = tmp_path / "features"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "localize_features.py",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(dataset),
            "--output-dir",
            str(output),
        ],
    )
    localize_features.main()
    rows, metadata = localize_common.load_feature_bundle(output)
    np.testing.assert_array_equal(rows[0]["true_coord"], [2.0, 3.0, 4.0])
    assert any(np.array_equal(rows[0]["peak_coord"], point) for point in coords)
    assert rows[0]["scalp_feature"].shape == (2,)
    assert metadata["metadata"]["coordinate_space"] == "surface"


def test_primary_rejects_raw():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "preprocess/spes_primary.py"),
            "--preserve-amplitude",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "reported primary inputs use centering" in result.stderr


def test_cluster_weighting():
    from evaluation.spes_intervals import summarize

    rows = []
    for participant, count, covered in [("a", 3, 0), ("b", 1, 1)]:
        for run in range(count):
            for model, error in [("ridge", 2.0), ("coordinate_mean", 5.0)]:
                rows.append(
                    dict(
                        participant_id=participant,
                        run_id=str(run),
                        model=model,
                        error_mm=error,
                        radius_mm=4.0,
                        covered=covered,
                    )
                )
    summary = summarize(rows, n_bootstrap=1000)
    assert summary["coverage"] == 0.25
    assert summary["coverage_ci"] == [0.0, 1.0]
    assert summary["mean_error_reduction_mm"] == 3.0
    assert summary["error_reduction_ci_mm"] == [3.0, 3.0]
