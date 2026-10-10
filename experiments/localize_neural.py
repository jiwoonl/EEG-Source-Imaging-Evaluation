#!/usr/bin/env python3
"""Evaluate scalp neural baselines with matched outer participant splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from localize_common import (
    AXIS_INDEX,
    binary_brier,
    binary_ece,
    binary_nll,
    pooled_auroc,
    stable_sigmoid,
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_tensor(
    row: dict[str, str],
    dataset_root: Path,
    *,
    post_start_s: float,
    post_end_s: float,
    n_time_bins: int,
) -> np.ndarray:
    epochs = np.load(dataset_root / row["eeg_epochs_npy"], mmap_mode="r")
    evoked = np.asarray(epochs.mean(axis=0), dtype=np.float32)
    channels = read_tsv(dataset_root / row["eeg_channels_tsv"])
    good = np.asarray(
        [channel.get("status", "good") == "good" for channel in channels],
        dtype=bool,
    )
    fs = float(row["sampling_frequency_hz"])
    zero_time = float(row["zero_time_s"])
    time_axis = np.arange(evoked.shape[1], dtype=np.float64) / fs - zero_time
    indices = np.flatnonzero((time_axis >= post_start_s) & (time_axis <= post_end_s))
    if indices.size < n_time_bins:
        raise ValueError(
            f"Too few samples for {row['participant_id']} {row['run_id']}."
        )
    edges = np.linspace(0, indices.size, n_time_bins + 1, dtype=np.int64)
    output = np.zeros((evoked.shape[0], n_time_bins), dtype=np.float32)
    for bin_index in range(n_time_bins):
        selected = indices[edges[bin_index] : edges[bin_index + 1]]
        output[:, bin_index] = evoked[:, selected].mean(axis=1)
    output[~good] = 0.0
    output -= output.mean(axis=0, keepdims=True)
    norm = float(np.linalg.norm(output))
    if norm > 0:
        output /= norm
    return output


def load_or_build_cache(
    manifest_rows: list[dict[str, str]],
    dataset_root: Path,
    cache_path: Path,
    *,
    post_start_s: float,
    post_end_s: float,
    n_time_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists():
        cache = np.load(cache_path)
        return (
            np.asarray(cache["x"], dtype=np.float32),
            np.asarray(cache["subjects"]),
            np.asarray(cache["run_ids"]),
            np.asarray(cache["true_coord"], dtype=np.float64),
        )
    tensors = [
        build_run_tensor(
            row,
            dataset_root,
            post_start_s=post_start_s,
            post_end_s=post_end_s,
            n_time_bins=n_time_bins,
        )
        for row in manifest_rows
    ]
    x = np.stack(tensors, axis=0)
    subjects = np.asarray([row["participant_id"] for row in manifest_rows])
    run_ids = np.asarray([row["run_id"] for row in manifest_rows])
    true_coord = np.asarray(
        [
            [
                float(row["surface_mid_x"]),
                float(row["surface_mid_y"]),
                float(row["surface_mid_z"]),
            ]
            for row in manifest_rows
        ],
        dtype=np.float64,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x=x,
        subjects=subjects,
        run_ids=run_ids,
        true_coord=true_coord,
    )
    return x, subjects, run_ids, true_coord


class EEGNetStyleClassifier(nn.Module):
    def __init__(self, n_channels: int, n_time: int, dropout: float) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(8),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                16, 16, kernel_size=(1, 5), padding=(0, 2), groups=16, bias=False
            ),
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(x[:, None])
        h = self.spatial(h)
        h = self.separable(h).flatten(1)
        return self.classifier(h).squeeze(-1)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size=5, padding=padding, dilation=dilation
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TCNClassifier(nn.Module):
    def __init__(self, n_channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Conv1d(n_channels, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            ResidualTCNBlock(hidden, 1, dropout),
            ResidualTCNBlock(hidden, 2, dropout),
            ResidualTCNBlock(hidden, 4, dropout),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.input(x)).mean(dim=-1)
        return self.head(h).squeeze(-1)


class CTNetLikeClassifier(nn.Module):
    """Compact convolutional-transformer adapted to the short evoked window."""

    def __init__(
        self, n_channels: int, n_time: int, hidden: int, dropout: float
    ) -> None:
        super().__init__()
        del n_time
        model_dim = max(32, hidden)
        n_heads = 4 if model_dim % 4 == 0 else 2
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(8),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(8, model_dim, kernel_size=(n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(model_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Dropout(dropout), nn.Linear(model_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.spatial(self.temporal(x[:, None])).squeeze(2).transpose(1, 2)
        h = self.transformer(h).mean(dim=1)
        return self.head(h).squeeze(-1)


def make_model(
    model_name: str, n_channels: int, n_time: int, hidden: int, dropout: float
) -> nn.Module:
    if model_name == "eegnet_style":
        return EEGNetStyleClassifier(n_channels, n_time, dropout)
    if model_name == "tcn":
        return TCNClassifier(n_channels, hidden, dropout)
    if model_name == "ctnet_like":
        return CTNetLikeClassifier(n_channels, n_time, hidden, dropout)
    raise ValueError(model_name)


def predict_logits(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return (
            model(torch.from_numpy(x).to(device=device, dtype=torch.float32))
            .cpu()
            .numpy()
            .astype(np.float64)
        )


def train_model(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, int, float, float]:
    set_seed(seed)
    model = make_model(
        model_name,
        train_x.shape[1],
        train_x.shape[2],
        args.hidden,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    train_tensor = torch.from_numpy(train_x).to(device=device, dtype=torch.float32)
    train_labels = torch.from_numpy(train_y.astype(np.float32)).to(device)
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_epoch = 0
    best_val = float("inf")
    patience = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = torch.randperm(train_x.shape[0], generator=generator)
        for start in range(0, train_x.shape[0], args.batch_size):
            indices = order[start : start + args.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_tensor[indices])
            loss = criterion(logits, train_labels[indices])
            loss.backward()
            optimizer.step()
        val_logits = predict_logits(model, val_x, device)
        val_nll = binary_nll(val_y, stable_sigmoid(val_logits))
        if val_nll < best_val - 1e-5:
            best_val = val_nll
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    model.load_state_dict(best_state)
    val_logits = predict_logits(model, val_x, device)
    temperature_rows = [
        (
            temperature,
            binary_nll(val_y, stable_sigmoid(val_logits / temperature)),
        )
        for temperature in args.temperature_grid
    ]
    temperature, calibrated_val_nll = min(
        temperature_rows, key=lambda item: (item[1], item[0])
    )
    return model, best_epoch, calibrated_val_nll, float(temperature)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "data/derived/localize_mi_stimulation_manifest.tsv",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "data/raw/localize-mi",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=repo_root / "data/cache/localize_mi/run_waveforms.npz",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results/localize_neural",
    )
    parser.add_argument("--models", nargs="+", default=["eegnet_style", "tcn"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--label-axis", choices=("x", "y", "z"), default="y")
    parser.add_argument("--post-start-s", type=float, default=0.0)
    parser.add_argument("--post-end-s", type=float, default=0.05)
    parser.add_argument("--n-time-bins", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--temperature-grid",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_rows = read_tsv(args.manifest.resolve())
    x, subjects, run_ids, true_coord = load_or_build_cache(
        manifest_rows,
        args.dataset_root.resolve(),
        args.cache.resolve(),
        post_start_s=args.post_start_s,
        post_end_s=args.post_end_s,
        n_time_bins=args.n_time_bins,
    )
    unique_subjects = sorted(set(subjects.tolist()))
    axis_index = AXIS_INDEX[args.label_axis]
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    prediction_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []

    for outer_index, holdout_subject in enumerate(unique_subjects):
        test_mask = subjects == holdout_subject
        outer_train_mask = ~test_mask
        outer_train_subjects = [
            subject for subject in unique_subjects if subject != holdout_subject
        ]
        validation_subject = outer_train_subjects[
            outer_index % len(outer_train_subjects)
        ]
        validation_mask = subjects == validation_subject
        fit_mask = outer_train_mask & ~validation_mask
        threshold = float(np.median(true_coord[outer_train_mask, axis_index]))
        labels = (true_coord[:, axis_index] >= threshold).astype(np.int64)

        mean = x[fit_mask].mean(axis=0, keepdims=True)
        std = x[fit_mask].std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        fit_x = ((x[fit_mask] - mean) / std).astype(np.float32)
        validation_x = ((x[validation_mask] - mean) / std).astype(np.float32)
        test_x = ((x[test_mask] - mean) / std).astype(np.float32)
        for model_name in args.models:
            seed_logits: list[np.ndarray] = []
            for seed in args.seeds:
                model, best_epoch, val_nll, temperature = train_model(
                    model_name,
                    fit_x,
                    labels[fit_mask],
                    validation_x,
                    labels[validation_mask],
                    args=args,
                    device=device,
                    seed=seed,
                )
                test_logits = predict_logits(model, test_x, device)
                seed_logits.append(test_logits / temperature)
                training_rows.append(
                    {
                        "holdout_subject": holdout_subject,
                        "validation_subject": validation_subject,
                        "model": model_name,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "validation_nll": val_nll,
                        "temperature": temperature,
                        "n_fit_runs": int(fit_mask.sum()),
                        "n_validation_runs": int(validation_mask.sum()),
                        "n_test_runs": int(test_mask.sum()),
                    }
                )
            scores = stable_sigmoid(np.mean(np.stack(seed_logits, axis=0), axis=0))
            for run_id, label, score in zip(
                run_ids[test_mask], labels[test_mask], scores
            ):
                prediction_rows.append(
                    {
                        "participant_id": holdout_subject,
                        "run_id": str(run_id),
                        "model": model_name,
                        "label": int(label),
                        "score": float(score),
                        "label_threshold": threshold,
                        "validation_subject": validation_subject,
                        "n_seeds": len(args.seeds),
                    }
                )
            print(
                f"{holdout_subject} {model_name}: nll={binary_nll(labels[test_mask], scores):.3f}",
                flush=True,
            )

    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for model_name in args.models:
        selected = [row for row in prediction_rows if row["model"] == model_name]
        pooled_y = np.asarray([int(row["label"]) for row in selected], dtype=np.int64)
        pooled_s = np.asarray(
            [float(row["score"]) for row in selected], dtype=np.float64
        )
        metric_rows.append(
            {
                "model": model_name,
                "n_runs": len(selected),
                "n_subjects": len(unique_subjects),
                "auroc": pooled_auroc(pooled_y, pooled_s),
                "nll": binary_nll(pooled_y, pooled_s),
                "brier": binary_brier(pooled_y, pooled_s),
                "ece": binary_ece(pooled_y, pooled_s),
            }
        )
        for subject in unique_subjects:
            rows = [row for row in selected if row["participant_id"] == subject]
            labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
            scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
            fold_rows.append(
                {
                    "holdout_subject": subject,
                    "model": model_name,
                    "n_runs": len(rows),
                    "positive_prevalence": float(np.mean(labels)),
                    "auroc": (
                        pooled_auroc(labels, scores)
                        if np.unique(labels).size == 2
                        else ""
                    ),
                    "nll": binary_nll(labels, scores),
                    "brier": binary_brier(labels, scores),
                    "ece": binary_ece(labels, scores),
                }
            )

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(results_dir / "prediction_rows.tsv", prediction_rows)
    write_tsv(results_dir / "training_details.tsv", training_rows)
    write_tsv(results_dir / "pooled_metrics.tsv", metric_rows)
    write_tsv(results_dir / "fold_metrics.tsv", fold_rows)
    config = vars(args) | {
        "manifest": str(args.manifest.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "cache": str(args.cache.resolve()),
        "results_dir": str(results_dir),
        "device_resolved": str(device),
        "n_runs": int(x.shape[0]),
        "n_channels": int(x.shape[1]),
        "n_time_bins": int(x.shape[2]),
    }
    (results_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Localize-MI scalp neural baselines",
        "",
        f"- Input: run-level evoked scalp waveform, `{x.shape[1]}` channels x `{x.shape[2]}` time bins.",
        f"- Outer split: LOSO over `{len(unique_subjects)}` subjects.",
        "- Early stopping and temperature selection use one training-fold validation subject.",
        f"- Ensemble seeds: `{', '.join(map(str, args.seeds))}`.",
        "",
    ]
    for row in metric_rows:
        lines.append(
            f"- {row['model']}: AUROC `{float(row['auroc']):.3f}`, "
            f"NLL `{float(row['nll']):.3f}`, Brier `{float(row['brier']):.3f}`, "
            f"ECE `{float(row['ece']):.3f}`."
        )
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metric_rows, indent=2))


if __name__ == "__main__":
    main()
