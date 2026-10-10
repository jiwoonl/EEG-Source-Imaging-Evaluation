#!/usr/bin/env python3
"""Evaluate EEGNet-style, TCN, masked-pretraining and FNO-style proxy encoders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def group_rows_by_cache(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["participant_id"], row["phase"])
        grouped.setdefault(key, []).append(row)
    for key in grouped:
        grouped[key].sort(key=lambda item: int(item["trial_index"]))
    return grouped


def load_examples(
    rows: list[dict[str, str]], cache_root: Path
) -> tuple[np.ndarray, np.ndarray]:
    grouped = group_rows_by_cache(rows)
    x_batches: list[np.ndarray] = []
    y_batches: list[np.ndarray] = []
    for subject_phase in sorted(grouped):
        block_rows = grouped[subject_phase]
        first = block_rows[0]
        scalp = np.load(cache_root / first["scalp_npy"], mmap_mode="r")
        source = np.load(cache_root / first["source_npy"], mmap_mode="r")
        indices = [int(row["trial_index"]) for row in block_rows]
        x_batches.append(np.asarray(scalp[indices], dtype=np.float32))
        y_batches.append(np.asarray(source[indices], dtype=np.float32))
    return np.concatenate(x_batches, axis=0), np.concatenate(y_batches, axis=0)


def maybe_subsample(
    x: np.ndarray, y: np.ndarray, limit: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if limit <= 0 or x.shape[0] <= limit:
        return x, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(x.shape[0]), size=limit, replace=False))
    return x[idx], y[idx]


def zscore_3d(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1.0
    return tuple(
        ((array - mean) / std).astype(np.float32) for array in (train, *others)
    ) + (mean.astype(np.float32), std.astype(np.float32))


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = y_true.reshape(-1).astype(np.float64)
    b = y_pred.reshape(-1).astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = y_true.reshape(-1).astype(np.float64)
    pred = y_pred.reshape(-1).astype(np.float64)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = y_true.reshape(-1).astype(np.float64)
    pred = y_pred.reshape(-1).astype(np.float64)
    denom = float(true.std())
    return 0.0 if denom == 0 else float(np.sqrt(np.mean((true - pred) ** 2)) / denom)


def sourcewise_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, source_labels: list[str]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_index, label in enumerate(source_labels):
        rows.append(
            {
                "source_label": label,
                "pearson_r": pearson_corr(
                    y_true[:, source_index], y_pred[:, source_index]
                ),
                "r2": r2_score(y_true[:, source_index], y_pred[:, source_index]),
                "nrmse": nrmse(y_true[:, source_index], y_pred[:, source_index]),
            }
        )
    return rows


def choose_validation(subjects: list[str], holdout_subject: str) -> str:
    holdout_index = subjects.index(holdout_subject)
    for offset in range(1, len(subjects)):
        candidate = subjects[(holdout_index + offset) % len(subjects)]
        if candidate != holdout_subject:
            return candidate
    raise RuntimeError("Could not select validation subject.")


def split_train_rows(
    train_rows: list[dict[str, str]],
    subjects: list[str],
    holdout_subject: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    validation_subject = choose_validation(subjects, holdout_subject)
    model_train_rows = [
        row for row in train_rows if row["participant_id"] != validation_subject
    ]
    validation_rows = [
        row for row in train_rows if row["participant_id"] == validation_subject
    ]
    if model_train_rows and validation_rows:
        return model_train_rows, validation_rows, validation_subject
    cutoff = max(1, int(round(len(train_rows) * 0.1)))
    return train_rows[:-cutoff], train_rows[-cutoff:], "trial_slice_fallback"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class EEGNetSourceNet(nn.Module):
    def __init__(
        self, n_scalp: int, n_source: int, hidden: int, dropout: float
    ) -> None:
        super().__init__()
        f1 = max(4, hidden // 8)
        depth = 2
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth, kernel_size=(n_scalp, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Conv1d(f1 * depth, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, n_source, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.temporal(x.unsqueeze(1)).squeeze(2)
        return self.head(features)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TCNSourceNet(nn.Module):
    def __init__(
        self, n_scalp: int, n_source: int, hidden: int, dropout: float
    ) -> None:
        super().__init__()
        self.input = nn.Conv1d(n_scalp, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            ResidualTCNBlock(hidden, 1, dropout),
            ResidualTCNBlock(hidden, 2, dropout),
            ResidualTCNBlock(hidden, 4, dropout),
            ResidualTCNBlock(hidden, 8, dropout),
        )
        self.output = nn.Conv1d(hidden, n_source, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(x)))


class SpectralConv1d(nn.Module):
    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / max(1, channels)
        self.weight = nn.Parameter(
            scale * torch.randn(channels, channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros_like(x_ft)
        modes = min(self.modes, x_ft.shape[-1])
        out_ft[..., :modes] = torch.einsum(
            "bcm,com->bom", x_ft[..., :modes], self.weight[..., :modes]
        )
        return torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)


class FNO1dSourceNet(nn.Module):
    def __init__(
        self, n_scalp: int, n_source: int, hidden: int, modes: int, dropout: float
    ) -> None:
        super().__init__()
        self.lift = nn.Conv1d(n_scalp, hidden, kernel_size=1)
        self.spec1 = SpectralConv1d(hidden, modes)
        self.mix1 = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.spec2 = SpectralConv1d(hidden, modes)
        self.mix2 = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(hidden, n_source, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        h = torch.nn.functional.gelu(self.spec1(h) + self.mix1(h))
        h = self.drop(h)
        h = torch.nn.functional.gelu(self.spec2(h) + self.mix2(h))
        return self.proj(h)


class SSLScalpAutoencoder(nn.Module):
    def __init__(self, n_scalp: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_scalp, hidden, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=9, padding=4),
            nn.GELU(),
        )
        self.decoder = nn.Conv1d(hidden, n_scalp, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class SSLSourceNet(nn.Module):
    def __init__(self, encoder: nn.Module, n_source: int, hidden: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Conv1d(hidden, n_source, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def make_model(
    model_name: str,
    n_scalp: int,
    n_source: int,
    hidden: int,
    modes: int,
    dropout: float,
) -> nn.Module:
    if model_name == "eegnet":
        return EEGNetSourceNet(n_scalp, n_source, hidden, dropout)
    if model_name == "tcn":
        return TCNSourceNet(n_scalp, n_source, hidden, dropout)
    if model_name == "fno":
        return FNO1dSourceNet(n_scalp, n_source, hidden, modes, dropout)
    raise ValueError(model_name)


def evaluate_loss(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.MSELoss()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = start + batch_size
            batch_x = torch.from_numpy(x[start:end]).to(device)
            batch_y = torch.from_numpy(y[start:end]).to(device)
            losses.append(float(criterion(model(batch_x), batch_y).item()))
    return float(np.mean(losses)) if losses else 0.0


def train_source_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, int, float, list[dict[str, object]]]:
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_epoch = 0
    best_val = float("inf")
    patience = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_loss = evaluate_loss(model, x_val, y_val, args.batch_size, device)
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
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
    return model, best_epoch, best_val, history


def pretrain_ssl(
    model: SSLScalpAutoencoder,
    x_train: np.ndarray,
    x_val: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> SSLScalpAutoencoder:
    dataset = TensorDataset(torch.from_numpy(x_train))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_val = float("inf")
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    for _ in range(max(1, args.ssl_pretrain_epochs)):
        model.train()
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            mask = (
                torch.rand(batch_x.shape, generator=rng, device=device)
                > args.ssl_mask_fraction
            ).float()
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch_x * mask)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
        val_loss = evaluate_loss(model, x_val, x_val, args.batch_size, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    return model


def predict(
    model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = start + batch_size
            batches.append(
                model(torch.from_numpy(x[start:end]).to(device))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return np.concatenate(batches, axis=0)


def run_fold_model(
    model_name: str,
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    x_val_raw: np.ndarray,
    y_val_raw: np.ndarray,
    x_test_raw: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, int, float, list[dict[str, object]]]:
    x_train, x_val, x_test, x_mean, x_std = zscore_3d(
        x_train_raw, x_val_raw, x_test_raw
    )
    y_train, y_val, _, y_mean, y_std = zscore_3d(y_train_raw, y_val_raw, y_train_raw)
    n_scalp = x_train.shape[1]
    n_source = y_train.shape[1]
    if model_name == "ssl_masked":
        ssl = SSLScalpAutoencoder(n_scalp, args.hidden_channels, args.dropout)
        ssl = pretrain_ssl(ssl, x_train, x_val, args=args, device=device, seed=seed)
        source_model = SSLSourceNet(ssl.encoder, n_source, args.hidden_channels)
        model, best_epoch, best_val, history = train_source_model(
            source_model,
            x_train,
            y_train,
            x_val,
            y_val,
            args=args,
            device=device,
            seed=seed,
        )
    else:
        model = make_model(
            model_name,
            n_scalp,
            n_source,
            args.hidden_channels,
            args.fno_modes,
            args.dropout,
        )
        model, best_epoch, best_val, history = train_source_model(
            model, x_train, y_train, x_val, y_val, args=args, device=device, seed=seed
        )
    pred_z = predict(model, x_test, args.batch_size, device)
    pred = (pred_z * y_std) + y_mean
    return pred.astype(np.float32), best_epoch, best_val, history


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=repo_root
        / "data/derived/ds004752_beamforming_lowfreq_1to12hz/ds004752_beamforming_loso_trial_split.tsv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=repo_root / "data/cache/ds004752/beamforming_common_lowfreq_1to12hz",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results/ds004752_beamforming_modern_baselines",
    )
    parser.add_argument(
        "--models", nargs="+", default=["eegnet", "tcn", "ssl_masked", "fno"]
    )
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--fno-modes", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ssl-pretrain-epochs", type=int, default=3)
    parser.add_argument("--ssl-mask-fraction", type=float, default=0.25)
    parser.add_argument("--max-train-trials-per-fold", type=int, default=0)
    parser.add_argument("--max-val-trials-per-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fold-id", type=str, default=None)
    parser.add_argument("--phase", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    split_manifest = args.split_manifest.resolve()
    cache_root = args.cache_root.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    split_rows = read_tsv(split_manifest)
    derived_dir = split_manifest.parent
    source_labels = [
        line.strip()
        for line in (derived_dir / "ds004752_beamforming_common_source_labels.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    folds = sorted({row["fold_id"] for row in split_rows})
    phases = sorted({row["phase"] for row in split_rows})
    if args.fold_id is not None:
        folds = [fold for fold in folds if fold == args.fold_id]
    if args.phase is not None:
        phases = [phase for phase in phases if phase == args.phase]
    subjects = sorted({row["participant_id"] for row in split_rows})

    fold_metric_rows: list[dict[str, object]] = []
    source_metric_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, dict[str, float]]] = {}

    for model_name in args.models:
        summary[model_name] = {}
        for phase in phases:
            phase_rows: list[dict[str, object]] = []
            for fold_id in folds:
                fold_phase_rows = [
                    row
                    for row in split_rows
                    if row["fold_id"] == fold_id and row["phase"] == phase
                ]
                train_rows = [row for row in fold_phase_rows if row["split"] == "train"]
                test_rows = [row for row in fold_phase_rows if row["split"] == "test"]
                holdout_subject = fold_phase_rows[0]["holdout_subject"]
                model_train_rows, validation_rows, validation_subject = (
                    split_train_rows(train_rows, subjects, holdout_subject)
                )
                x_train, y_train = load_examples(model_train_rows, cache_root)
                x_val, y_val = load_examples(validation_rows, cache_root)
                x_test, y_test = load_examples(test_rows, cache_root)
                x_train, y_train = maybe_subsample(
                    x_train,
                    y_train,
                    args.max_train_trials_per_fold,
                    args.seed + len(fold_id) + len(model_name),
                )
                x_val, y_val = maybe_subsample(
                    x_val,
                    y_val,
                    args.max_val_trials_per_fold,
                    args.seed + 3 * len(fold_id) + len(model_name),
                )
                y_pred, best_epoch, best_val, fold_history = run_fold_model(
                    model_name,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    x_test,
                    args,
                    device,
                    args.seed,
                )
                metrics = {
                    "model": model_name,
                    "fold_id": fold_id,
                    "holdout_subject": holdout_subject,
                    "validation_subject": validation_subject,
                    "phase": phase,
                    "device": str(device),
                    "seed": args.seed,
                    "hidden_channels": args.hidden_channels,
                    "batch_size": args.batch_size,
                    "max_epochs": args.max_epochs,
                    "patience": args.patience,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                    "n_train_trials": x_train.shape[0],
                    "n_val_trials": x_val.shape[0],
                    "n_test_trials": x_test.shape[0],
                    "pearson_r": pearson_corr(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred),
                    "nrmse": nrmse(y_test, y_pred),
                }
                fold_metric_rows.append(metrics)
                phase_rows.append(metrics)
                for history in fold_history:
                    history_rows.append(
                        {
                            "model": model_name,
                            "fold_id": fold_id,
                            "holdout_subject": holdout_subject,
                            "validation_subject": validation_subject,
                            "phase": phase,
                            **history,
                        }
                    )
                for row in sourcewise_metrics(y_test, y_pred, source_labels):
                    source_metric_rows.append(
                        {
                            "model": model_name,
                            "fold_id": fold_id,
                            "holdout_subject": holdout_subject,
                            "validation_subject": validation_subject,
                            "phase": phase,
                            **row,
                        }
                    )
                print(
                    f"{model_name} {phase} {fold_id}: r={metrics['pearson_r']:.3f} "
                    f"r2={metrics['r2']:.3f} best_epoch={best_epoch}",
                    flush=True,
                )
            summary[model_name][phase] = {
                "mean_pearson_r": float(
                    np.mean([row["pearson_r"] for row in phase_rows])
                ),
                "std_pearson_r": float(
                    np.std([row["pearson_r"] for row in phase_rows])
                ),
                "mean_r2": float(np.mean([row["r2"] for row in phase_rows])),
                "std_r2": float(np.std([row["r2"] for row in phase_rows])),
                "mean_nrmse": float(np.mean([row["nrmse"] for row in phase_rows])),
                "std_nrmse": float(np.std([row["nrmse"] for row in phase_rows])),
                "n_folds": len(phase_rows),
            }

    source_summary_rows: list[dict[str, object]] = []
    for model_name in args.models:
        for phase in phases:
            for source_label in source_labels:
                rows = [
                    row
                    for row in source_metric_rows
                    if row["model"] == model_name
                    and row["phase"] == phase
                    and row["source_label"] == source_label
                ]
                source_summary_rows.append(
                    {
                        "model": model_name,
                        "phase": phase,
                        "source_label": source_label,
                        "mean_pearson_r": float(
                            np.mean([row["pearson_r"] for row in rows])
                        ),
                        "std_pearson_r": float(
                            np.std([row["pearson_r"] for row in rows])
                        ),
                        "mean_r2": float(np.mean([row["r2"] for row in rows])),
                        "std_r2": float(np.std([row["r2"] for row in rows])),
                        "mean_nrmse": float(np.mean([row["nrmse"] for row in rows])),
                        "std_nrmse": float(np.std([row["nrmse"] for row in rows])),
                    }
                )

    compact_rows: list[dict[str, object]] = []
    for model_name, phases_dict in summary.items():
        for phase, metrics in phases_dict.items():
            compact_rows.append({"baseline": model_name, "phase": phase, **metrics})

    write_tsv(
        results_dir / "fold_metrics.tsv",
        list(fold_metric_rows[0].keys()),
        fold_metric_rows,
    )
    write_tsv(
        results_dir / "source_metrics.tsv",
        list(source_metric_rows[0].keys()),
        source_metric_rows,
    )
    write_tsv(
        results_dir / "source_summary.tsv",
        list(source_summary_rows[0].keys()),
        source_summary_rows,
    )
    write_tsv(
        results_dir / "training_history.tsv", list(history_rows[0].keys()), history_rows
    )
    write_tsv(
        results_dir / "modern_baseline_summary.tsv",
        list(compact_rows[0].keys()),
        compact_rows,
    )
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (results_dir / "config.json").write_text(
        json.dumps(
            vars(args) | {"device_resolved": str(device)}, indent=2, default=str
        ),
        encoding="utf-8",
    )
    (results_dir / "summary.md").write_text(
        "# ds004752 encoder baselines\n\n"
        f"- models: `{', '.join(args.models)}`\n"
        f"- folds per phase: `{len(folds)}`\n"
        f"- phases: `{', '.join(phases)}`\n"
        "- use `modern_baseline_summary.tsv` for the encoder comparison\n",
        encoding="utf-8",
    )
    print(results_dir / "modern_baseline_summary.tsv")


if __name__ == "__main__":
    main()
