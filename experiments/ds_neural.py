#!/usr/bin/env python3
"""Evaluate participant-specific empirical operators in the neural proxy ablation."""

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


def zscore_train(
    train: np.ndarray, other: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (other - mean) / std, mean, std


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = y_true.reshape(-1).astype(np.float64)
    b = y_pred.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = y_true.reshape(-1).astype(np.float64)
    pred = y_pred.reshape(-1).astype(np.float64)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = y_true.reshape(-1).astype(np.float64)
    pred = y_pred.reshape(-1).astype(np.float64)
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    denom = float(true.std())
    return 0.0 if denom == 0 else rmse / denom


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = group_rows_by_cache(rows)
    x_batches: list[np.ndarray] = []
    y_batches: list[np.ndarray] = []
    subject_batches: list[np.ndarray] = []
    for subject_phase in sorted(grouped):
        block_rows = grouped[subject_phase]
        first = block_rows[0]
        scalp = np.load(cache_root / first["scalp_npy"], mmap_mode="r")
        source = np.load(cache_root / first["source_npy"], mmap_mode="r")
        indices = [int(row["trial_index"]) for row in block_rows]
        x_batches.append(
            np.asarray(scalp[indices], dtype=np.float32).reshape(len(indices), -1)
        )
        y_batches.append(
            np.asarray(source[indices], dtype=np.float32).reshape(len(indices), -1)
        )
        subject_batches.append(np.asarray([first["participant_id"]] * len(indices)))
    return (
        np.concatenate(x_batches, axis=0),
        np.concatenate(y_batches, axis=0),
        np.concatenate(subject_batches, axis=0),
    )


def sourcewise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    source_labels: list[str],
    n_timepoints: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_sources = len(source_labels)
    y_true_3d = y_true.reshape(y_true.shape[0], n_sources, n_timepoints)
    y_pred_3d = y_pred.reshape(y_pred.shape[0], n_sources, n_timepoints)
    for source_index, label in enumerate(source_labels):
        true = y_true_3d[:, source_index, :]
        pred = y_pred_3d[:, source_index, :]
        rows.append(
            {
                "source_label": label,
                "pearson_r": pearson_corr(true, pred),
                "r2": r2_score(true, pred),
                "nrmse": nrmse(true, pred),
            }
        )
    return rows


def choose_validation(subjects: list[str], holdout_subject: str) -> str:
    holdout_index = subjects.index(holdout_subject)
    for offset in range(1, len(subjects)):
        candidate = subjects[(holdout_index + offset) % len(subjects)]
        if candidate != holdout_subject:
            return candidate
    raise RuntimeError("Could not select a validation subject.")


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


def fit_forward_matrix(
    x_train_z: np.ndarray,
    y_train_z: np.ndarray,
    n_scalp_channels: int,
    n_source_channels: int,
    n_timepoints: int,
    alpha: float,
) -> np.ndarray:
    y_flat = (
        y_train_z.reshape(-1, n_source_channels, n_timepoints)
        .transpose(0, 2, 1)
        .reshape(-1, n_source_channels)
    )
    x_flat = (
        x_train_z.reshape(-1, n_scalp_channels, n_timepoints)
        .transpose(0, 2, 1)
        .reshape(-1, n_scalp_channels)
    )
    gram = y_flat.T @ y_flat
    gram.flat[:: gram.shape[0] + 1] += alpha
    weights = np.linalg.solve(gram, y_flat.T @ x_flat)
    return weights.T.astype(np.float32)


def fit_operator_bank(
    x_train_z: np.ndarray,
    y_train_z: np.ndarray,
    subject_ids: np.ndarray,
    n_scalp_channels: int,
    n_source_channels: int,
    n_timepoints: int,
    alpha: float,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    operator_subjects = sorted({str(subject_id) for subject_id in subject_ids.tolist()})
    operators: list[np.ndarray] = []
    subject_counts: dict[str, int] = {}
    for subject_id in operator_subjects:
        mask = subject_ids == subject_id
        subject_counts[subject_id] = int(mask.sum())
        operators.append(
            fit_forward_matrix(
                x_train_z=x_train_z[mask],
                y_train_z=y_train_z[mask],
                n_scalp_channels=n_scalp_channels,
                n_source_channels=n_source_channels,
                n_timepoints=n_timepoints,
                alpha=alpha,
            )
        )
    return (
        np.stack(operators, axis=0).astype(np.float32),
        operator_subjects,
        subject_counts,
    )


def mean_gate_entropy(weights: np.ndarray) -> float:
    clipped = np.clip(weights.astype(np.float64), 1e-8, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    return float(np.mean(entropy))


class SubjectAwareForwardNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        n_scalp_channels: int,
        n_source_channels: int,
        n_timepoints: int,
        forward_bank: np.ndarray,
        gate_temperature: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.source_head = nn.Linear(hidden_dim, output_dim)
        self.gate_head = nn.Linear(hidden_dim, int(forward_bank.shape[0]))
        self.n_scalp_channels = n_scalp_channels
        self.n_source_channels = n_source_channels
        self.n_timepoints = n_timepoints
        self.gate_temperature = gate_temperature
        self.register_buffer(
            "forward_bank", torch.from_numpy(forward_bank.astype(np.float32))
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        source = self.source_head(latent)
        gate_logits = self.gate_head(latent) / self.gate_temperature
        gate_weights = torch.softmax(gate_logits, dim=-1)
        source_3d = source.view(-1, self.n_source_channels, self.n_timepoints)
        scalp_bank = torch.einsum("kcs,bst->bkct", self.forward_bank, source_3d)
        scalp_3d = torch.einsum("bk,bkct->bct", gate_weights, scalp_bank)
        scalp = scalp_3d.reshape(x.shape[0], self.n_scalp_channels * self.n_timepoints)
        return source, scalp, gate_logits, gate_weights


def evaluate_losses(
    model: nn.Module,
    x_array: np.ndarray,
    y_array: np.ndarray,
    batch_size: int,
    forward_weight: float,
    device: torch.device,
) -> tuple[float, float, float]:
    source_criterion = nn.MSELoss()
    forward_criterion = nn.MSELoss()
    total_losses: list[float] = []
    source_losses: list[float] = []
    forward_losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x_array.shape[0], batch_size):
            end = start + batch_size
            batch_x = torch.from_numpy(x_array[start:end]).to(
                device=device, dtype=torch.float32
            )
            batch_y = torch.from_numpy(y_array[start:end]).to(
                device=device, dtype=torch.float32
            )
            batch_source_pred, batch_scalp_pred, _, _ = model(batch_x)
            source_loss = source_criterion(batch_source_pred, batch_y)
            forward_loss = forward_criterion(batch_scalp_pred, batch_x)
            total_loss = source_loss + (forward_weight * forward_loss)
            total_losses.append(float(total_loss.item()))
            source_losses.append(float(source_loss.item()))
            forward_losses.append(float(forward_loss.item()))
    return (
        float(np.mean(total_losses)),
        float(np.mean(source_losses)),
        float(np.mean(forward_losses)),
    )


def predict_arrays(
    model: nn.Module,
    x_array: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_batches: list[np.ndarray] = []
    scalp_batches: list[np.ndarray] = []
    gate_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x_array.shape[0], batch_size):
            end = start + batch_size
            batch_x = torch.from_numpy(x_array[start:end]).to(
                device=device, dtype=torch.float32
            )
            batch_source_pred, batch_scalp_pred, _, batch_gate_weights = model(batch_x)
            source_batches.append(batch_source_pred.cpu().numpy().astype(np.float32))
            scalp_batches.append(batch_scalp_pred.cpu().numpy().astype(np.float32))
            gate_batches.append(batch_gate_weights.cpu().numpy().astype(np.float32))
    return (
        np.concatenate(source_batches, axis=0),
        np.concatenate(scalp_batches, axis=0),
        np.concatenate(gate_batches, axis=0),
    )


def train_fold_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_subjects: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_scalp_channels: int,
    n_source_channels: int,
    n_timepoints: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    nn.Module, dict[str, object], list[dict[str, object]], int, float, float, float
]:
    x_train_z, x_val_z, x_mean, x_std = zscore_train(x_train, x_val)
    y_train_z, y_val_z, y_mean, y_std = zscore_train(y_train, y_val)
    forward_bank, operator_subjects, operator_trial_counts = fit_operator_bank(
        x_train_z=x_train_z,
        y_train_z=y_train_z,
        subject_ids=train_subjects,
        n_scalp_channels=n_scalp_channels,
        n_source_channels=n_source_channels,
        n_timepoints=n_timepoints,
        alpha=args.forward_ridge_alpha,
    )
    operator_index = {
        subject_id: idx for idx, subject_id in enumerate(operator_subjects)
    }
    gate_targets = np.asarray(
        [operator_index[str(subject_id)] for subject_id in train_subjects],
        dtype=np.int64,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(x_train_z.astype(np.float32)),
        torch.from_numpy(y_train_z.astype(np.float32)),
        torch.from_numpy(gate_targets),
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    model = SubjectAwareForwardNet(
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=y_train.shape[1],
        dropout=args.dropout,
        n_scalp_channels=n_scalp_channels,
        n_source_channels=n_source_channels,
        n_timepoints=n_timepoints,
        forward_bank=forward_bank,
        gate_temperature=args.gate_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    source_criterion = nn.MSELoss()
    forward_criterion = nn.MSELoss()
    gate_criterion = nn.CrossEntropyLoss()

    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_source_loss = float("inf")
    best_val_forward_loss = float("inf")
    patience_counter = 0
    history_rows: list[dict[str, object]] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_total_losses: list[float] = []
        train_source_losses: list[float] = []
        train_forward_losses: list[float] = []
        train_gate_losses: list[float] = []
        train_gate_accs: list[float] = []
        train_gate_entropies: list[float] = []
        for batch_x, batch_y, batch_subject_idx in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_subject_idx = batch_subject_idx.to(device)
            optimizer.zero_grad(set_to_none=True)
            (
                batch_source_pred,
                batch_scalp_pred,
                batch_gate_logits,
                batch_gate_weights,
            ) = model(batch_x)
            source_loss = source_criterion(batch_source_pred, batch_y)
            forward_loss = forward_criterion(batch_scalp_pred, batch_x)
            gate_loss = gate_criterion(batch_gate_logits, batch_subject_idx)
            total_loss = (
                source_loss
                + (args.forward_weight * forward_loss)
                + (args.gate_weight * gate_loss)
            )
            total_loss.backward()
            optimizer.step()
            train_total_losses.append(float(total_loss.item()))
            train_source_losses.append(float(source_loss.item()))
            train_forward_losses.append(float(forward_loss.item()))
            train_gate_losses.append(float(gate_loss.item()))
            train_gate_accs.append(
                float(
                    (batch_gate_logits.argmax(dim=-1) == batch_subject_idx)
                    .float()
                    .mean()
                    .item()
                )
            )
            batch_entropy = -(
                batch_gate_weights * torch.log(batch_gate_weights.clamp_min(1e-8))
            ).sum(dim=-1)
            train_gate_entropies.append(float(batch_entropy.mean().item()))

        val_total_loss, val_source_loss, val_forward_loss = evaluate_losses(
            model=model,
            x_array=x_val_z.astype(np.float32),
            y_array=y_val_z.astype(np.float32),
            batch_size=args.batch_size,
            forward_weight=args.forward_weight,
            device=device,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_total_loss": (
                    float(np.mean(train_total_losses)) if train_total_losses else 0.0
                ),
                "train_source_loss": (
                    float(np.mean(train_source_losses)) if train_source_losses else 0.0
                ),
                "train_forward_loss": (
                    float(np.mean(train_forward_losses))
                    if train_forward_losses
                    else 0.0
                ),
                "train_gate_loss": (
                    float(np.mean(train_gate_losses)) if train_gate_losses else 0.0
                ),
                "train_gate_accuracy": (
                    float(np.mean(train_gate_accs)) if train_gate_accs else 0.0
                ),
                "train_gate_entropy": (
                    float(np.mean(train_gate_entropies))
                    if train_gate_entropies
                    else 0.0
                ),
                "val_total_loss": val_total_loss,
                "val_source_loss": val_source_loss,
                "val_forward_loss": val_forward_loss,
            }
        )

        if val_total_loss < best_val_loss - 1e-6:
            best_val_loss = val_total_loss
            best_val_source_loss = val_source_loss
            best_val_forward_loss = val_forward_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    model.load_state_dict(best_state)
    stats = {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
        "forward_bank": forward_bank.astype(np.float32),
        "operator_subjects": operator_subjects,
        "operator_trial_counts": operator_trial_counts,
    }
    return (
        model,
        stats,
        history_rows,
        best_epoch,
        best_val_loss,
        best_val_source_loss,
        best_val_forward_loss,
    )


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=repo_root / "data/derived/ds004752_beamforming_loso_trial_split.tsv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=repo_root / "data/cache/ds004752/beamforming_common",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results/ds004752_beamforming_subject_aware_v1",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--forward-weight", type=float, default=2.0)
    parser.add_argument("--gate-weight", type=float, default=0.2)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--forward-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
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
    scalp_labels = [
        line.strip()
        for line in (derived_dir / "ds004752_beamforming_common_scalp_labels.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    phase_metadata = {
        row["phase"]: row
        for row in read_tsv(derived_dir / "ds004752_beamforming_cache_manifest.tsv")
    }

    folds = sorted({row["fold_id"] for row in split_rows})
    phases = sorted({row["phase"] for row in split_rows})
    if args.fold_id is not None:
        folds = [fold_id for fold_id in folds if fold_id == args.fold_id]
        if not folds:
            raise RuntimeError(f"Fold {args.fold_id} not found in split manifest.")
    if args.phase is not None:
        phases = [phase for phase in phases if phase == args.phase]
        if not phases:
            raise RuntimeError(f"Phase {args.phase} not found in split manifest.")
    subjects = sorted({row["participant_id"] for row in split_rows})

    fold_metric_rows: list[dict[str, object]] = []
    source_metric_rows: list[dict[str, object]] = []
    source_summary_rows: list[dict[str, object]] = []
    operator_usage_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}

    n_scalp_channels = len(scalp_labels)
    n_source_channels = len(source_labels)

    for phase in phases:
        phase_fold_rows: list[dict[str, object]] = []
        n_timepoints = int(phase_metadata[phase]["n_timepoints"])
        for fold_id in folds:
            fold_phase_rows = [
                row
                for row in split_rows
                if row["fold_id"] == fold_id and row["phase"] == phase
            ]
            train_rows = [row for row in fold_phase_rows if row["split"] == "train"]
            test_rows = [row for row in fold_phase_rows if row["split"] == "test"]
            holdout_subject = fold_phase_rows[0]["holdout_subject"]
            model_train_rows, validation_rows, validation_subject = split_train_rows(
                train_rows, subjects, holdout_subject
            )

            x_train, y_train, train_subjects = load_examples(
                model_train_rows, cache_root
            )
            x_val, y_val, _ = load_examples(validation_rows, cache_root)
            x_test, y_test, _ = load_examples(test_rows, cache_root)

            (
                model,
                stats,
                fold_history,
                best_epoch,
                best_val_loss,
                best_val_source_loss,
                best_val_forward_loss,
            ) = train_fold_model(
                x_train=x_train,
                y_train=y_train,
                train_subjects=train_subjects,
                x_val=x_val,
                y_val=y_val,
                n_scalp_channels=n_scalp_channels,
                n_source_channels=n_source_channels,
                n_timepoints=n_timepoints,
                args=args,
                device=device,
            )

            x_test_z = ((x_test - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
            source_pred_z, scalp_pred_z, gate_weights = predict_arrays(
                model, x_test_z, args.batch_size, device
            )
            y_pred = (source_pred_z * stats["y_std"]) + stats["y_mean"]
            x_recon = (scalp_pred_z * stats["x_std"]) + stats["x_mean"]

            metrics = {
                "fold_id": fold_id,
                "holdout_subject": holdout_subject,
                "validation_subject": validation_subject,
                "phase": phase,
                "model": "subject_aware_forward_consistent",
                "device": str(device),
                "seed": args.seed,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "forward_weight": args.forward_weight,
                "gate_weight": args.gate_weight,
                "gate_temperature": args.gate_temperature,
                "forward_ridge_alpha": args.forward_ridge_alpha,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_source_loss": best_val_source_loss,
                "best_val_forward_loss": best_val_forward_loss,
                "n_train_trials": x_train.shape[0],
                "n_val_trials": x_val.shape[0],
                "n_test_trials": x_test.shape[0],
                "n_operator_subjects": len(stats["operator_subjects"]),
                "pearson_r": pearson_corr(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "nrmse": nrmse(y_test, y_pred),
                "scalp_pearson_r": pearson_corr(x_test, x_recon),
                "scalp_r2": r2_score(x_test, x_recon),
                "scalp_nrmse": nrmse(x_test, x_recon),
                "mean_gate_entropy": mean_gate_entropy(gate_weights),
                "mean_gate_top1_weight": float(np.mean(gate_weights.max(axis=1))),
                "forward_bank_fro": float(np.linalg.norm(stats["forward_bank"])),
            }
            fold_metric_rows.append(metrics)
            phase_fold_rows.append(metrics)

            mean_gate_weights = gate_weights.mean(axis=0)
            for subject_index, operator_subject in enumerate(
                stats["operator_subjects"]
            ):
                operator_usage_rows.append(
                    {
                        "fold_id": fold_id,
                        "holdout_subject": holdout_subject,
                        "validation_subject": validation_subject,
                        "phase": phase,
                        "operator_subject": operator_subject,
                        "operator_trial_count": stats["operator_trial_counts"][
                            operator_subject
                        ],
                        "mean_gate_weight": float(mean_gate_weights[subject_index]),
                    }
                )

            for history in fold_history:
                history_rows.append(
                    {
                        "fold_id": fold_id,
                        "holdout_subject": holdout_subject,
                        "validation_subject": validation_subject,
                        "phase": phase,
                        **history,
                    }
                )

            for source_metric in sourcewise_metrics(
                y_test, y_pred, source_labels, n_timepoints
            ):
                source_metric_rows.append(
                    {
                        "fold_id": fold_id,
                        "holdout_subject": holdout_subject,
                        "validation_subject": validation_subject,
                        "phase": phase,
                        "model": "subject_aware_forward_consistent",
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "forward_weight": args.forward_weight,
                        "gate_weight": args.gate_weight,
                        "gate_temperature": args.gate_temperature,
                        "forward_ridge_alpha": args.forward_ridge_alpha,
                        **source_metric,
                    }
                )

        summary[phase] = {
            "mean_pearson_r": float(
                np.mean([row["pearson_r"] for row in phase_fold_rows])
            ),
            "std_pearson_r": float(
                np.std([row["pearson_r"] for row in phase_fold_rows])
            ),
            "mean_r2": float(np.mean([row["r2"] for row in phase_fold_rows])),
            "std_r2": float(np.std([row["r2"] for row in phase_fold_rows])),
            "mean_nrmse": float(np.mean([row["nrmse"] for row in phase_fold_rows])),
            "std_nrmse": float(np.std([row["nrmse"] for row in phase_fold_rows])),
            "mean_scalp_pearson_r": float(
                np.mean([row["scalp_pearson_r"] for row in phase_fold_rows])
            ),
            "std_scalp_pearson_r": float(
                np.std([row["scalp_pearson_r"] for row in phase_fold_rows])
            ),
            "mean_scalp_r2": float(
                np.mean([row["scalp_r2"] for row in phase_fold_rows])
            ),
            "std_scalp_r2": float(np.std([row["scalp_r2"] for row in phase_fold_rows])),
            "mean_scalp_nrmse": float(
                np.mean([row["scalp_nrmse"] for row in phase_fold_rows])
            ),
            "std_scalp_nrmse": float(
                np.std([row["scalp_nrmse"] for row in phase_fold_rows])
            ),
            "mean_gate_entropy": float(
                np.mean([row["mean_gate_entropy"] for row in phase_fold_rows])
            ),
            "mean_gate_top1_weight": float(
                np.mean([row["mean_gate_top1_weight"] for row in phase_fold_rows])
            ),
            "mean_best_epoch": float(
                np.mean([row["best_epoch"] for row in phase_fold_rows])
            ),
            "mean_best_val_loss": float(
                np.mean([row["best_val_loss"] for row in phase_fold_rows])
            ),
            "mean_best_val_source_loss": float(
                np.mean([row["best_val_source_loss"] for row in phase_fold_rows])
            ),
            "mean_best_val_forward_loss": float(
                np.mean([row["best_val_forward_loss"] for row in phase_fold_rows])
            ),
        }

    for phase in phases:
        for source_label in source_labels:
            rows = [
                row
                for row in source_metric_rows
                if row["phase"] == phase and row["source_label"] == source_label
            ]
            source_summary_rows.append(
                {
                    "phase": phase,
                    "source_label": source_label,
                    "mean_pearson_r": float(
                        np.mean([row["pearson_r"] for row in rows])
                    ),
                    "std_pearson_r": float(np.std([row["pearson_r"] for row in rows])),
                    "mean_r2": float(np.mean([row["r2"] for row in rows])),
                    "std_r2": float(np.std([row["r2"] for row in rows])),
                    "mean_nrmse": float(np.mean([row["nrmse"] for row in rows])),
                    "std_nrmse": float(np.std([row["nrmse"] for row in rows])),
                }
            )

    config = {
        "model": "subject_aware_forward_consistent",
        "split_manifest": str(split_manifest),
        "cache_root": str(cache_root),
        "device": str(device),
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "forward_weight": args.forward_weight,
        "gate_weight": args.gate_weight,
        "gate_temperature": args.gate_temperature,
        "forward_ridge_alpha": args.forward_ridge_alpha,
        "subject_aware_forward_operator": "per-fold bank of subject-specific linear source->scalp maps fitted on training subjects only",
        "gating_head": "sample-conditioned softmax over training-subject forward operators",
    }

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
        results_dir / "operator_usage.tsv",
        list(operator_usage_rows[0].keys()),
        operator_usage_rows,
    )
    write_tsv(
        results_dir / "training_history.tsv", list(history_rows[0].keys()), history_rows
    )
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (results_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("Wrote subject-aware forward-consistent baseline results:")
    for filename in [
        "fold_metrics.tsv",
        "source_metrics.tsv",
        "source_summary.tsv",
        "operator_usage.tsv",
        "training_history.tsv",
        "summary.json",
        "config.json",
    ]:
        print(results_dir / filename)


if __name__ == "__main__":
    main()
