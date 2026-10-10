#!/usr/bin/env python3
"""Run a LOSO MLP baseline on cached ds004752 beamforming trials."""

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
        x_batches.append(
            np.asarray(scalp[indices], dtype=np.float32).reshape(len(indices), -1)
        )
        y_batches.append(
            np.asarray(source[indices], dtype=np.float32).reshape(len(indices), -1)
        )
    return np.concatenate(x_batches, axis=0), np.concatenate(y_batches, axis=0)


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


class BeamformingMLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def evaluate_loss(
    model: nn.Module,
    x_array: np.ndarray,
    y_array: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.MSELoss()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, x_array.shape[0], batch_size):
            end = start + batch_size
            batch_x = torch.from_numpy(x_array[start:end]).to(
                device=device, dtype=torch.float32
            )
            batch_y = torch.from_numpy(y_array[start:end]).to(
                device=device, dtype=torch.float32
            )
            batch_pred = model(batch_x)
            losses.append(float(criterion(batch_pred, batch_y).item()))
    return float(np.mean(losses)) if losses else 0.0


def predict_array(
    model: nn.Module,
    x_array: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_array.shape[0], batch_size):
            end = start + batch_size
            batch_x = torch.from_numpy(x_array[start:end]).to(
                device=device, dtype=torch.float32
            )
            batches.append(model(batch_x).cpu().numpy().astype(np.float32))
    return np.concatenate(batches, axis=0)


def train_fold_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, np.ndarray], list[dict[str, object]], int, float]:
    x_train_z, x_val_z, x_mean, x_std = zscore_train(x_train, x_val)
    y_train_z, y_val_z, y_mean, y_std = zscore_train(y_train, y_val)

    train_dataset = TensorDataset(
        torch.from_numpy(x_train_z.astype(np.float32)),
        torch.from_numpy(y_train_z.astype(np.float32)),
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

    model = BeamformingMLP(
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=y_train.shape[1],
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    history_rows: list[dict[str, object]] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            batch_pred = model(batch_x)
            loss = criterion(batch_pred, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_loss = evaluate_loss(
            model,
            x_val_z.astype(np.float32),
            y_val_z.astype(np.float32),
            args.batch_size,
            device,
        )
        history_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
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
    }
    return model, stats, history_rows, best_epoch, best_val_loss


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
        default=repo_root / "results/ds004752_beamforming_mlp_h256",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    history_rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}

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

            x_train, y_train = load_examples(model_train_rows, cache_root)
            x_val, y_val = load_examples(validation_rows, cache_root)
            x_test, y_test = load_examples(test_rows, cache_root)

            model, stats, fold_history, best_epoch, best_val_loss = train_fold_model(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                args=args,
                device=device,
            )

            x_test_z = ((x_test - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
            y_pred_z = predict_array(model, x_test_z, args.batch_size, device)
            y_pred = (y_pred_z * stats["y_std"]) + stats["y_mean"]

            metrics = {
                "fold_id": fold_id,
                "holdout_subject": holdout_subject,
                "validation_subject": validation_subject,
                "phase": phase,
                "model": "mlp",
                "device": str(device),
                "seed": args.seed,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "n_train_trials": x_train.shape[0],
                "n_val_trials": x_val.shape[0],
                "n_test_trials": x_test.shape[0],
                "pearson_r": pearson_corr(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "nrmse": nrmse(y_test, y_pred),
            }
            fold_metric_rows.append(metrics)
            phase_fold_rows.append(metrics)

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
                        "model": "mlp",
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
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
            "mean_best_epoch": float(
                np.mean([row["best_epoch"] for row in phase_fold_rows])
            ),
            "mean_best_val_loss": float(
                np.mean([row["best_val_loss"] for row in phase_fold_rows])
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
        "model": "mlp",
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
        results_dir / "training_history.tsv", list(history_rows[0].keys()), history_rows
    )
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (results_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("Wrote MLP baseline results:")
    for filename in [
        "fold_metrics.tsv",
        "source_metrics.tsv",
        "source_summary.tsv",
        "training_history.tsv",
        "summary.json",
        "config.json",
    ]:
        print(results_dir / filename)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
