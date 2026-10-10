#!/usr/bin/env python3
"""Run a direct-translation autoencoder baseline on ds004752 low-frequency beamforming trials."""

from __future__ import annotations

import argparse
import copy
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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class SourceAutoencoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class TranslationAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.scalp_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.scalp_encoder(x))


def evaluate_loss(
    model: nn.Module,
    x_array: np.ndarray,
    y_array: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    criterion = nn.MSELoss()
    losses: list[float] = []
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
            losses.append(float(criterion(model(batch_x), batch_y).item()))
    return float(np.mean(losses)) if losses else 0.0


def predict_array(
    model: nn.Module,
    x_array: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x_array.shape[0], batch_size):
            end = start + batch_size
            batch_x = torch.from_numpy(x_array[start:end]).to(
                device=device, dtype=torch.float32
            )
            outputs.append(model(batch_x).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
    freeze_decoder_epochs: int = 0,
) -> tuple[nn.Module, list[dict[str, object]], int, float]:
    train_dataset = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed + seed_offset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
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
        if freeze_decoder_epochs > 0 and hasattr(model, "decoder"):
            freeze = epoch <= freeze_decoder_epochs
            for parameter in model.decoder.parameters():
                parameter.requires_grad = not freeze

        model.train()
        batch_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_loss = evaluate_loss(model, x_val, y_val, args.batch_size, device)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if val_loss < (best_val_loss - args.min_delta):
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    model.load_state_dict(best_state)
    return model, history_rows, best_epoch, best_val_loss


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
        "--latent-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--ae-hidden-dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--translation-hidden-dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--freeze-decoder-epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results/ds004752_beamforming_translation_ae_ld64",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    split_manifest = args.split_manifest.resolve()
    cache_root = args.cache_root.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

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

    metric_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    source_summary_rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}

    folds = sorted({row["fold_id"] for row in split_rows})
    phases = sorted({row["phase"] for row in split_rows})
    subjects = sorted({row["participant_id"] for row in split_rows})

    for phase in phases:
        phase_fold_rows: list[dict[str, object]] = []
        n_timepoints = int(phase_metadata[phase]["n_timepoints"])
        for fold_index, fold_id in enumerate(folds):
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

            x_train_z, x_val_z, x_mean, x_std = zscore_train(x_train, x_val)
            _, x_test_z, _, _ = zscore_train(x_train, x_test)
            y_train_z, y_val_z, y_mean, y_std = zscore_train(y_train, y_val)
            _, y_test_z, _, _ = zscore_train(y_train, y_test)

            autoencoder = SourceAutoencoder(
                input_dim=y_train.shape[1],
                hidden_dim=args.ae_hidden_dim,
                latent_dim=args.latent_dim,
                dropout=args.dropout,
            ).to(device)
            autoencoder, ae_history, ae_best_epoch, ae_best_val_loss = train_model(
                model=autoencoder,
                x_train=y_train_z,
                y_train=y_train_z,
                x_val=y_val_z,
                y_val=y_val_z,
                args=args,
                device=device,
                seed_offset=(fold_index * 2),
                freeze_decoder_epochs=0,
            )
            for row in ae_history:
                row.update(
                    {
                        "fold_id": fold_id,
                        "phase": phase,
                        "stage": "source_autoencoder",
                    }
                )

            translator = TranslationAutoencoder(
                input_dim=x_train.shape[1],
                hidden_dim=args.translation_hidden_dim,
                latent_dim=args.latent_dim,
                dropout=args.dropout,
                decoder=copy.deepcopy(autoencoder.decoder),
            ).to(device)
            translator, translation_history, best_epoch, best_val_loss = train_model(
                model=translator,
                x_train=x_train_z,
                y_train=y_train_z,
                x_val=x_val_z,
                y_val=y_val_z,
                args=args,
                device=device,
                seed_offset=(fold_index * 2) + 1,
                freeze_decoder_epochs=args.freeze_decoder_epochs,
            )
            for row in translation_history:
                row.update(
                    {
                        "fold_id": fold_id,
                        "phase": phase,
                        "stage": "translation",
                    }
                )
            history_rows.extend(ae_history)
            history_rows.extend(translation_history)

            y_pred_z = predict_array(translator, x_test_z, args.batch_size, device)
            y_pred = (y_pred_z * y_std) + y_mean

            metrics = {
                "fold_id": fold_id,
                "holdout_subject": holdout_subject,
                "validation_subject": validation_subject,
                "phase": phase,
                "n_train_trials": x_train.shape[0],
                "n_val_trials": x_val.shape[0],
                "n_test_trials": x_test.shape[0],
                "latent_dim": args.latent_dim,
                "pearson_r": pearson_corr(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "nrmse": nrmse(y_test, y_pred),
                "ae_best_epoch": ae_best_epoch,
                "ae_best_val_loss": ae_best_val_loss,
                "translation_best_epoch": best_epoch,
                "translation_best_val_loss": best_val_loss,
            }
            metric_rows.append(metrics)
            phase_fold_rows.append(metrics)

            for source_metric in sourcewise_metrics(
                y_test, y_pred, source_labels, n_timepoints
            ):
                source_rows.append(
                    {
                        "fold_id": fold_id,
                        "holdout_subject": holdout_subject,
                        "phase": phase,
                        "latent_dim": args.latent_dim,
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
            "mean_ae_best_epoch": float(
                np.mean([row["ae_best_epoch"] for row in phase_fold_rows])
            ),
            "mean_translation_best_epoch": float(
                np.mean([row["translation_best_epoch"] for row in phase_fold_rows])
            ),
            "mean_ae_best_val_loss": float(
                np.mean([row["ae_best_val_loss"] for row in phase_fold_rows])
            ),
            "mean_translation_best_val_loss": float(
                np.mean([row["translation_best_val_loss"] for row in phase_fold_rows])
            ),
        }

    for phase in phases:
        for source_label in source_labels:
            rows = [
                row
                for row in source_rows
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

    write_tsv(
        results_dir / "fold_metrics.tsv", list(metric_rows[0].keys()), metric_rows
    )
    write_tsv(
        results_dir / "source_metrics.tsv", list(source_rows[0].keys()), source_rows
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
        json.dumps(
            {
                "split_manifest": str(split_manifest),
                "cache_root": str(cache_root),
                "latent_dim": args.latent_dim,
                "ae_hidden_dim": args.ae_hidden_dim,
                "translation_hidden_dim": args.translation_hidden_dim,
                "dropout": args.dropout,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "freeze_decoder_epochs": args.freeze_decoder_epochs,
                "seed": args.seed,
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Wrote translation autoencoder baseline results:")
    print(results_dir / "fold_metrics.tsv")
    print(results_dir / "source_metrics.tsv")
    print(results_dir / "source_summary.tsv")
    print(results_dir / "training_history.tsv")
    print(results_dir / "summary.json")
    print(results_dir / "config.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
