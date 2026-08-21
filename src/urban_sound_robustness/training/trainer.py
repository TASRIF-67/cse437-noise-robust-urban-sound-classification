"""Training orchestration, checkpoints, history, and early stopping."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from urban_sound_robustness.training.engine import EpochResult, run_epoch
from urban_sound_robustness.training.factory import (
    create_loss_function,
    create_optimizer,
    create_scheduler,
)


LOGGER = logging.getLogger("urban_sound_robustness.training")


class EarlyStopping:
    """Track a monitored value and report when patience is exhausted."""

    def __init__(self, *, mode: str, patience: int) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("Early-stopping mode must be min or max.")
        if patience < 1:
            raise ValueError("Early-stopping patience must be at least one.")
        self.mode = mode
        self.patience = patience
        self.best_value: float | None = None
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        """Return true when the current value exhausts configured patience."""
        improved = (
            self.best_value is None
            or self.mode == "max"
            and value > self.best_value
            or self.mode == "min"
            and value < self.best_value
        )
        if improved:
            self.best_value = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


@dataclass(frozen=True)
class TrainingOutcome:
    """Paths and state produced by a completed training call."""

    history: pd.DataFrame
    history_path: Path
    best_checkpoint: Path | None
    last_checkpoint: Path | None
    best_metric: float | None
    epochs_completed: int
    stopped_early: bool


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, float],
    configuration: Mapping[str, Any] | None = None,
) -> Path:
    """Save model/optimizer states and reproducibility metadata."""
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_class": model.__class__.__name__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": dict(metrics),
            "configuration": None if configuration is None else dict(configuration),
        },
        resolved,
    )
    return resolved


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore a model and optionally optimizer from a project checkpoint."""
    checkpoint = torch.load(
        Path(path).expanduser().resolve(),
        map_location=map_location,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


class Trainer:
    """Coordinate reusable epoch loops and experiment-state persistence."""

    def __init__(
        self,
        model: nn.Module,
        class_names: Sequence[str],
        training_settings: Mapping[str, Any],
        *,
        device: torch.device,
        checkpoint_directory: str | Path,
        history_path: str | Path,
        tensorboard_directory: str | Path | None = None,
        configuration: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.class_names = tuple(class_names)
        self.settings = dict(training_settings)
        self.device = device
        self.checkpoint_directory = (
            Path(checkpoint_directory).expanduser().resolve()
        )
        self.history_path = Path(history_path).expanduser().resolve()
        self.tensorboard_directory = (
            None
            if tensorboard_directory is None
            else Path(tensorboard_directory).expanduser().resolve()
        )
        self.configuration = configuration
        self.loss_function = create_loss_function(self.settings["loss"])
        self.optimizer = create_optimizer(
            self.model.parameters(), self.settings["optimizer"]
        )
        self.scheduler = create_scheduler(
            self.optimizer, self.settings.get("scheduler")
        )
        amp_enabled = bool(self.settings.get("mixed_precision", False)) and (
            device.type == "cuda"
        )
        self.gradient_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )

    def fit(
        self,
        train_loader,
        validation_loader,
        *,
        epochs: int | None = None,
        max_train_batches: int | None = None,
        max_validation_batches: int | None = None,
    ) -> TrainingOutcome:
        """Train, validate, checkpoint, schedule, and record history."""
        total_epochs = int(self.settings["epochs"]) if epochs is None else epochs
        if total_epochs < 1:
            raise ValueError("epochs must be at least one.")
        checkpoint_settings = dict(self.settings.get("checkpointing", {}))
        monitor = str(checkpoint_settings.get("monitor", "macro_f1"))
        mode = str(checkpoint_settings.get("mode", "max"))
        save_best_enabled = bool(checkpoint_settings.get("save_best", True))
        save_last_enabled = bool(checkpoint_settings.get("save_last", True))
        early_settings = dict(self.settings.get("early_stopping", {}))
        early_stopping = None
        if early_settings.get("enabled", False):
            early_stopping = EarlyStopping(
                mode=str(early_settings.get("mode", mode)),
                patience=int(early_settings["patience"]),
            )

        writer = None
        logging_settings = dict(self.settings.get("logging", {}))
        if logging_settings.get("tensorboard", False):
            if self.tensorboard_directory is None:
                raise ValueError(
                    "tensorboard_directory is required when TensorBoard is enabled."
                )
            writer = SummaryWriter(self.tensorboard_directory)

        history_rows: list[dict[str, float | int]] = []
        best_metric: float | None = None
        best_checkpoint: Path | None = None
        last_checkpoint: Path | None = None
        stopped_early = False
        try:
            for epoch in range(1, total_epochs + 1):
                LOGGER.info("Starting epoch %d/%d", epoch, total_epochs)
                train_result = run_epoch(
                    self.model,
                    train_loader,
                    self.loss_function,
                    self.device,
                    self.class_names,
                    optimizer=self.optimizer,
                    gradient_scaler=self.gradient_scaler,
                    mixed_precision=bool(
                        self.settings.get("mixed_precision", False)
                    ),
                    gradient_accumulation_steps=int(
                        self.settings.get("gradient_accumulation_steps", 1)
                    ),
                    max_batches=max_train_batches,
                )
                validation_result = run_epoch(
                    self.model,
                    validation_loader,
                    self.loss_function,
                    self.device,
                    self.class_names,
                    mixed_precision=bool(
                        self.settings.get("mixed_precision", False)
                    ),
                    max_batches=max_validation_batches,
                )
                learning_rate = float(self.optimizer.param_groups[0]["lr"])
                row = self._history_row(
                    epoch,
                    train_result,
                    validation_result,
                    learning_rate,
                )
                history_rows.append(row)
                monitored_value = float(validation_result.metrics[monitor])
                LOGGER.info(
                    "Epoch %d complete: train_loss=%.4f validation_loss=%.4f "
                    "validation_%s=%.4f",
                    epoch,
                    train_result.loss,
                    validation_result.loss,
                    monitor,
                    monitored_value,
                )
                improved = (
                    best_metric is None
                    or mode == "max"
                    and monitored_value > best_metric
                    or mode == "min"
                    and monitored_value < best_metric
                )
                if improved:
                    best_metric = monitored_value
                    if save_best_enabled:
                        best_checkpoint = save_checkpoint(
                            self.checkpoint_directory / "best.pt",
                            model=self.model,
                            optimizer=self.optimizer,
                            epoch=epoch,
                            metrics=validation_result.metrics,
                            configuration=self.configuration,
                        )
                        LOGGER.info("Saved best checkpoint: %s", best_checkpoint)
                if save_last_enabled:
                    last_checkpoint = save_checkpoint(
                        self.checkpoint_directory / "last.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=validation_result.metrics,
                        configuration=self.configuration,
                    )

                history = pd.DataFrame(history_rows)
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
                history.to_csv(self.history_path, index=False)
                if writer is not None:
                    self._write_tensorboard(
                        writer, epoch, train_result, validation_result, learning_rate
                    )
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(monitored_value)
                elif self.scheduler is not None:
                    self.scheduler.step()
                if early_stopping is not None and early_stopping.update(
                    monitored_value
                ):
                    stopped_early = True
                    LOGGER.info("Early stopping triggered after epoch %d", epoch)
                    break
        finally:
            if writer is not None:
                writer.close()

        final_history = pd.DataFrame(history_rows)
        return TrainingOutcome(
            history=final_history,
            history_path=self.history_path,
            best_checkpoint=best_checkpoint,
            last_checkpoint=last_checkpoint,
            best_metric=best_metric,
            epochs_completed=len(final_history),
            stopped_early=stopped_early,
        )

    @staticmethod
    def _history_row(
        epoch: int,
        train: EpochResult,
        validation: EpochResult,
        learning_rate: float,
    ) -> dict[str, float | int]:
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train.loss,
            "validation_loss": validation.loss,
        }
        for name, value in train.metrics.items():
            row[f"train_{name}"] = value
        for name, value in validation.metrics.items():
            row[f"validation_{name}"] = value
        return row

    @staticmethod
    def _write_tensorboard(
        writer: SummaryWriter,
        epoch: int,
        train: EpochResult,
        validation: EpochResult,
        learning_rate: float,
    ) -> None:
        writer.add_scalar("loss/train", train.loss, epoch)
        writer.add_scalar("loss/validation", validation.loss, epoch)
        writer.add_scalar("learning_rate", learning_rate, epoch)
        for name, value in train.metrics.items():
            writer.add_scalar(f"metrics/train_{name}", value, epoch)
        for name, value in validation.metrics.items():
            writer.add_scalar(f"metrics/validation_{name}", value, epoch)
