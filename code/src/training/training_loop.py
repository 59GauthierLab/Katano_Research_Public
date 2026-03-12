"""Training loop utilities extracted from main module."""

from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from logging import getLogger
from math import ceil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler

from config.training_config import EarlyStoppingConfig


logger = getLogger(__name__)


def _maybe_debug_cuda_sync(
    *,
    enabled: bool,
    device: torch.device,
    stage: str,
    epoch: int,
    batch_idx: int,
) -> None:
    if not enabled or device.type != "cuda":
        return
    logger.info(
        "CUDA sync (%s) epoch=%d batch=%d",
        stage,
        epoch + 1,
        batch_idx + 1,
    )
    torch.cuda.synchronize()


@dataclass
class InterpretabilitySettings:
    enabled: bool
    split: str
    interval: int
    indices: list[int]
    source_x: np.ndarray | np.memmap | None
    source_y: np.ndarray | np.memmap | None


@dataclass
class TrainEpochMetrics:
    loss: float
    acc: float
    acc_top3: float
    sample_count: int


@dataclass
class EvalEpochMetrics:
    loss: float
    acc: float
    acc_top3: float
    batches: int
    samples: int
    macro_f1: float | None = None
    jpeg_acc: float | None = None
    jpeg_support: int | None = None


class NonFiniteLossError(RuntimeError):
    """Raised when a non-finite loss value is encountered during training."""

    def __init__(self, *, epoch: int, batch_index: int, loss_value: float) -> None:
        self.epoch = epoch
        self.batch_index = batch_index
        self.loss_value = loss_value
        super().__init__(
            f"Non-finite loss detected at epoch {epoch + 1}, batch {batch_index + 1}: {loss_value}"
        )


def _build_train_indices(
    *,
    total_samples: int,
    seed: int,
    epoch: int,
) -> np.ndarray:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return torch.randperm(total_samples, generator=generator).numpy()


def prepare_eval_subset(
    *,
    val_x: np.ndarray | np.memmap,
    val_y: np.ndarray | np.memmap,
    subset_ratio: float | None,
    subset_seed: int,
) -> tuple[np.ndarray | np.memmap, np.ndarray | np.memmap, int, float]:
    total_samples = len(val_y)
    if total_samples == 0:
        return val_x, val_y, 0, 0.0

    if subset_ratio is None or subset_ratio >= 1.0:
        return val_x, val_y, total_samples, 1.0

    subset_size = max(1, int(total_samples * subset_ratio))
    rng = np.random.default_rng(subset_seed)
    indices = rng.permutation(total_samples)[:subset_size]
    return val_x[indices], val_y[indices], subset_size, subset_size / total_samples


@dataclass
class _EarlyStoppingState:
    best_metric_value: float | None = None
    best_epoch_index: int | None = None
    wait_count: int = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    triggered: bool = False


def _train_single_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    train_x: np.ndarray | np.memmap,
    train_y: np.ndarray | np.memmap,
    batch_size: int,
    seed: int,
    device: torch.device,
    epoch: int,
    start_epoch: int,
    progress_step: int,
    label_smoothing: float,
    clip_grad_norm: float | None,
    debug_cuda_sync: bool,
) -> TrainEpochMetrics:
    """Run a single training epoch and collect aggregate metrics."""

    model.train()

    if batch_size <= 0:
        return TrainEpochMetrics(loss=0.0, acc=0.0, acc_top3=0.0, sample_count=0)

    total_samples = len(train_y)
    total_batches = ceil(total_samples / batch_size) if total_samples else 0
    if total_batches == 0:
        return TrainEpochMetrics(loss=0.0, acc=0.0, acc_top3=0.0, sample_count=0)

    train_indices = _build_train_indices(
        total_samples=total_samples,
        seed=seed,
        epoch=epoch,
    )

    epoch_loss_sum = 0.0
    epoch_sample_count = 0
    train_correct_top1 = 0
    train_correct_top3 = 0

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        batch_indices = train_indices[start : start + batch_size]
        slab = train_x[batch_indices]
        if not slab.flags.writeable:
            slab = np.array(slab, copy=True)
        inputs = torch.as_tensor(slab).to(
            device=device,
            dtype=torch.uint8,
            non_blocking=(device.type == "cuda"),
        )
        inputs = inputs.long()
        label_slab = train_y[batch_indices]
        if not label_slab.flags.writeable:
            label_slab = np.array(label_slab, copy=True)
        labels = torch.as_tensor(label_slab).to(
            device=device,
            dtype=torch.long,
            non_blocking=(device.type == "cuda"),
        )
        optimizer.zero_grad()

        use_amp = scaler.is_enabled()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(inputs)
        _maybe_debug_cuda_sync(
            enabled=debug_cuda_sync,
            device=device,
            stage="forward",
            epoch=epoch,
            batch_idx=batch_idx,
        )
        loss = F.cross_entropy(
            outputs,
            labels,
            label_smoothing=label_smoothing,
        )
        _maybe_debug_cuda_sync(
            enabled=debug_cuda_sync,
            device=device,
            stage="loss",
            epoch=epoch,
            batch_idx=batch_idx,
        )

        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            logger.error(
                "Epoch %d batch %d: non-finite loss detected (value=%s)",
                epoch + 1,
                batch_idx + 1,
                loss_value,
            )
            raise NonFiniteLossError(
                epoch=epoch,
                batch_index=batch_idx,
                loss_value=loss_value,
            )

        with torch.no_grad():
            logits = outputs.detach()
            preds_top1 = torch.argmax(logits, dim=1)
            train_correct_top1 += (preds_top1 == labels).sum().item()

            topk = min(3, logits.size(1))
            if topk > 0:
                topk_indices = torch.topk(logits, k=topk, dim=1).indices
                matches_top3 = topk_indices.eq(labels.view(-1, 1))
                train_correct_top3 += matches_top3.any(dim=1).sum().item()

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        _maybe_debug_cuda_sync(
            enabled=debug_cuda_sync,
            device=device,
            stage="backward",
            epoch=epoch,
            batch_idx=batch_idx,
        )
        if clip_grad_norm is not None:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=clip_grad_norm,
            )
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        _maybe_debug_cuda_sync(
            enabled=debug_cuda_sync,
            device=device,
            stage="step",
            epoch=epoch,
            batch_idx=batch_idx,
        )

        batch_samples = labels.size(0)
        epoch_loss_sum += loss.item() * batch_samples
        epoch_sample_count += batch_samples

        if epoch == start_epoch and (batch_idx % progress_step == 0):
            percent = int(batch_idx / total_batches * 100) if total_batches else 0
            logger.info(
                f"Epoch {epoch + 1}:"
                + f"progress {percent}% (batch {batch_idx + 1}/{total_batches})"
            )

    avg_train_loss = (
        float(epoch_loss_sum) / float(epoch_sample_count)
        if epoch_sample_count > 0
        else 0.0
    )
    train_acc = (
        float(train_correct_top1) / float(epoch_sample_count)
        if epoch_sample_count > 0
        else 0.0
    )
    train_acc_top3 = (
        float(train_correct_top3) / float(epoch_sample_count)
        if epoch_sample_count > 0
        else 0.0
    )

    return TrainEpochMetrics(
        loss=avg_train_loss,
        acc=train_acc,
        acc_top3=train_acc_top3,
        sample_count=epoch_sample_count,
    )


def _should_run_validation(epoch: int, start_epoch: int, val_interval: int) -> bool:
    """Return True when validation should run at the given epoch."""

    return (epoch == start_epoch) or ((epoch + 1) % val_interval == 0)


def _run_validation_epoch(
    *,
    model: torch.nn.Module,
    val_x: np.ndarray | np.memmap,
    val_y: np.ndarray | np.memmap,
    batch_size: int,
    device: torch.device,
    max_batches: int | None,
    jpeg_label_id: int | None,
) -> EvalEpochMetrics:
    """Evaluate the model on validation data."""

    return eval_model(
        model,
        val_x,
        val_y,
        batch_size,
        device,
        max_batches=max_batches,
        jpeg_label_id=jpeg_label_id,
    )


def _write_learning_curve_row(
    *,
    path: Path,
    epoch_index: int,
    train_metrics: TrainEpochMetrics,
    eval_metrics: EvalEpochMetrics | None,
    current_lr: float,
) -> None:
    """Append a row to the learning curve CSV file."""

    row = [
        epoch_index + 1,
        train_metrics.loss,
        train_metrics.acc,
        train_metrics.acc_top3,
        "" if eval_metrics is None else eval_metrics.loss,
        "" if eval_metrics is None else eval_metrics.acc,
        "" if eval_metrics is None else eval_metrics.acc_top3,
        "" if eval_metrics is None else eval_metrics.macro_f1,
        "" if eval_metrics is None else eval_metrics.jpeg_acc,
        "" if eval_metrics is None else eval_metrics.jpeg_support,
        current_lr,
    ]

    with path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)


def _maybe_capture_interpretability(
    *,
    interpret: InterpretabilitySettings,
    model: torch.nn.Module,
    device: torch.device,
    run_dir: Path,
    epoch: int,
    start_epoch: int,
    should_eval: bool,
    capture_interpretability_fn: Callable[..., None],
) -> None:
    """Capture interpretability artefacts if the configuration demands it."""

    if (
        not interpret.enabled
        or not should_eval
        or interpret.source_x is None
        or interpret.source_y is None
    ):
        return

    capture_epoch = (epoch + 1) % interpret.interval == 0
    if not capture_epoch and epoch == start_epoch:
        capture_epoch = True

    if not capture_epoch:
        return

    sample_arrays = [np.asarray(interpret.source_x[idx]) for idx in interpret.indices]
    sample_labels = [int(interpret.source_y[idx]) for idx in interpret.indices]

    capture_interpretability_fn(
        model,
        device,
        run_dir,
        epoch,
        samples=sample_arrays,
        labels=sample_labels,
        indices=interpret.indices,
        split=interpret.split,
    )


def _save_epoch_checkpoint(
    *,
    save_checkpoint_fn: Callable[..., None],
    epoch: int,
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    seed: int,
    run_type: str,
    epochs: int,
    batch_size: int,
    train_metrics: TrainEpochMetrics,
    eval_metrics: EvalEpochMetrics | None,
    current_lr: float,
) -> None:
    """Persist a checkpoint for the given epoch."""

    save_checkpoint_fn(
        run_dir=run_dir,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        seed=seed,
        run_type=run_type,
        extra={
            "epochs": epochs,
            "batch_size": batch_size,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.acc,
            "train_acc_top3": train_metrics.acc_top3,
            "val_loss": None if eval_metrics is None else eval_metrics.loss,
            "val_acc": None if eval_metrics is None else eval_metrics.acc,
            "val_acc_top3": None if eval_metrics is None else eval_metrics.acc_top3,
            "val_macro_f1": None if eval_metrics is None else eval_metrics.macro_f1,
            "val_jpeg_acc": None if eval_metrics is None else eval_metrics.jpeg_acc,
            "val_jpeg_support": None if eval_metrics is None else eval_metrics.jpeg_support,
            "lr": current_lr,
        },
    )
    logger.info(f"Epoch {epoch + 1}: checkpoint saved")


def _halve_learning_rate(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> tuple[list[float], list[float]]:
    """Halve learning rates for optimizer and scheduler, returning before/after."""

    before_lrs = [group["lr"] for group in optimizer.param_groups]

    base_lrs_before = list(getattr(scheduler, "base_lrs", []))
    last_lrs_before = list(getattr(scheduler, "_last_lr", []))

    after_lrs = []
    for group in optimizer.param_groups:
        new_lr = group["lr"] * 0.5
        group["lr"] = new_lr
        after_lrs.append(new_lr)

    if base_lrs_before:
        scheduler.base_lrs = [lr * 0.5 for lr in base_lrs_before]
    if last_lrs_before:
        scheduler._last_lr = [lr * 0.5 for lr in last_lrs_before]

    return before_lrs, after_lrs


def _apply_learning_rate_target(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    target_lrs: list[float],
) -> tuple[list[float], list[float]]:
    """Set learning rates to target values for optimizer and scheduler."""

    before_lrs = [group["lr"] for group in optimizer.param_groups]
    if len(target_lrs) != len(before_lrs):
        raise ValueError("target_lrs length mismatch with optimizer param groups")

    for group, new_lr in zip(optimizer.param_groups, target_lrs):
        group["lr"] = new_lr

    base_lrs_before = list(getattr(scheduler, "base_lrs", []))
    last_lrs_before = list(getattr(scheduler, "_last_lr", []))
    ratios = [
        (new_lr / before_lr) if before_lr else 0.0
        for before_lr, new_lr in zip(before_lrs, target_lrs)
    ]

    if base_lrs_before and len(base_lrs_before) == len(ratios):
        scheduler.base_lrs = [
            lr * ratio for lr, ratio in zip(base_lrs_before, ratios)
        ]
    if last_lrs_before and len(last_lrs_before) == len(ratios):
        scheduler._last_lr = [
            lr * ratio for lr, ratio in zip(last_lrs_before, ratios)
        ]

    return before_lrs, target_lrs


def _maybe_fast_forward_scheduler(
    *,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    start_epoch: int,
    resume_checkpoint: dict[str, Any] | None,
) -> None:
    """
    Advance the scheduler using the chainable API so callers do not need
    to pass an epoch argument (deprecated in PyTorch 2.5+).
    """

    if start_epoch <= 0 or resume_checkpoint is not None:
        return

    for _ in range(start_epoch):
        scheduler.step()

    logger.info(
        "Scheduler fast-forwarded by %d steps to align with start_epoch "
        "without using the deprecated epoch argument",
        start_epoch,
    )


def _save_best_checkpoint(run_dir: Path) -> None:
    """Copy the latest checkpoint to a best checkpoint path."""

    latest_path = run_dir / "checkpoint_latest.pt"
    best_path = run_dir / "checkpoint_best.pt"
    if not latest_path.exists():
        logger.warning("Best checkpoint update skipped: %s not found", latest_path)
        return
    shutil.copyfile(latest_path, best_path)
    logger.info("Best checkpoint updated: %s", best_path.name)


def _perform_validation(
    *,
    should_eval: bool,
    epoch: int,
    val_interval: int,
    model: torch.nn.Module,
    val_x: np.ndarray | np.memmap,
    val_y: np.ndarray | np.memmap,
    batch_size: int,
    device: torch.device,
    max_batches: int | None,
    subset_ratio: float | None,
    subset_seed: int,
    jpeg_label_id: int | None,
) -> EvalEpochMetrics | None:
    if not should_eval:
        logger.info(
            f"Epoch {epoch + 1}: Validation skipped (interval={val_interval})"
        )
        return None

    val_total_samples = len(val_y)
    val_x_subset, val_y_subset, subset_samples, subset_ratio_used = prepare_eval_subset(
        val_x=val_x,
        val_y=val_y,
        subset_ratio=subset_ratio,
        subset_seed=subset_seed,
    )

    eval_metrics = _run_validation_epoch(
        model=model,
        val_x=val_x_subset,
        val_y=val_y_subset,
        batch_size=batch_size,
        device=device,
        max_batches=max_batches,
        jpeg_label_id=jpeg_label_id,
    )
    val_ratio_display = (
        f"{subset_ratio_used:.3f}" if val_total_samples > 0 else "n/a"
    )
    subset_log = (
        f", val_samples_used={eval_metrics.samples}/{val_total_samples}, "
        f"val_subset_samples={subset_samples}, "
        f"val_subset_ratio={val_ratio_display}"
    )
    val_log = (
        f"Epoch {epoch + 1}: Validation:"
        + f" batches={eval_metrics.batches}, "
        + f"samples={eval_metrics.samples}, "
        + f"val_loss={eval_metrics.loss:.4f}, "
        + f"val_acc={eval_metrics.acc:.3f}, "
        + f"acc_top3={eval_metrics.acc_top3:.3f}"
        + subset_log
    )
    if eval_metrics.macro_f1 is not None:
        val_log += f", macro_f1={eval_metrics.macro_f1:.3f}"
    if eval_metrics.jpeg_support is not None:
        jpeg_display = (
            f"{eval_metrics.jpeg_acc:.3f}"
            if eval_metrics.jpeg_acc is not None
            else "n/a"
        )
        val_log += f", jpeg_acc={jpeg_display} (support={eval_metrics.jpeg_support})"
    logger.info(val_log)
    return eval_metrics


def _update_early_stopping(
    *,
    state: _EarlyStoppingState,
    early_stopping: EarlyStoppingConfig,
    eval_metrics: EvalEpochMetrics,
    epoch: int,
    model: torch.nn.Module,
    unwrap_compiled_fn: Callable[[torch.nn.Module], torch.nn.Module],
) -> tuple[bool, bool]:
    metric_name = early_stopping.metric
    if metric_name == "val_loss":
        metric_value = eval_metrics.loss
    elif metric_name == "val_acc":
        metric_value = eval_metrics.acc
    else:
        metric_value = eval_metrics.macro_f1

    if metric_value is None:
        logger.warning(
            "Early stopping metric '%s' is unavailable at epoch %d; skipping",
            metric_name,
            epoch + 1,
        )
        return False, False

    if math.isnan(metric_value):
        logger.warning(
            "Early stopping metric '%s' is NaN at epoch %d; skipping",
            metric_name,
            epoch + 1,
        )
        return False, False

    if state.best_metric_value is None:
        improved = True
    elif early_stopping.mode == "min":
        improved = metric_value < state.best_metric_value - early_stopping.min_delta
    else:
        improved = metric_value > state.best_metric_value + early_stopping.min_delta

    if improved:
        prev_best = state.best_metric_value
        state.best_metric_value = metric_value
        state.best_epoch_index = epoch
        state.wait_count = 0
        if early_stopping.restore_best:
            base_model = unwrap_compiled_fn(model)
            state.best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in base_model.state_dict().items()
            }
        if prev_best is None:
            logger.info(
                "Early stopping metric '%s' initialized at %.4f (epoch %d)",
                metric_name,
                metric_value,
                epoch + 1,
            )
        else:
            direction = "decreased" if early_stopping.mode == "min" else "increased"
            logger.info(
                "Early stopping metric '%s' %s from %.4f to %.4f (epoch %d)",
                metric_name,
                direction,
                prev_best,
                metric_value,
                epoch + 1,
            )
        return False, True

    state.wait_count += 1
    logger.info(
        "Early stopping metric '%s' not improved (%d/%d)",
        metric_name,
        state.wait_count,
        early_stopping.patience,
    )
    if state.wait_count >= early_stopping.patience:
        state.triggered = True
        if (
            state.best_metric_value is not None
            and state.best_epoch_index is not None
        ):
            logger.info(
                "Early stopping triggered at epoch %d (best epoch %d, %s=%.4f)",
                epoch + 1,
                state.best_epoch_index + 1,
                metric_name,
                state.best_metric_value,
            )
        else:
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
        return True, False

    return False, False


def run_training_loop(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    resume_checkpoint: dict[str, Any] | None,
    train_x: np.ndarray | np.memmap,
    train_y: np.ndarray | np.memmap,
    val_x: np.ndarray | np.memmap,
    val_y: np.ndarray | np.memmap,
    batch_size: int,
    epochs: int,
    start_epoch: int,
    seed: int,
    device: torch.device,
    run_dir: Path,
    run_type: str,
    val_interval: int,
    val_max_batches: int | None,
    val_subset_ratio: float | None,
    val_subset_seed: int | None,
    jpeg_label_id: int | None,
    clip_grad_norm: float | None,
    label_smoothing: float,
    debug_cuda_sync: bool,
    use_amp: bool,
    early_stopping: EarlyStoppingConfig | None,
    interpret: InterpretabilitySettings,
    learning_curve_path: Path,
    save_checkpoint_fn: Callable[..., None],
    load_checkpoint_fn: Callable[..., dict[str, Any]],
    capture_interpretability_fn: Callable[..., None],
    unwrap_compiled_fn: Callable[[torch.nn.Module], torch.nn.Module],
) -> None:
    """Train the model for the given configuration."""

    total_batches = ceil(len(train_y) / batch_size) if batch_size else 0
    progress_step = max(1, total_batches // 10) if total_batches else 1

    if start_epoch >= epochs:
        logger.info(
            "チェックポイントのエポック数が学習回数に到達しているため、追加学習はスキップします"
        )
        return

    early_stopping_state = _EarlyStoppingState()
    subset_seed = seed if val_subset_seed is None else val_subset_seed

    latest_checkpoint_path = run_dir / "checkpoint_latest.pt"
    scaler = GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    recovery_attempts = 0
    recovery_prev_lrs: list[float] | None = None

    _maybe_fast_forward_scheduler(
        scheduler=scheduler,
        start_epoch=start_epoch,
        resume_checkpoint=resume_checkpoint,
    )

    epoch = start_epoch
    while epoch < epochs:
        try:
            train_metrics = _train_single_epoch(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                train_x=train_x,
                train_y=train_y,
                batch_size=batch_size,
                seed=seed,
                device=device,
                epoch=epoch,
                start_epoch=start_epoch,
                progress_step=progress_step,
                label_smoothing=label_smoothing,
                clip_grad_norm=clip_grad_norm,
                debug_cuda_sync=debug_cuda_sync,
            )
        except NonFiniteLossError as err:
            logger.warning(
                "Epoch %d batch %d: loss became %s; attempting recovery",
                err.epoch + 1,
                err.batch_index + 1,
                err.loss_value,
            )

            recovery_attempts += 1
            if recovery_attempts > 3:
                logger.error(
                    "Non-finite loss recovery failed after %d attempts; aborting",
                    recovery_attempts - 1,
                )
                raise

            if not latest_checkpoint_path.exists():
                logger.error(
                    "最新のチェックポイントが存在しないため、NaN 発生後の復旧に失敗しました"
                )
                raise

            _ = load_checkpoint_fn(
                checkpoint_path=latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )

            base_lrs = [group["lr"] for group in optimizer.param_groups]
            if recovery_prev_lrs is None:
                target_lrs = [lr * 0.5 for lr in base_lrs]
                log_before_lrs = base_lrs
            else:
                target_lrs = [lr * 0.5 for lr in recovery_prev_lrs]
                log_before_lrs = recovery_prev_lrs

            before_lrs, after_lrs = _apply_learning_rate_target(
                optimizer,
                scheduler,
                target_lrs,
            )
            recovery_prev_lrs = after_lrs
            before_lr_repr = log_before_lrs[0] if log_before_lrs else float("nan")
            after_lr_repr = after_lrs[0] if after_lrs else float("nan")

            logger.info(
                "チェックポイント %s を読み込み、学習率を %.6g -> %.6g に調整しました",
                latest_checkpoint_path.name,
                before_lr_repr,
                after_lr_repr,
            )

            logger.info(
                "Non-finite loss recovery completed: optimizer.step をスキップし、"
                "チェックポイントから復旧済み"
            )

            optimizer.zero_grad(set_to_none=True)
            continue

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_metrics.loss:.4f}, "
            f"train_acc={train_metrics.acc:.4f}, "
            f"train_acc_top3={train_metrics.acc_top3:.4f}, "
            f"lr={current_lr:.6g}"
        )

        should_eval = _should_run_validation(epoch, start_epoch, val_interval)
        eval_metrics = _perform_validation(
            should_eval=should_eval,
            epoch=epoch,
            val_interval=val_interval,
            model=model,
            val_x=val_x,
            val_y=val_y,
            batch_size=batch_size,
            device=device,
            max_batches=val_max_batches,
            subset_ratio=val_subset_ratio,
            subset_seed=subset_seed,
            jpeg_label_id=jpeg_label_id,
        )

        stop_training = False
        improved = False
        if early_stopping and should_eval and eval_metrics is not None:
            stop_training, improved = _update_early_stopping(
                state=early_stopping_state,
                early_stopping=early_stopping,
                eval_metrics=eval_metrics,
                epoch=epoch,
                model=model,
                unwrap_compiled_fn=unwrap_compiled_fn,
            )

        _write_learning_curve_row(
            path=learning_curve_path,
            epoch_index=epoch,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            current_lr=current_lr,
        )

        _maybe_capture_interpretability(
            interpret=interpret,
            model=model,
            device=device,
            run_dir=run_dir,
            epoch=epoch,
            start_epoch=start_epoch,
            should_eval=should_eval,
            capture_interpretability_fn=capture_interpretability_fn,
        )

        scheduler.step()

        logger.info(f"Epoch {epoch + 1}: done!")

        _save_epoch_checkpoint(
            save_checkpoint_fn=save_checkpoint_fn,
            epoch=epoch,
            run_dir=run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            seed=seed,
            run_type=run_type,
            epochs=epochs,
            batch_size=batch_size,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            current_lr=current_lr,
        )
        recovery_attempts = 0
        recovery_prev_lrs = None
        if improved:
            _save_best_checkpoint(run_dir)

        if stop_training:
            break

        epoch += 1

    if (
        early_stopping
        and early_stopping.restore_best
        and early_stopping_state.best_model_state is not None
        and early_stopping_state.best_epoch_index is not None
    ):
        base_model = unwrap_compiled_fn(model)
        base_model.load_state_dict(early_stopping_state.best_model_state)
        metric_repr = (
            f"{early_stopping_state.best_metric_value:.4f}"
            if early_stopping_state.best_metric_value is not None
            else "n/a"
        )
        logger.info(
            "Restored best model weights from epoch %d for evaluation (%s=%s)",
            early_stopping_state.best_epoch_index + 1,
            early_stopping.metric,
            metric_repr,
        )


def eval_model(
    model: torch.nn.Module,
    x_memmap: np.ndarray | np.memmap,
    y_memmap: np.ndarray | np.memmap,
    batch_size: int,
    device: torch.device,
    *,
    max_batches: int | None = None,
    jpeg_label_id: int | None = None,
    confusion_output_path: Path | None = None,
) -> EvalEpochMetrics:
    """
    Run evaluation on the provided dataset and compute metrics.
    """

    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    loss_sum = 0.0
    confusion: np.ndarray | None = None
    num_classes: int | None = None
    jpeg_correct = 0
    jpeg_support = 0

    total_batches = ceil(len(y_memmap) / batch_size)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    with torch.no_grad():
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            slab = x_memmap[start : start + batch_size]
            if not slab.flags.writeable:
                slab = np.array(slab, copy=True)
            inputs = torch.as_tensor(slab).to(
                device=device,
                dtype=torch.uint8,
                non_blocking=(device.type == "cuda"),
            )
            inputs = inputs.long()
            true_np = y_memmap[start : start + batch_size]
            if not true_np.flags.writeable:
                true_np = np.array(true_np, copy=True)
            targets = torch.as_tensor(true_np).to(
                device=device,
                dtype=torch.long,
                non_blocking=(device.type == "cuda"),
            )
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)

            preds_top1 = torch.argmax(logits, dim=1)
            correct_top1 += (preds_top1 == targets).sum().item()
            if jpeg_label_id is not None:
                jpeg_mask = targets == jpeg_label_id
                jpeg_support += int(jpeg_mask.sum().item())
                jpeg_correct += int(
                    (preds_top1[jpeg_mask] == targets[jpeg_mask]).sum().item()
                )
            pred_np = preds_top1.detach().cpu().numpy().astype(np.int64, copy=False)
            targets_np = targets.detach().cpu().numpy().astype(np.int64, copy=False)

            if confusion is None:
                num_classes = logits.size(1)
                confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
            elif num_classes is not None and logits.size(1) != num_classes:
                raise ValueError("評価途中でクラス数が変化しました")

            if confusion is not None and num_classes is not None:
                index = targets_np * num_classes + pred_np
                counts = np.bincount(
                    index,
                    minlength=num_classes * num_classes,
                )
                confusion += counts.reshape(num_classes, num_classes)

            topk = min(3, logits.size(1))
            if topk > 0:
                topk_indices = torch.topk(logits, k=topk, dim=1).indices
                matches_top3 = topk_indices.eq(targets.view(-1, 1))
                correct_top3 += matches_top3.any(dim=1).sum().item()

            batch_size_actual = targets.size(0)
            total += batch_size_actual
            loss_sum += loss.item() * batch_size_actual

    accuracy = float(correct_top1) / float(total) if total > 0 else 0.0
    top3_accuracy = float(correct_top3) / float(total) if total > 0 else 0.0
    avg_loss = float(loss_sum) / float(total) if total > 0 else 0.0

    macro_f1: float | None = None
    if confusion is not None and num_classes is not None and total > 0:
        per_class_scores: list[float] = []
        for class_idx in range(num_classes):
            tp = float(confusion[class_idx, class_idx])
            fp = float(confusion[:, class_idx].sum() - tp)
            fn = float(confusion[class_idx, :].sum() - tp)
            denom = (2.0 * tp) + fp + fn
            if denom > 0.0:
                per_class_scores.append((2.0 * tp) / denom)
            else:
                per_class_scores.append(0.0)
        if per_class_scores:
            macro_f1 = float(np.mean(per_class_scores))

    jpeg_acc: float | None = None
    jpeg_support_value: int | None = None
    if jpeg_label_id is not None:
        jpeg_support_value = jpeg_support
        if jpeg_support > 0:
            jpeg_acc = float(jpeg_correct) / float(jpeg_support)

    if confusion_output_path is not None:
        if confusion is None or num_classes is None:
            raise ValueError("confusion matrix が生成できませんでした")
        if confusion.dtype != np.int64:
            raise ValueError(
                f"confusion dtype must be int64 (got {confusion.dtype})"
            )
        if confusion.shape != (num_classes, num_classes):
            raise ValueError(
                f"confusion shape mismatch: {confusion.shape} != ({num_classes}, {num_classes})"
            )
        if total > 0 and int(confusion.sum()) != int(total):
            raise ValueError(
                f"confusion sum mismatch: {confusion.sum()} != {total}"
            )
        if np.any(confusion < 0):
            raise ValueError("confusion matrix に負の値が含まれています")
        confusion_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(confusion_output_path, confusion)
        logger.info("Saved confusion matrix: %s", confusion_output_path)

    return EvalEpochMetrics(
        loss=avg_loss,
        acc=accuracy,
        acc_top3=top3_accuracy,
        batches=total_batches,
        samples=total,
        macro_f1=macro_f1,
        jpeg_acc=jpeg_acc,
        jpeg_support=jpeg_support_value,
    )
