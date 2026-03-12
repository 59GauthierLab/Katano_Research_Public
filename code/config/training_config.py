"""Utility helpers for building model and training configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Any, Tuple

import numpy as np


logger = getLogger(__name__)


@dataclass
class EarlyStoppingConfig:
    metric: str
    mode: str
    patience: int
    min_delta: float
    restore_best: bool


@dataclass
class TrainingHyperParams:
    batch_size: int
    epochs: int
    lr: float
    eta_min: float
    optimizer: str
    scheduler: str
    warmup_epochs: int
    warmup_start_factor: float
    val_interval: int
    phase: str
    val_max_batches: int | None
    val_subset_ratio: float | None
    val_subset_seed: int | None
    n_subset: int
    n_classes: int
    weight_decay: float
    clip_grad_norm: float | None
    label_smoothing: float
    debug_cuda_sync: bool
    use_amp: bool
    early_stopping: EarlyStoppingConfig | None


def _parse_early_stopping_config(
    common_training: dict[str, Any],
) -> EarlyStoppingConfig | None:
    """Parse early stopping configuration and return a dataclass if enabled."""

    early_cfg = common_training.get("early_stopping")
    if not early_cfg:
        return None

    if not isinstance(early_cfg, dict):
        raise TypeError("training.common.early_stopping は辞書形式である必要があります")

    if not bool(early_cfg.get("enable", False)):
        return None

    metric_raw = str(early_cfg.get("metric", "val_loss")).strip().lower()
    metric = metric_raw or "val_loss"
    allowed_metrics = {"val_loss", "val_acc", "val_macro_f1"}
    if metric not in allowed_metrics:
        raise ValueError(
            "training.common.early_stopping.metric は "
            "'val_loss' / 'val_acc' / 'val_macro_f1' のいずれかで指定してください"
        )

    mode_raw = early_cfg.get("mode")
    if mode_raw is None or str(mode_raw).strip() == "":
        mode = "min" if metric == "val_loss" else "max"
    else:
        mode = str(mode_raw).strip().lower()
        if mode not in {"min", "max"}:
            raise ValueError(
                "training.common.early_stopping.mode は 'min' または 'max' で指定してください"
            )

    patience_raw = early_cfg.get("patience", 5)
    try:
        patience = int(patience_raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 設定エラー
        raise ValueError(
            "early_stopping.patience は正の整数で指定してください"
        ) from exc
    if patience < 1:
        raise ValueError("early_stopping.patience は 1 以上で指定してください")

    min_delta_raw = early_cfg.get("min_delta", 0.0)
    try:
        min_delta = float(min_delta_raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 設定エラー
        raise ValueError(
            "early_stopping.min_delta は数値で指定してください"
        ) from exc
    if min_delta < 0.0:
        raise ValueError("early_stopping.min_delta は 0 以上で指定してください")

    restore_best = bool(early_cfg.get("restore_best", True))

    return EarlyStoppingConfig(
        metric=metric,
        mode=mode,
        patience=patience,
        min_delta=min_delta,
        restore_best=restore_best,
    )


def _resolve_training_sections(
    training_root: dict[str, Any],
    run_type: str,
) -> Tuple[dict[str, Any], dict[str, Any]]:
    if run_type not in training_root:
        raise KeyError(f"training.{run_type} が config から見つかりません")

    common_training = training_root.get("common", {})
    if not isinstance(common_training, dict):
        raise TypeError("training.common は辞書形式である必要があります")

    specific_training = training_root.get(run_type, {})
    if not isinstance(specific_training, dict):
        raise TypeError(f"training.{run_type} は辞書形式である必要があります")

    return common_training, specific_training


def _normalize_val_interval(raw: Any) -> int:
    val_interval_raw = int(raw)
    if val_interval_raw < 1:
        logger.warning("val_interval が 1 未満のため 1 に切り上げて適用します")
    return max(1, val_interval_raw)


def _normalize_phase(raw: Any) -> str:
    if raw is None:
        return "final"
    value = str(raw).strip().lower()
    if not value:
        return "final"
    if value not in {"search", "final"}:
        raise ValueError("training.phase は 'search' または 'final' で指定してください")
    return value


def _normalize_val_max_batches(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 設定エラー
        raise ValueError("training.val_max_batches は整数で指定してください") from exc
    if value < 1:
        logger.warning("val_max_batches が 1 未満のため無効化します")
        return None
    return value


def _normalize_val_subset_ratio(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 設定エラー
        raise ValueError("training.val_subset_ratio は数値で指定してください") from exc
    if value <= 0.0:
        logger.warning("val_subset_ratio が 0 以下のため無効化します")
        return None
    if value >= 1.0:
        return 1.0
    return value


def _normalize_val_subset_seed(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 設定エラー
        raise ValueError("training.val_subset_seed は整数で指定してください") from exc


def _normalize_weight_decay(raw: Any) -> float:
    try:
        weight_decay = float(raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 設定エラー
        raise ValueError("training weight_decay は数値で指定してください") from exc
    if weight_decay < 0.0:
        raise ValueError("training weight_decay は 0 以上で指定してください")
    return weight_decay


def _normalize_clip_grad_norm(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        clip_grad_norm_val = float(raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 設定エラー
        raise ValueError(
            "training clip_grad_norm は数値で指定してください"
        ) from exc
    if clip_grad_norm_val <= 0.0:
        logger.warning(
            "clip_grad_norm が正の値ではないため、勾配クリッピングを無効化します"
        )
        return None
    return clip_grad_norm_val


def _normalize_label_smoothing(raw: Any) -> float:
    try:
        label_smoothing = float(raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 設定エラー
        raise ValueError(
            "training label_smoothing は数値で指定してください"
        ) from exc
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("training label_smoothing は 0 以上 1 未満で指定してください")
    return label_smoothing


def _normalize_optimizer(raw: Any) -> str:
    if raw is None:
        return "adam"
    name = str(raw).strip().lower()
    if name not in {"adam", "adamw", "rmsprop"}:
        raise ValueError(
            "training.optimizer は 'adam' / 'adamw' / 'rmsprop' のいずれかで指定してください"
        )
    return name


def _normalize_warmup_epochs(raw: Any) -> int:
    if raw is None:
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("training.warmup_epochs は整数で指定してください") from exc
    return max(0, value)


def _normalize_scheduler(raw: Any) -> str:
    if raw is None:
        return "cosine"
    name = str(raw).strip().lower()
    if not name:
        return "cosine"
    if name not in {"cosine", "none"}:
        raise ValueError("training.scheduler は 'cosine' または 'none' で指定してください")
    return name


def _normalize_warmup_start_factor(raw: Any) -> float:
    if raw is None:
        return 0.1
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("training.warmup_start_factor は数値で指定してください") from exc
    if value <= 0.0 or value > 1.0:
        raise ValueError("training.warmup_start_factor は 0 より大きく 1 以下で指定してください")
    return value


def build_training_hyperparams(
    run_type: str,
    training_root: dict[str, Any],
    train_y: np.ndarray,
) -> TrainingHyperParams:
    """Merge and normalize training hyper-parameters for a given run type."""

    common_training, specific_training = _resolve_training_sections(
        training_root,
        run_type,
    )
    tcfg = {**common_training, **specific_training}

    batch_size = tcfg["batch_size"]
    logger.info(f"batch_size: {batch_size}")

    epochs = tcfg["epochs"]
    logger.info(f"epochs: {epochs}")

    lr = tcfg["lr"]
    logger.info(f"lr: {lr}")

    optimizer_name = _normalize_optimizer(tcfg.get("optimizer"))
    logger.info(f"optimizer: {optimizer_name}")

    scheduler_name = _normalize_scheduler(tcfg.get("scheduler"))
    logger.info(f"scheduler: {scheduler_name}")

    warmup_epochs = _normalize_warmup_epochs(tcfg.get("warmup_epochs"))
    logger.info(f"warmup_epochs: {warmup_epochs}")

    warmup_start_factor = _normalize_warmup_start_factor(
        tcfg.get("warmup_start_factor")
    )
    logger.info(f"warmup_start_factor: {warmup_start_factor}")

    n_classes = int(train_y.max()) + 1
    logger.info(f"n_classes: {n_classes}")

    n_subset = tcfg["n_subset"]
    logger.info(f"n_subset: {n_subset}")

    eta_min = tcfg.get("eta_min", 1e-7)
    logger.info(f"eta_min: {eta_min}")

    val_interval = _normalize_val_interval(tcfg.get("val_interval", 1))
    logger.info(f"val_interval: {val_interval}")

    phase = _normalize_phase(tcfg.get("phase"))
    logger.info(f"phase: {phase}")

    val_max_batches = _normalize_val_max_batches(tcfg.get("val_max_batches"))
    logger.info(
        "val_max_batches: %s",
        "None" if val_max_batches is None else f"{val_max_batches}",
    )

    val_subset_ratio = _normalize_val_subset_ratio(tcfg.get("val_subset_ratio"))
    if phase == "search" and val_subset_ratio is None:
        val_subset_ratio = 0.4
    if val_subset_ratio is None:
        logger.info("val_subset_ratio: None")
    else:
        logger.info(f"val_subset_ratio: {val_subset_ratio}")

    val_subset_seed = _normalize_val_subset_seed(tcfg.get("val_subset_seed"))
    logger.info(
        "val_subset_seed: %s",
        "None" if val_subset_seed is None else f"{val_subset_seed}",
    )

    weight_decay = _normalize_weight_decay(tcfg.get("weight_decay", 0.0))
    logger.info(f"weight_decay: {weight_decay}")

    clip_grad_norm_val = _normalize_clip_grad_norm(tcfg.get("clip_grad_norm"))
    logger.info(
        "clip_grad_norm: %s",
        "None" if clip_grad_norm_val is None else f"{clip_grad_norm_val}",
    )

    label_smoothing = _normalize_label_smoothing(tcfg.get("label_smoothing", 0.0))
    logger.info(f"label_smoothing: {label_smoothing}")

    debug_cuda_sync = bool(tcfg.get("debug_cuda_sync", False))
    logger.info("debug_cuda_sync: %s", debug_cuda_sync)

    use_amp = bool(tcfg.get("amp", False))
    logger.info("amp: %s", use_amp)

    early_stopping_cfg = _parse_early_stopping_config(common_training)
    if early_stopping_cfg:
        logger.info(
            "early_stopping: metric=%s mode=%s patience=%d min_delta=%s restore_best=%s",
            early_stopping_cfg.metric,
            early_stopping_cfg.mode,
            early_stopping_cfg.patience,
            early_stopping_cfg.min_delta,
            early_stopping_cfg.restore_best,
        )

    return TrainingHyperParams(
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        eta_min=eta_min,
        optimizer=optimizer_name,
        scheduler=scheduler_name,
        warmup_epochs=warmup_epochs,
        warmup_start_factor=warmup_start_factor,
        val_interval=val_interval,
        phase=phase,
        val_max_batches=val_max_batches,
        val_subset_ratio=val_subset_ratio,
        val_subset_seed=val_subset_seed,
        n_subset=n_subset,
        n_classes=n_classes,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm_val,
        label_smoothing=label_smoothing,
        debug_cuda_sync=debug_cuda_sync,
        use_amp=use_amp,
        early_stopping=early_stopping_cfg,
    )


def build_model_config(
    run_type: str,
    model_root: dict[str, Any],
) -> dict[str, Any]:
    """Merge model.common overrides with run-type specific configuration."""

    if not isinstance(model_root, dict):
        raise TypeError("model 設定は辞書形式で指定してください")

    common_cfg = model_root.get("common") or {}
    if not isinstance(common_cfg, dict):
        raise TypeError("model.common は辞書形式で指定してください")

    specific_cfg = model_root.get(run_type)
    if specific_cfg is None:
        raise KeyError(f"model.{run_type} が config から見つかりません")
    if not isinstance(specific_cfg, dict):
        raise TypeError(f"model.{run_type} は辞書形式である必要があります")

    merged_cfg = {**common_cfg, **specific_cfg}

    for key in ("embed_dim", "hidden_dim", "dropout"):
        if key not in merged_cfg:
            raise KeyError(f"model 設定に '{key}' が不足しています")

    if run_type == "cnn":
        required = ("conv_channels", "kernel_size", "pool_size")
    elif run_type in {"lstm", "gru"}:
        required = ("num_layers", "bidirectional")
    elif run_type == "transformer":
        required = ("num_layers", "num_heads", "ffn_dim")
    else:
        raise KeyError(f"model.{run_type} 用の検証が未対応です")

    for key in required:
        if key not in merged_cfg:
            raise KeyError(f"model.{run_type} 設定に '{key}' が不足しています")

    return merged_cfg
