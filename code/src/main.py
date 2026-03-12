"""FiFTy 派生モデルの学習・評価エントリポイントである。

出力先ディレクトリ構成:
result/
└── YYYY-MM-DD_HH-MM-SS_<tag>/
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover - script execution path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

# 機械学習関連
from src.models import (
    FiFTyGRUModel,
    FiFTyLSTMModel,
    FiFTyModel,
    FiFTyTransformerModel,
)
import numpy as np
import random
import torch
import torch.nn
import torch.nn.functional as F
import os
import json
import math
import hashlib
import threading

# 描画ユーティリティ
import matplotlib
import matplotlib.pyplot as plt

# ロギング設定
from logging import (
    getLogger,
    FileHandler,
    StreamHandler,
    Formatter,
    INFO,
)
from dataclasses import dataclass
from math import ceil
from torchinfo import summary
from torchview import draw_graph
import datetime as _dt
import csv
import subprocess
import yaml
from typing import Any, Callable, Sequence

from config.training_config import (
    EarlyStoppingConfig,
    TrainingHyperParams,
    build_model_config,
    build_training_hyperparams,
)
from src.training import (
    EvalEpochMetrics,
    InterpretabilitySettings,
    TrainEpochMetrics,
    eval_model,
    run_training_loop,
)
from src.utils.learning_curve_plots import generate_learning_curve_plots
from src.utils.confusion_reports import generate_reports


# 描画バックエンド設定
matplotlib.use("Agg")


# プロジェクトパス設定
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent


# チェックポイント関連ヘルパー
def _unwrap_compiled(model: torch.nn.Module) -> torch.nn.Module:
    """torch.compile されたモデルなら元の Module を返す（なければそのまま）"""
    return getattr(model, "_orig_mod", model)


def log_parameter_scale(model: torch.nn.Module) -> None:
    """モデルのパラメータ規模をログへ出力する。"""
    base_model = _unwrap_compiled(model)
    total_params = sum(param.numel() for param in base_model.parameters())
    trainable_params = sum(
        param.numel() for param in base_model.parameters() if param.requires_grad
    )
    logger.info(f"parameter_scale: {total_params}")
    logger.info(f"trainable_parameters: {trainable_params}")


def load_config() -> dict[str, Any]:
    """config.yml を読み込み、環境変数で上書き。"""

    default_path = PROJECT_ROOT / "config" / "config.yml"
    config_data = (
        yaml.safe_load(
            default_path.read_text(encoding="utf-8"),
        )
        or {}
    )

    env_base_dir = os.environ.get("FIFTY_DATA_BASE_DIR")
    if env_base_dir:
        config_data.setdefault("data", {})["base_dir"] = env_base_dir

    env_result_dir = os.environ.get("FIFTY_RESULT_DIR")
    if env_result_dir:
        config_data.setdefault("experiment", {})["result_dir"] = env_result_dir

    env_run_type = os.environ.get("FIFTY_RUN_TYPE")
    if env_run_type:
        config_data["type"] = env_run_type

    return config_data


config = load_config()


def _normalize_label_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _load_class_names_catalog(path: Path) -> dict[str, list[str]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"クラス定義 JSON の読み込みに失敗しました: {path}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"クラス定義 JSON の形式が不正です: {path}")

    catalog: dict[str, list[str]] = {}
    for key, value in raw.items():
        if not isinstance(value, list):
            raise ValueError(f"クラス定義が配列ではありません: {path} (key={key})")
        names: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"クラス名が文字列ではありません: {path} (key={key})")
            names.append(item)
        catalog[str(key)] = names

    if not catalog:
        raise ValueError(f"クラス定義が空です: {path}")

    return catalog


def _select_class_names(
    catalog: dict[str, list[str]],
    n_classes: int,
) -> tuple[str, list[str]]:
    matches = [
        (scenario_id, names)
        for scenario_id, names in catalog.items()
        if len(names) == n_classes
    ]
    if not matches:
        available = ", ".join(
            f"{scenario_id}={len(names)}" for scenario_id, names in catalog.items()
        )
        raise ValueError(
            "クラス定義の件数が一致しません。"
            f" n_classes={n_classes}, available={available}"
        )
    if len(matches) > 1:
        scenario_ids = [scenario_id for scenario_id, _ in matches]
        raise ValueError(f"複数のクラス定義が一致しました: {scenario_ids}")
    return matches[0]


def _resolve_jpeg_label_id(class_names: list[str]) -> int | None:
    jpeg_candidates = []
    for label_id, name in enumerate(class_names):
        normalized = _normalize_label_name(name)
        if normalized in {"jpg", "jpeg"}:
            jpeg_candidates.append(label_id)
    if not jpeg_candidates:
        return None
    if len(jpeg_candidates) > 1:
        raise ValueError(f"JPEG クラスが複数見つかりました: {jpeg_candidates}")
    return jpeg_candidates[0]


def resolve_label_map(
    data_cfg: dict[str, Any],
    n_classes: int,
) -> tuple[dict[int, str], int | None, Path, str]:
    base_dir = Path(data_cfg.get("base_dir", "")).expanduser()
    candidates = [
        base_dir / "classes_Human-readable_labels.json",
        PROJECT_ROOT / "dataset" / "classes_Human-readable_labels.json",
    ]
    class_path = next((path for path in candidates if path.exists()), None)
    if class_path is None:
        raise FileNotFoundError(
            "classes_Human-readable_labels.json が見つかりません。"
            " data.base_dir か dataset/ を確認してください。"
        )

    catalog = _load_class_names_catalog(class_path)
    scenario_id, class_names = _select_class_names(catalog, n_classes)
    label_map = {idx: name for idx, name in enumerate(class_names)}
    jpeg_label_id = _resolve_jpeg_label_id(class_names)
    if jpeg_label_id is None:
        logger.info(
            "JPEG クラスが見つからないため jpeg_acc は not_available として扱います (scenario=%s)",
            scenario_id,
        )
    return label_map, jpeg_label_id, class_path, scenario_id


def save_label_map(
    *,
    run_dir: Path,
    label_map: dict[int, str],
    jpeg_label_id: int | None,
    source_path: Path,
) -> None:
    label_map_json = json.dumps(label_map, sort_keys=True, ensure_ascii=True)
    label_map_sha256 = hashlib.sha256(label_map_json.encode("utf-8")).hexdigest()
    payload = {
        "source": str(source_path),
        "jpeg_label_id": jpeg_label_id,
        "label_map_sha256": label_map_sha256,
        "label_map": label_map,
    }
    out_path = run_dir / "label_map.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )


def _normalize_run_types(raw_run_type: Any) -> list[str]:
    """
    config.type が単一値でも配列でも受け取り、正規化済みリストを返す。
    文字列はカンマ区切りも許容し、小文字・トリムして空要素を除去する。
    """

    if raw_run_type is None:
        return ["cnn"]

    if isinstance(raw_run_type, (list, tuple)):
        candidates = list(raw_run_type)
    else:
        candidates = [raw_run_type]

    normalized: list[str] = []
    for item in candidates:
        if item is None:
            continue
        for part in str(item).split(","):
            name = part.strip().lower()
            if name:
                normalized.append(name)

    return normalized or ["cnn"]


raw_run_type = config.get("type", "cnn")
run_types = _normalize_run_types(raw_run_type)
config["type"] = raw_run_type

# ロギング設定
logger = getLogger(__name__)  # モジュール単位でロガーを取得する。
log_level = INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL


@dataclass
class RunDirectory:
    run_dir: Path
    checkpoint_path: Path
    resuming: bool


@dataclass
class ResumeState:
    data: dict[str, Any] | None
    start_epoch: int


@dataclass
class DatasetSplits:
    train_x: np.memmap
    train_y: np.memmap
    val_x: np.memmap
    val_y: np.memmap
    test_x: np.memmap
    test_y: np.memmap


def set_seed(seed=42):
    """
    乱数シードを固定し、実験の再現性を確保する。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_memmap(split: str) -> tuple[np.memmap, np.memmap]:
    """
    指定した split の `.npy` をメモリマップとして読み込み、メモリ使用量を抑制する。
    """
    base_dir = Path(config["data"]["base_dir"]).expanduser()  # `~` を展開する。
    base = base_dir / split

    # 遅延ロードで RAM 使用量を抑える。
    x = np.load(base / "x.npy", mmap_mode="r")
    y = np.load(base / "y.npy", mmap_mode="r")
    return x, y


def determine_default_tag(repo_dir: Path) -> str:
    """git のコミット ID からデフォルトタグを取得する。"""

    try:
        raw = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=repo_dir,
        )
    except Exception:
        return ""

    return raw.decode("utf-8").strip()


def _resolve_model_dir_name(run_type: str) -> str:
    """結果ディレクトリ名を run_type から決定する。"""
    if run_type == "transformer":
        return "Transformer"
    return run_type.upper()


def prepare_run_directory(
    experiment_cfg: dict[str, Any],
    default_tag: str,
    argv: Sequence[str],
    run_type: str,
) -> RunDirectory:
    """実行ディレクトリとチェックポイントパスを決定する。"""

    base_result_root = Path(experiment_cfg["result_dir"]).expanduser()
    if not base_result_root.is_absolute():
        base_result_root = (PROJECT_ROOT / base_result_root).resolve()

    model_dir_name = _resolve_model_dir_name(run_type)
    result_root = base_result_root / model_dir_name

    resume_target = experiment_cfg.get("resume_from")
    if resume_target:
        candidate = Path(resume_target).expanduser()
        if not candidate.is_absolute():
            candidate = base_result_root / resume_target
            if not candidate.exists():
                candidate = result_root / resume_target

        if candidate.is_file():
            return RunDirectory(
                run_dir=candidate.parent,
                checkpoint_path=candidate,
                resuming=True,
            )

        if candidate.is_dir():
            run_dir = candidate
            checkpoint_path = run_dir / "checkpoint_latest.pt"
            return RunDirectory(
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                resuming=True,
            )

        msg = f"resume_from が指すパスが存在しません: {candidate}"
        raise FileNotFoundError(msg)

    timestamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = argv[1] if len(argv) > 1 else default_tag
    run_dir = result_root / f"{timestamp}_{tag}"
    checkpoint_path = run_dir / "checkpoint_latest.pt"
    return RunDirectory(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        resuming=False,
    )


def load_resume_state(
    run_setup: RunDirectory,
    *,
    device: torch.device,
    run_type: str,
    seed: int,
) -> ResumeState:
    """再開時のチェックポイント情報を読み込む。"""

    if not run_setup.resuming:
        return ResumeState(data=None, start_epoch=0)

    checkpoint_data = torch.load(
        run_setup.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    start_epoch = int(checkpoint_data.get("epoch", 0) or 0)
    ckpt_run_type = checkpoint_data.get("run_type")
    if ckpt_run_type and ckpt_run_type != run_type:
        raise ValueError(
            f"チェックポイントの run_type ({ckpt_run_type}) と "
            f"現在の設定 ({run_type}) が一致しません"
        )

    ckpt_seed = checkpoint_data.get("seed")
    if ckpt_seed is not None and ckpt_seed != seed:
        logger.warning(
            f"チェックポイントのシード値 ({ckpt_seed}) が現在の設定 ({seed}) と異なります"
        )

    logger.info(
        f"Resuming from epoch {start_epoch}"
        + f"(checkpoint saved at {checkpoint_data.get('saved_at', 'n/a')})"
    )

    return ResumeState(data=checkpoint_data, start_epoch=start_epoch)


def load_dataset_splits(data_cfg: dict[str, Any]) -> DatasetSplits:
    """設定から訓練・検証・テスト用の memmap を読み込む。"""

    splits = data_cfg["splits"]
    train_x, train_y = load_memmap(splits[0])
    val_x, val_y = load_memmap(splits[1])
    test_x, test_y = load_memmap(splits[2])
    return DatasetSplits(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
    )


def configure_interpretability_settings(
    run_type: str,
    interpret_cfg: dict[str, Any] | None,
    datasets: DatasetSplits,
    default_interval: int,
) -> InterpretabilitySettings:
    """可視化設定を解釈し、利用可能性を判定する。"""

    interpret_cfg = interpret_cfg or {}
    interpret_enabled = bool(interpret_cfg.get("enable", False))
    interpret_split = interpret_cfg.get("split", "val")
    interpret_interval = max(1, int(interpret_cfg.get("interval", default_interval)))
    interpret_num_samples = max(1, int(interpret_cfg.get("num_samples", 1)))
    interpret_indices_cfg = interpret_cfg.get("indices")
    interpret_indices: list[int] = []
    interpret_source_x: np.ndarray | np.memmap | None = None
    interpret_source_y: np.ndarray | np.memmap | None = None

    if interpret_enabled:
        supported_run_types = {"cnn", "lstm", "gru"}
        if run_type not in supported_run_types:
            logger.warning(
                "Interpretability is enabled but not supported for run_type '%s'",
                run_type,
            )
            interpret_enabled = False
        else:
            split_map = {
                "train": (datasets.train_x, datasets.train_y),
                "val": (datasets.val_x, datasets.val_y),
                "test": (datasets.test_x, datasets.test_y),
            }
            if interpret_split not in split_map:
                logger.warning(
                    "Interpretability split '%s' is invalid; disabling",
                    interpret_split,
                )
                interpret_enabled = False
            else:
                interpret_source_x, interpret_source_y = split_map[interpret_split]
                available = len(interpret_source_y)
                if available == 0:
                    logger.warning(
                        "Interpretability requested but split '%s' has no samples",
                        interpret_split,
                    )
                    interpret_enabled = False
                else:
                    if interpret_indices_cfg:
                        indices_candidate: list[int] = []
                        for raw_idx in interpret_indices_cfg:
                            try:
                                idx_val = int(raw_idx)
                            except (TypeError, ValueError):
                                continue
                            if 0 <= idx_val < available:
                                indices_candidate.append(idx_val)
                        if not indices_candidate:
                            logger.warning(
                                "Interpretability indices are empty after filtering;"
                                + " disabling"
                            )
                            interpret_enabled = False
                        else:
                            interpret_indices = indices_candidate[
                                :interpret_num_samples
                            ]
                    else:
                        interpret_indices = list(
                            range(min(interpret_num_samples, available))
                        )

    if interpret_enabled:
        logger.info(
            "interpretability: split=%s, interval=%d, samples=%s",
            interpret_split,
            interpret_interval,
            interpret_indices,
        )
    else:
        interpret_source_x = None
        interpret_source_y = None

    return InterpretabilitySettings(
        enabled=interpret_enabled,
        split=interpret_split,
        interval=interpret_interval,
        indices=interpret_indices,
        source_x=interpret_source_x,
        source_y=interpret_source_y,
    )


def apply_subset_if_needed(
    train_x: np.ndarray | np.memmap,
    train_y: np.ndarray | np.memmap,
    n_subset: int,
    seed: int,
) -> tuple[np.ndarray | np.memmap, np.ndarray | np.memmap]:
    """サブセット指定があれば学習データを間引く。"""

    if n_subset and (n_subset < len(train_y)):
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(len(train_y), size=n_subset, replace=False)
        return train_x[idx], train_y[idx]

    return train_x, train_y


def initialize_model(
    run_type: str,
    model_cfg: dict[str, Any],
    n_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    """run_type に応じたモデルを構築する。"""

    if run_type == "gru":
        model = FiFTyGRUModel(
            n_classes=n_classes,
            embed_dim=model_cfg["embed_dim"],
            hidden=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            bidirectional=model_cfg["bidirectional"],
            dropout=model_cfg["dropout"],
            embedding_dropout=model_cfg.get("embedding_dropout", 0.0),
            layer_norm=model_cfg.get("layernorm", False),
        )
    elif run_type == "lstm":
        model = FiFTyLSTMModel(
            n_classes=n_classes,
            embed_dim=model_cfg["embed_dim"],
            hidden=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            bidirectional=model_cfg["bidirectional"],
            dropout=model_cfg["dropout"],
            embedding_dropout=model_cfg.get("embedding_dropout", 0.0),
            layer_norm=model_cfg.get("layernorm", False),
        )
    elif run_type == "transformer":
        model = FiFTyTransformerModel(
            n_classes=n_classes,
            embed_dim=model_cfg["embed_dim"],
            hidden=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            num_heads=model_cfg["num_heads"],
            ffn_dim=model_cfg["ffn_dim"],
            dropout=model_cfg["dropout"],
            embedding_dropout=model_cfg.get("embedding_dropout", 0.0),
            layer_norm=model_cfg.get("layernorm", False),
        )
    elif run_type == "cnn":
        model = FiFTyModel(
            n_classes=n_classes,
            embed_dim=model_cfg["embed_dim"],
            conv_channels=model_cfg["conv_channels"],
            hidden=model_cfg["hidden_dim"],
            kernel_size=model_cfg["kernel_size"],
            pool_size=model_cfg["pool_size"],
            dropout=model_cfg["dropout"],
            num_blocks=model_cfg.get("num_blocks", 2),
            embedding_dropout=model_cfg.get("embedding_dropout", 0.0),
            layer_norm=model_cfg.get("layernorm", False),
        )
    else:
        raise ValueError(f"Unsupported run_type: {run_type}")

    return model.to(device)


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    epochs: int,
    eta_min: float,
    *,
    optimizer_name: str,
    scheduler_name: str,
    warmup_epochs: int,
    warmup_start_factor: float,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Optimizer と Scheduler を構築する。"""

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-7,
            amsgrad=False,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-7,
            amsgrad=False,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=0.9,  # Keras rho
            eps=1e-7,  # Keras epsilon
            weight_decay=0.0,  # Keras には同等設定がないため 0.0 とする。
            momentum=0.0,
            centered=False,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    if scheduler_name == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda _: 1.0,
        )
        return optimizer, scheduler

    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    cosine_epochs = max(1, epochs - warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=eta_min,
    )

    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine

    return optimizer, scheduler


def ensure_learning_curve_file(run_dir: Path) -> Path:
    """学習曲線出力用 CSV の存在を保証する。"""

    learning_curve_path = run_dir / "learning_curve.csv"
    if not learning_curve_path.exists():
        with learning_curve_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "train_acc_top3",
                    "val_loss",
                    "val_acc",
                    "val_acc_top3",
                    "val_macro_f1",
                    "val_jpeg_acc",
                    "val_jpeg_support",
                    "lr",
                ]
            )

    return learning_curve_path


def configure_logging(run_dir: Path) -> None:
    """
    ルートロガーを指定し、標準出力とファイル出力に同じフォーマットで出力する（二重化）。
    一部を標準出力にし、他はデフォルトの標準エラーに出したい場合は、フィルタなどが必要。
    """

    # 1. ルートロガーにしきい値を設定する。
    root_logger = getLogger()
    root_logger.setLevel(log_level)

    # 2. 既存ハンドラを初期化する。
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 3. フォーマッタを定義する。
    fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = Formatter(fmt, datefmt=datefmt)

    # 4. ファイルハンドラ: `log.txt` に出力する。
    fh = FileHandler(run_dir / "log.txt", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    # 5. コンソールハンドラ: 標準出力に出力する。
    ch = StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)


def resolve_device(device_name: str | None) -> torch.device:
    """
    設定文字列から torch.device を解決する。

    - 空文字列や None は自動判定
    - "gpu" は "cuda" のエイリアス
    - CUDA が要求されたが利用できない場合は例外
    """

    if device_name is None:
        normalized = "auto"
    else:
        normalized = device_name.strip().lower()
        if normalized == "":
            normalized = "auto"

    if normalized == "gpu":
        normalized = "cuda"

    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        resolved = torch.device(normalized)
    except (
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:  # pragma: no cover - 設定エラー
        msg = f"Unsupported device specifier: {device_name!r}"
        raise ValueError(msg) from exc

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but not available. "
            "Enable a GPU runtime or set experiment.device to 'cpu'."
        )

    return resolved


def _normalize_task_type(task_type: str) -> str:
    """解析用タスク種別を正規化し、未対応なら例外を送出する。"""
    normalized = task_type.lower()
    if normalized not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task_type: {task_type!r}")
    return normalized


def _empty_analysis_result(task: str) -> dict[str, Any]:
    """サンプルが無い場合の解析結果テンプレートを返す。"""
    dtype = np.float32 if task == "regression" else np.int64
    return {
        "task": task,
        "sample_count": 0,
        "targets": np.empty(0, dtype=dtype),
        "predictions": np.empty(0, dtype=dtype),
        "true_probs_correct": np.empty(0, dtype=np.float32),
        "true_probs_incorrect": np.empty(0, dtype=np.float32),
    }


def _determine_sample_limit(max_samples: int | None) -> int | None:
    """最大サンプル数設定を整数へ正規化し、負値を防ぐ。"""
    if max_samples is None:
        return None
    return max(0, int(max_samples))


def _append_classification_batch(
    *,
    logits: torch.Tensor,
    targets_np: np.ndarray,
    device: torch.device,
    predictions_chunks: list[np.ndarray],
    targets_chunks: list[np.ndarray],
    true_probs_correct: list[float],
    true_probs_incorrect: list[float],
) -> None:
    """分類タスクの 1 バッチ分を集計し、配列へ蓄積する。"""
    targets_arr = np.asarray(targets_np, dtype=np.int64).copy()
    targets_tensor = torch.from_numpy(targets_arr).to(device)

    probs = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1)
    true_probs = probs.gather(1, targets_tensor.unsqueeze(1)).squeeze(1)

    pred_np = pred_idx.detach().cpu().numpy().astype(np.int64, copy=False)

    predictions_chunks.append(pred_np.copy())
    targets_chunks.append(targets_arr.copy())

    true_probs_np = true_probs.detach().cpu().numpy()
    correct_mask = pred_np == targets_arr
    if correct_mask.any():
        true_probs_correct.extend(true_probs_np[correct_mask].tolist())
    if (~correct_mask).any():
        true_probs_incorrect.extend(
            true_probs_np[np.logical_not(correct_mask)].tolist()
        )


def _append_regression_batch(
    *,
    logits: torch.Tensor,
    targets_np: np.ndarray,
    predictions_chunks: list[np.ndarray],
    targets_chunks: list[np.ndarray],
) -> None:
    """回帰タスクの 1 バッチ分を集計し、配列へ蓄積する。"""
    outputs = logits.detach().cpu().numpy()
    predictions = np.squeeze(outputs, axis=1) if outputs.ndim > 1 else outputs
    predictions = predictions.astype(np.float32, copy=False)
    targets_arr = np.asarray(targets_np, dtype=np.float32).copy()

    predictions_chunks.append(predictions.copy())
    targets_chunks.append(targets_arr.copy())


def collect_analysis_data(
    model: torch.nn.Module,
    x_memmap: np.memmap,
    y_memmap: np.memmap,
    batch_size: int,
    device: torch.device,
    *,
    task_type: str,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """追加解析用に予測結果を収集する。"""

    normalized_task = _normalize_task_type(task_type)

    total = len(y_memmap)
    if total == 0:
        return _empty_analysis_result(normalized_task)

    sample_limit = _determine_sample_limit(max_samples)
    if sample_limit == 0:
        return _empty_analysis_result(normalized_task)

    prev_training = model.training
    model.eval()

    targets_chunks: list[np.ndarray] = []
    predictions_chunks: list[np.ndarray] = []
    true_probs_correct: list[float] = []
    true_probs_incorrect: list[float] = []

    collected = 0
    total_batches = ceil(total / batch_size)

    try:
        with torch.no_grad():
            for batch_idx in range(total_batches):
                if sample_limit is not None and collected >= sample_limit:
                    break

                start = batch_idx * batch_size
                end = min(start + batch_size, total)

                if sample_limit is not None:
                    remaining = sample_limit - collected
                    if remaining <= 0:
                        break
                    end = min(end, start + remaining)

                if end <= start:
                    continue

                slab = x_memmap[start:end].astype(np.uint8)
                inputs = torch.from_numpy(slab.copy()).long().to(device)
                targets_np = np.asarray(y_memmap[start:end])

                logits = model(inputs)

                if normalized_task == "classification":
                    _append_classification_batch(
                        logits=logits,
                        targets_np=targets_np,
                        device=device,
                        predictions_chunks=predictions_chunks,
                        targets_chunks=targets_chunks,
                        true_probs_correct=true_probs_correct,
                        true_probs_incorrect=true_probs_incorrect,
                    )
                else:
                    _append_regression_batch(
                        logits=logits,
                        targets_np=targets_np,
                        predictions_chunks=predictions_chunks,
                        targets_chunks=targets_chunks,
                    )

                collected += end - start

    finally:
        model.train(prev_training)

    target_dtype = np.float32 if normalized_task == "regression" else np.int64

    targets = (
        np.concatenate(targets_chunks)
        if targets_chunks
        else np.empty(0, dtype=target_dtype)
    )
    predictions = (
        np.concatenate(predictions_chunks)
        if predictions_chunks
        else np.empty(0, dtype=target_dtype)
    )

    return {
        "task": normalized_task,
        "sample_count": collected,
        "targets": targets,
        "predictions": predictions,
        "true_probs_correct": np.asarray(true_probs_correct, dtype=np.float32),
        "true_probs_incorrect": np.asarray(true_probs_incorrect, dtype=np.float32),
    }


def save_regression_scatter(
    actual: np.ndarray,
    predicted: np.ndarray,
    out_path: Path,
    *,
    max_points: int = 5000,
    seed: int = 42,
) -> None:
    """実測値と予測値の散布図を保存する。"""

    if actual.size == 0 or predicted.size == 0:
        return

    actual_arr = np.asarray(actual, dtype=np.float64)
    predicted_arr = np.asarray(predicted, dtype=np.float64)

    if max_points > 0 and actual_arr.size > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(actual_arr.size, size=max_points, replace=False)
        actual_arr = actual_arr[indices]
        predicted_arr = predicted_arr[indices]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual_arr, predicted_arr, s=10, alpha=0.6, edgecolors="none")

    axis_min = float(min(actual_arr.min(), predicted_arr.min()))
    axis_max = float(max(actual_arr.max(), predicted_arr.max()))
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        linestyle="--",
        color="black",
        linewidth=1.0,
    )

    ax.set_xlabel("実測値")
    ax.set_ylabel("予測値")
    ax.set_title("実測値 vs 予測値")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_probability_histogram(
    correct_probs: np.ndarray,
    incorrect_probs: np.ndarray,
    out_path: Path,
) -> None:
    """正解・誤分類別の出力確率ヒストグラムを描画する。"""

    if correct_probs.size == 0 and incorrect_probs.size == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0.0, 1.0, 21)

    plotted = False
    if correct_probs.size > 0:
        ax.hist(
            correct_probs,
            bins=bins,
            alpha=0.6,
            label="正解サンプル",
            color="#1f77b4",
        )
        plotted = True

    if incorrect_probs.size > 0:
        ax.hist(
            incorrect_probs,
            bins=bins,
            alpha=0.6,
            label="誤分類サンプル",
            color="#ff7f0e",
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("正解クラスの予測確率")
    ax.set_ylabel("サンプル数")
    ax.set_title("出力確率の比較")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_checkpoint(
    *,
    run_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    seed: int,
    run_type: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """学習途中の状態を保存し、再開に備える。"""
    model_to_save = _unwrap_compiled(model)

    checkpoint = {
        "epoch": epoch + 1,  # 再開時に次に実行するエポック番号
        "seed": seed,
        "run_type": run_type,
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    if extra:
        checkpoint["extra"] = extra

    latest_path = run_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    epoch_path = run_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
    torch.save(checkpoint, epoch_path)


def load_checkpoint(
    *,
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    checkpoint_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """保存済みチェックポイントを読み込み、各モジュールへ復元する。"""

    checkpoint = (
        checkpoint_data
        if checkpoint_data is not None
        else torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )
    state = checkpoint["model_state"]

    # `_orig_mod.` 接頭辞付きキーは後方互換のため除去する。
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}

    # compile 済み・未コンパイルのいずれでも実体へロードする。
    target = _unwrap_compiled(model)
    target.load_state_dict(state, strict=True)

    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint


def save_model_visuals(
    model: torch.nn.Module,
    run_dir: Path,
    input_length: int,
    batch_size: int,
) -> None:
    """
    可視化ユーティリティ
    torchview・torchinfo の結果を run_dir に保存する。
    """
    # `torchview`: レイヤー構造を SVG で保存する。
    graph = draw_graph(
        model,
        input_size=(1, input_length),  # 描画用途のためバッチサイズ 1 とする。
        expand_nested=True,
        save_graph=False,
    )
    graph.visual_graph.render(
        filename=run_dir / "torchview", format="svg", cleanup=True
    )

    # `torchinfo`: 形状とパラメータ数をテキストで保存する。
    info = summary(
        model,
        input_size=(batch_size, input_length),
        col_names=("input_size", "output_size", "num_params"),
        verbose=0,
    )
    (run_dir / "torchinfo.txt").write_text(str(info), encoding="utf-8")


def _extract_cnn_feature_maps(
    cnn_model: FiFTyModel,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    CNN モデルから中間特徴マップを抽出して返す。

    戻り値は (logits, 特徴辞書)。特徴辞書のキーは blockXX / gap / fc1。
    """

    activations: dict[str, torch.Tensor] = {}
    x = inputs.long()
    x = cnn_model.embed(x)
    x = x.permute(0, 2, 1)

    for idx, conv in enumerate(cnn_model.blocks, start=1):
        x = conv(x)
        x = cnn_model.activation(x)
        activations[f"block{idx:02d}"] = x.detach().cpu()
        x = cnn_model.pool(x)

    x = x.permute(0, 2, 1)
    x = cnn_model.pre_gap_layer_norm(x)
    x = x.permute(0, 2, 1)
    x = cnn_model.gap(x).squeeze(-1)
    activations["gap"] = x.detach().cpu()
    x = cnn_model.head_dropout(x)
    x = cnn_model.fc1(x)
    x = cnn_model.activation(x)
    activations["fc1"] = x.detach().cpu()
    logits = cnn_model.fc2(x)
    return logits.detach().cpu(), activations


def _compute_lstm_gates(
    lstm_module: torch.nn.LSTM,
    layer_input: torch.Tensor,
) -> list[list[dict[str, torch.Tensor]]]:
    """LSTM の各層・方向ごとのゲート活性を計算する。"""

    num_layers = lstm_module.num_layers
    num_directions = 2 if lstm_module.bidirectional else 1
    hidden_size = lstm_module.hidden_size
    batch, seq_len, _ = layer_input.shape

    current_input = layer_input
    diagnostics: list[list[dict[str, torch.Tensor]]] = []

    for layer_idx in range(num_layers):
        layer_diagnostics: list[dict[str, torch.Tensor]] = []
        direction_outputs: list[torch.Tensor] = []

        for direction in range(num_directions):
            suffix = "" if direction == 0 else "_reverse"
            weight_ih = getattr(
                lstm_module,
                f"weight_ih_l{layer_idx}{suffix}",
            )
            weight_hh = getattr(
                lstm_module,
                f"weight_hh_l{layer_idx}{suffix}",
            )
            bias_ih = getattr(
                lstm_module,
                f"bias_ih_l{layer_idx}{suffix}",
            )
            bias_hh = getattr(
                lstm_module,
                f"bias_hh_l{layer_idx}{suffix}",
            )

            hx = torch.zeros(
                batch,
                hidden_size,
                device=current_input.device,
                dtype=current_input.dtype,
            )
            cx = torch.zeros_like(hx)

            hidden_seq = torch.zeros(
                batch,
                seq_len,
                hidden_size,
                device=current_input.device,
                dtype=current_input.dtype,
            )
            cell_seq = torch.zeros_like(hidden_seq)
            ingate_seq = torch.zeros_like(hidden_seq)
            forgetgate_seq = torch.zeros_like(hidden_seq)
            cellgate_seq = torch.zeros_like(hidden_seq)
            outgate_seq = torch.zeros_like(hidden_seq)

            time_range = (
                range(seq_len) if direction == 0 else range(seq_len - 1, -1, -1)
            )

            for t in time_range:
                x_t = current_input[:, t, :]
                gates = F.linear(x_t, weight_ih, bias_ih) + F.linear(
                    hx, weight_hh, bias_hh
                )
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=1)

                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = torch.tanh(cellgate)
                outgate = torch.sigmoid(outgate)

                cy = forgetgate * cx + ingate * cellgate
                hy = outgate * torch.tanh(cy)

                hidden_seq[:, t, :] = hy
                cell_seq[:, t, :] = cy
                ingate_seq[:, t, :] = ingate
                forgetgate_seq[:, t, :] = forgetgate
                cellgate_seq[:, t, :] = cellgate
                outgate_seq[:, t, :] = outgate

                hx = hy
                cx = cy

            direction_outputs.append(hidden_seq)
            layer_diagnostics.append(
                {
                    "hidden_sequence": hidden_seq.detach().clone(),
                    "cell_sequence": cell_seq.detach().clone(),
                    "ingate": ingate_seq.detach().clone(),
                    "forgetgate": forgetgate_seq.detach().clone(),
                    "cellgate": cellgate_seq.detach().clone(),
                    "outgate": outgate_seq.detach().clone(),
                    "final_hidden": hx.detach().clone(),
                    "final_cell": cx.detach().clone(),
                }
            )

        if num_directions == 1:
            current_input = direction_outputs[0]
        else:
            current_input = torch.cat(direction_outputs, dim=2)

        diagnostics.append(layer_diagnostics)

    return diagnostics


def _collect_lstm_diagnostics(
    lstm_model: FiFTyLSTMModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    """LSTM モデルの中間情報を収集する。"""

    embedded = lstm_model.embed(inputs.long())
    lstm_out, (h_n, c_n) = lstm_model.lstm(embedded)

    if lstm_model.lstm.bidirectional:
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
    else:
        h_last = h_n[-1]

    dropped = lstm_model.dropout(h_last)
    logits = lstm_model.fc(dropped)

    gate_info = _compute_lstm_gates(lstm_model.lstm, embedded)

    return {
        "embedded": embedded.detach().cpu(),
        "lstm_output": lstm_out.detach().cpu(),
        "h_n": h_n.detach().cpu(),
        "c_n": c_n.detach().cpu(),
        "pre_fc": dropped.detach().cpu(),
        "logits": logits.detach().cpu(),
        "gates": gate_info,
    }


def _compute_gru_gates(
    gru_module: torch.nn.GRU,
    layer_input: torch.Tensor,
) -> list[list[dict[str, torch.Tensor]]]:
    """GRU の各層・方向ごとのゲート活性を計算する。"""

    num_layers = gru_module.num_layers
    num_directions = 2 if gru_module.bidirectional else 1
    hidden_size = gru_module.hidden_size
    batch, seq_len, _ = layer_input.shape

    current_input = layer_input
    diagnostics: list[list[dict[str, torch.Tensor]]] = []

    for layer_idx in range(num_layers):
        layer_diagnostics: list[dict[str, torch.Tensor]] = []
        direction_outputs: list[torch.Tensor] = []

        for direction in range(num_directions):
            suffix = "" if direction == 0 else "_reverse"
            weight_ih = getattr(
                gru_module,
                f"weight_ih_l{layer_idx}{suffix}",
            )
            weight_hh = getattr(
                gru_module,
                f"weight_hh_l{layer_idx}{suffix}",
            )
            bias_ih = getattr(
                gru_module,
                f"bias_ih_l{layer_idx}{suffix}",
            )
            bias_hh = getattr(
                gru_module,
                f"bias_hh_l{layer_idx}{suffix}",
            )

            hx = torch.zeros(
                batch,
                hidden_size,
                device=current_input.device,
                dtype=current_input.dtype,
            )

            hidden_seq = torch.zeros(
                batch,
                seq_len,
                hidden_size,
                device=current_input.device,
                dtype=current_input.dtype,
            )
            reset_seq = torch.zeros_like(hidden_seq)
            update_seq = torch.zeros_like(hidden_seq)
            new_seq = torch.zeros_like(hidden_seq)

            time_range = (
                range(seq_len) if direction == 0 else range(seq_len - 1, -1, -1)
            )

            for t in time_range:
                x_t = current_input[:, t, :]
                gate_x = F.linear(x_t, weight_ih, bias_ih)
                gate_h = F.linear(hx, weight_hh, bias_hh)

                x_r, x_z, x_n = gate_x.chunk(3, dim=1)
                h_r, h_z, h_n = gate_h.chunk(3, dim=1)

                resetgate = torch.sigmoid(x_r + h_r)
                updategate = torch.sigmoid(x_z + h_z)
                newgate = torch.tanh(x_n + resetgate * h_n)

                hy = newgate + updategate * (hx - newgate)

                hidden_seq[:, t, :] = hy
                reset_seq[:, t, :] = resetgate
                update_seq[:, t, :] = updategate
                new_seq[:, t, :] = newgate

                hx = hy

            direction_outputs.append(hidden_seq)
            layer_diagnostics.append(
                {
                    "hidden_sequence": hidden_seq.detach().clone(),
                    "resetgate": reset_seq.detach().clone(),
                    "updategate": update_seq.detach().clone(),
                    "newgate": new_seq.detach().clone(),
                    "final_hidden": hx.detach().clone(),
                }
            )

        if num_directions == 1:
            current_input = direction_outputs[0]
        else:
            current_input = torch.cat(direction_outputs, dim=2)

        diagnostics.append(layer_diagnostics)

    return diagnostics


def _collect_gru_diagnostics(
    gru_model: FiFTyGRUModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    """GRU モデルの中間情報を収集する。"""

    embedded = gru_model.embed(inputs.long())
    gru_out, h_n = gru_model.gru(embedded)

    if gru_model.gru.bidirectional:
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
    else:
        h_last = h_n[-1]

    dropped = gru_model.dropout(h_last)
    logits = gru_model.fc(dropped)

    gate_info = _compute_gru_gates(gru_model.gru, embedded)

    return {
        "embedded": embedded.detach().cpu(),
        "gru_output": gru_out.detach().cpu(),
        "h_n": h_n.detach().cpu(),
        "pre_fc": dropped.detach().cpu(),
        "logits": logits.detach().cpu(),
        "gates": gate_info,
    }


def capture_cnn_interpretability(
    model: torch.nn.Module,
    device: torch.device,
    run_dir: Path,
    epoch: int,
    *,
    samples: Sequence[np.ndarray],
    labels: Sequence[int],
    indices: Sequence[int],
    split: str,
) -> None:
    """CNN モデルの中間特徴量をファイルに保存する。"""

    if not samples:
        return

    base_model = _unwrap_compiled(model)
    if not isinstance(base_model, FiFTyModel):  # 型安全性のための検査
        logger.warning(
            "Interpretability capture skipped: target model is not FiFTyModel"
        )
        return

    out_root = run_dir / "interpretability" / f"epoch_{epoch + 1:03d}"
    out_root.mkdir(parents=True, exist_ok=True)

    prev_training_compiled = model.training
    prev_training_base = base_model.training

    base_model.eval()
    model.eval()

    summary_records: list[dict[str, Any]] = []

    try:
        with torch.no_grad():
            for sample_idx, sample_arr, label in zip(indices, samples, labels):
                input_arr = np.asarray(sample_arr, dtype=np.uint8).copy()
                input_tensor = torch.from_numpy(input_arr).unsqueeze(0).to(device)

                logits, feature_maps = _extract_cnn_feature_maps(
                    base_model, input_tensor
                )

                sample_dir = out_root / f"sample_{sample_idx:05d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                for name, tensor in feature_maps.items():
                    np.save(
                        sample_dir / f"{name}.npy",
                        tensor.squeeze(0).numpy(),
                    )

                probs = F.softmax(logits, dim=1).squeeze(0)
                pred_label = int(torch.argmax(probs).item())
                metadata = {
                    "epoch": epoch + 1,
                    "split": split,
                    "sample_index": int(sample_idx),
                    "target_label": int(label),
                    "predicted_label": pred_label,
                    "probabilities": probs.tolist(),
                }
                (sample_dir / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                summary_records.append(
                    {
                        "sample_index": metadata["sample_index"],
                        "target_label": metadata["target_label"],
                        "predicted_label": metadata["predicted_label"],
                        "confidence": float(probs.max().item()),
                    }
                )

        summary = {
            "epoch": epoch + 1,
            "split": split,
            "samples": summary_records,
        }
        (out_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "Feature maps exported for %s samples (split=%s) at epoch %d",
            len(summary_records),
            split,
            epoch + 1,
        )

    except Exception as exc:  # 失敗しても学習を止めない
        logger.exception(
            "Interpretability capture failed at epoch %d: %s",
            epoch + 1,
            exc,
        )

    finally:
        base_model.train(prev_training_base)
        model.train(prev_training_compiled)


def capture_lstm_interpretability(
    model: torch.nn.Module,
    device: torch.device,
    run_dir: Path,
    epoch: int,
    *,
    samples: Sequence[np.ndarray],
    labels: Sequence[int],
    indices: Sequence[int],
    split: str,
) -> None:
    """LSTM モデルの中間情報を保存する。"""

    if not samples:
        return

    base_model = _unwrap_compiled(model)
    if not isinstance(base_model, FiFTyLSTMModel):
        logger.warning(
            "Interpretability capture skipped: target model is not FiFTyLSTMModel"
        )
        return

    out_root = run_dir / "interpretability" / f"epoch_{epoch + 1:03d}"
    out_root.mkdir(parents=True, exist_ok=True)

    prev_training_compiled = model.training
    prev_training_base = base_model.training

    base_model.eval()
    model.eval()

    summary_records: list[dict[str, Any]] = []

    try:
        with torch.no_grad():
            for sample_idx, sample_arr, label in zip(indices, samples, labels):
                input_arr = np.asarray(sample_arr, dtype=np.uint8).copy()
                input_tensor = torch.from_numpy(input_arr).unsqueeze(0).to(device)

                diagnostics = _collect_lstm_diagnostics(base_model, input_tensor)
                logits = diagnostics["logits"]
                probs = F.softmax(logits, dim=1).squeeze(0)
                pred_label = int(torch.argmax(probs).item())

                sample_dir = out_root / f"sample_{sample_idx:05d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                np.save(sample_dir / "embedded.npy", diagnostics["embedded"].numpy())
                np.save(
                    sample_dir / "lstm_output.npy", diagnostics["lstm_output"].numpy()
                )
                np.save(sample_dir / "h_n.npy", diagnostics["h_n"].numpy())
                np.save(sample_dir / "c_n.npy", diagnostics["c_n"].numpy())
                np.save(sample_dir / "pre_fc.npy", diagnostics["pre_fc"].numpy())
                np.save(sample_dir / "logits.npy", logits.numpy())

                gate_info = diagnostics["gates"]
                for layer_idx, layer_data in enumerate(gate_info):
                    for direction_idx, direction_data in enumerate(layer_data):
                        direction_name = "forward" if direction_idx == 0 else "reverse"
                        np.savez(
                            sample_dir
                            / f"lstm_layer_{layer_idx:02d}_{direction_name}.npz",
                            hidden_sequence=direction_data["hidden_sequence"]
                            .cpu()
                            .numpy(),
                            cell_sequence=direction_data["cell_sequence"].cpu().numpy(),
                            ingate=direction_data["ingate"].cpu().numpy(),
                            forgetgate=direction_data["forgetgate"].cpu().numpy(),
                            cellgate=direction_data["cellgate"].cpu().numpy(),
                            outgate=direction_data["outgate"].cpu().numpy(),
                            final_hidden=direction_data["final_hidden"].cpu().numpy(),
                            final_cell=direction_data["final_cell"].cpu().numpy(),
                        )

                metadata = {
                    "epoch": epoch + 1,
                    "split": split,
                    "sample_index": int(sample_idx),
                    "target_label": int(label),
                    "predicted_label": pred_label,
                    "probabilities": probs.tolist(),
                }
                (sample_dir / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                summary_records.append(
                    {
                        "sample_index": metadata["sample_index"],
                        "target_label": metadata["target_label"],
                        "predicted_label": metadata["predicted_label"],
                        "confidence": float(probs.max().item()),
                    }
                )

        summary = {
            "epoch": epoch + 1,
            "split": split,
            "samples": summary_records,
        }
        (out_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "LSTM feature exports completed for %s samples (split=%s) at epoch %d",
            len(summary_records),
            split,
            epoch + 1,
        )

    except Exception as exc:
        logger.exception(
            "LSTM interpretability capture failed at epoch %d: %s",
            epoch + 1,
            exc,
        )

    finally:
        base_model.train(prev_training_base)
        model.train(prev_training_compiled)


def capture_gru_interpretability(
    model: torch.nn.Module,
    device: torch.device,
    run_dir: Path,
    epoch: int,
    *,
    samples: Sequence[np.ndarray],
    labels: Sequence[int],
    indices: Sequence[int],
    split: str,
) -> None:
    """GRU モデルの中間情報を保存する。"""

    if not samples:
        return

    base_model = _unwrap_compiled(model)
    if not isinstance(base_model, FiFTyGRUModel):
        logger.warning(
            "Interpretability capture skipped: target model is not FiFTyGRUModel"
        )
        return

    out_root = run_dir / "interpretability" / f"epoch_{epoch + 1:03d}"
    out_root.mkdir(parents=True, exist_ok=True)

    prev_training_compiled = model.training
    prev_training_base = base_model.training

    base_model.eval()
    model.eval()

    summary_records: list[dict[str, Any]] = []

    try:
        with torch.no_grad():
            for sample_idx, sample_arr, label in zip(indices, samples, labels):
                input_arr = np.asarray(sample_arr, dtype=np.uint8).copy()
                input_tensor = torch.from_numpy(input_arr).unsqueeze(0).to(device)

                diagnostics = _collect_gru_diagnostics(base_model, input_tensor)
                logits = diagnostics["logits"]
                probs = F.softmax(logits, dim=1).squeeze(0)
                pred_label = int(torch.argmax(probs).item())

                sample_dir = out_root / f"sample_{sample_idx:05d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                np.save(sample_dir / "embedded.npy", diagnostics["embedded"].numpy())
                np.save(
                    sample_dir / "gru_output.npy", diagnostics["gru_output"].numpy()
                )
                np.save(sample_dir / "h_n.npy", diagnostics["h_n"].numpy())
                np.save(sample_dir / "pre_fc.npy", diagnostics["pre_fc"].numpy())
                np.save(sample_dir / "logits.npy", logits.numpy())

                gate_info = diagnostics["gates"]
                for layer_idx, layer_data in enumerate(gate_info):
                    for direction_idx, direction_data in enumerate(layer_data):
                        direction_name = "forward" if direction_idx == 0 else "reverse"
                        np.savez(
                            sample_dir
                            / f"gru_layer_{layer_idx:02d}_{direction_name}.npz",
                            hidden_sequence=direction_data["hidden_sequence"]
                            .cpu()
                            .numpy(),
                            resetgate=direction_data["resetgate"].cpu().numpy(),
                            updategate=direction_data["updategate"].cpu().numpy(),
                            newgate=direction_data["newgate"].cpu().numpy(),
                            final_hidden=direction_data["final_hidden"].cpu().numpy(),
                        )

                metadata = {
                    "epoch": epoch + 1,
                    "split": split,
                    "sample_index": int(sample_idx),
                    "target_label": int(label),
                    "predicted_label": pred_label,
                    "probabilities": probs.tolist(),
                }
                (sample_dir / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                summary_records.append(
                    {
                        "sample_index": metadata["sample_index"],
                        "target_label": metadata["target_label"],
                        "predicted_label": metadata["predicted_label"],
                        "confidence": float(probs.max().item()),
                    }
                )

        summary = {
            "epoch": epoch + 1,
            "split": split,
            "samples": summary_records,
        }
        (out_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "GRU feature exports completed for %s samples (split=%s) at epoch %d",
            len(summary_records),
            split,
            epoch + 1,
        )

    except Exception as exc:
        logger.exception(
            "GRU interpretability capture failed at epoch %d: %s",
            epoch + 1,
            exc,
        )

    finally:
        base_model.train(prev_training_base)
        model.train(prev_training_compiled)


def capture_interpretability(
    model: torch.nn.Module,
    device: torch.device,
    run_dir: Path,
    epoch: int,
    *,
    samples: Sequence[np.ndarray],
    labels: Sequence[int],
    indices: Sequence[int],
    split: str,
) -> None:
    """モデル種別に応じて適切な可視化を実行する。"""

    base_model = _unwrap_compiled(model)

    if isinstance(base_model, FiFTyModel):
        capture_cnn_interpretability(
            model,
            device,
            run_dir,
            epoch,
            samples=samples,
            labels=labels,
            indices=indices,
            split=split,
        )
    elif isinstance(base_model, FiFTyLSTMModel):
        capture_lstm_interpretability(
            model,
            device,
            run_dir,
            epoch,
            samples=samples,
            labels=labels,
            indices=indices,
            split=split,
        )
    elif isinstance(base_model, FiFTyGRUModel):
        capture_gru_interpretability(
            model,
            device,
            run_dir,
            epoch,
            samples=samples,
            labels=labels,
            indices=indices,
            split=split,
        )
    else:
        logger.warning(
            "Interpretability capture skipped: unsupported model type %s",
            type(base_model).__name__,
        )


def run_single_experiment(run_type: str, device: torch.device) -> Path:
    """
    単一の run_type について学習・評価を実行する。
    """
    experiment_cfg = config.get("experiment", {})
    seed = experiment_cfg["seed"]
    set_seed(seed)

    default_tag = determine_default_tag(PROJECT_ROOT)
    run_setup = prepare_run_directory(experiment_cfg, default_tag, sys.argv, run_type)
    run_setup.run_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(run_setup.run_dir)
    logger.info(f"config.type: {run_type}")
    logger.info(f"\ndevice: {device.type} ({device})")
    logger.info(f"run_dir: {run_setup.run_dir}")
    if run_setup.resuming:
        logger.info(f"resume_checkpoint: {run_setup.checkpoint_path}")
        if not run_setup.checkpoint_path.exists():
            raise FileNotFoundError(
                "resume_from が指定されていますが、チェックポイントが見つかりません"
            )

    resume_state = load_resume_state(
        run_setup,
        device=device,
        run_type=run_type,
        seed=seed,
    )

    datasets = load_dataset_splits(config["data"])
    model_cfg = build_model_config(run_type, config.get("model", {}))
    training_params = build_training_hyperparams(
        run_type,
        config.get("training", {}),
        datasets.train_y,
    )
    label_map, jpeg_label_id, label_map_path, scenario_id = resolve_label_map(
        config["data"],
        training_params.n_classes,
    )
    save_label_map(
        run_dir=run_setup.run_dir,
        label_map=label_map,
        jpeg_label_id=jpeg_label_id,
        source_path=label_map_path,
    )
    jpeg_label_display = "n/a" if jpeg_label_id is None else f"{jpeg_label_id}"
    logger.info(
        "jpeg_label_id: %s (scenario=%s, source=%s)",
        jpeg_label_display,
        scenario_id,
        label_map_path,
    )
    interpret_settings = configure_interpretability_settings(
        run_type,
        experiment_cfg.get("interpretability", {}),
        datasets,
        training_params.val_interval,
    )

    logger.info(f"n_subset: {training_params.n_subset}")
    train_x, train_y = apply_subset_if_needed(
        datasets.train_x,
        datasets.train_y,
        training_params.n_subset,
        seed,
    )

    model = initialize_model(run_type, model_cfg, training_params.n_classes, device)
    log_parameter_scale(model)

    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        training_params.lr,
        training_params.weight_decay,
        training_params.epochs,
        training_params.eta_min,
        optimizer_name=training_params.optimizer,
        scheduler_name=training_params.scheduler,
        warmup_epochs=training_params.warmup_epochs,
        warmup_start_factor=training_params.warmup_start_factor,
    )

    if run_setup.resuming and resume_state.data is not None:
        load_checkpoint(
            checkpoint_path=run_setup.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_data=resume_state.data,
        )

    if not run_setup.resuming:
        save_model_visuals(
            model,
            run_setup.run_dir,
            input_length=train_x.shape[1],
            batch_size=training_params.batch_size,
        )

    # 現行設定では RNN 系 (GRU/LSTM) に対して `torch.compile` を無効化する。
    do_compile = config.get("experiment", {}).get("torch_compile", True)
    if do_compile and run_type not in {"gru", "lstm"}:
        model = torch.compile(model, mode="reduce-overhead")

    learning_curve_path = ensure_learning_curve_file(run_setup.run_dir)

    if training_params.debug_cuda_sync and device.type == "cuda":
        if os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        logger.info("CUDA_LAUNCH_BLOCKING=1 (debug_cuda_sync enabled)")

    retry_limit = 3
    retry_count = 0
    resume_checkpoint = resume_state.data
    start_epoch = resume_state.start_epoch

    while True:
        try:
            _run_in_new_thread(
                lambda: run_training_loop(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    resume_checkpoint=resume_checkpoint,
                    train_x=train_x,
                    train_y=train_y,
                    val_x=datasets.val_x,
                    val_y=datasets.val_y,
                    batch_size=training_params.batch_size,
                    epochs=training_params.epochs,
                    start_epoch=start_epoch,
                    seed=seed,
                    device=device,
                    run_dir=run_setup.run_dir,
                    run_type=run_type,
                    val_interval=training_params.val_interval,
                    val_max_batches=training_params.val_max_batches,
                    val_subset_ratio=training_params.val_subset_ratio,
                    val_subset_seed=training_params.val_subset_seed,
                    jpeg_label_id=jpeg_label_id,
                    clip_grad_norm=training_params.clip_grad_norm,
                    label_smoothing=training_params.label_smoothing,
                    debug_cuda_sync=training_params.debug_cuda_sync,
                    use_amp=training_params.use_amp,
                    early_stopping=training_params.early_stopping,
                    interpret=interpret_settings,
                    learning_curve_path=learning_curve_path,
                    save_checkpoint_fn=save_checkpoint,
                    load_checkpoint_fn=load_checkpoint,
                    capture_interpretability_fn=capture_interpretability,
                    unwrap_compiled_fn=_unwrap_compiled,
                )
            )
            break
        except RuntimeError as err:
            if _is_cuda_misaligned_address(err) and retry_count < retry_limit:
                retry_count += 1
                logger.warning(
                    "CUDA misaligned address detected. "
                    "Retrying from latest checkpoint (%d/%d).",
                    retry_count,
                    retry_limit,
                )
                latest_path = run_setup.run_dir / "checkpoint_latest.pt"
                if not latest_path.exists():
                    logger.error(
                        "最新のチェックポイントが見つかりません: %s", latest_path
                    )
                    raise
                resume_checkpoint = load_checkpoint(
                    checkpoint_path=latest_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                )
                start_epoch = int(resume_checkpoint.get("epoch", 0) or 0)
                continue
            raise

    test_metrics = eval_model(
        model,
        datasets.test_x,
        datasets.test_y,
        training_params.batch_size,
        device,
        jpeg_label_id=jpeg_label_id,
        confusion_output_path=run_setup.run_dir / "confusion_matrix.npy",
    )
    generate_reports(
        confusion_path=run_setup.run_dir / "confusion_matrix.npy",
        label_map_path=run_setup.run_dir / "label_map.json",
        output_dir=run_setup.run_dir,
    )
    macro_f1_log = (
        f", macro_f1={test_metrics.macro_f1:.3f}"
        if test_metrics.macro_f1 is not None
        else ""
    )
    jpeg_log = ""
    if test_metrics.jpeg_support is not None:
        jpeg_display = (
            f"{test_metrics.jpeg_acc:.3f}"
            if test_metrics.jpeg_acc is not None
            else "n/a"
        )
        jpeg_log = f", jpeg_acc={jpeg_display} (support={test_metrics.jpeg_support})"
    logger.info(
        f"Full test accuracy: {test_metrics.acc:.3f} "
        f"(batches={test_metrics.batches}, samples={test_metrics.samples}, "
        f"loss={test_metrics.loss:.4f}, top3={test_metrics.acc_top3:.3f}"
        + macro_f1_log
        + jpeg_log
        + ")"
    )

    try:
        generated = generate_learning_curve_plots(
            learning_curve_path,
            output_dir=run_setup.run_dir,
            verbose=False,
        )
    except Exception:  # pragma: no cover - plotting is auxiliary
        logger.exception("Failed to generate learning curve plots")
    else:
        if generated:
            logger.info(
                "Learning curve plots saved: %s",
                ", ".join(path.name for path in generated),
            )
    return run_setup.run_dir


def _run_summary_results_script() -> None:
    """終了前に結果要約スクリプトを 1 度だけ走らせる。"""
    script_path = PROJECT_ROOT / "src" / "utils" / "summarize_results.py"
    if not script_path.exists():
        logger.warning("summarize_results.py が見つかりません: %s", script_path)
        return

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        logger.exception("summarize_results.py の実行に失敗しました")
        return

    if completed.stdout:
        logger.info("summarize_results.py stdout:\n%s", completed.stdout.strip())
    if completed.stderr:
        logger.warning("summarize_results.py stderr:\n%s", completed.stderr.strip())
    if completed.returncode != 0:
        logger.error(
            "summarize_results.py が非ゼロ終了コード (%d) を返しました",
            completed.returncode,
        )


def _run_post_model_commands(model_name: str, run_dir: Path) -> None:
    """モデル学習後の post-run コマンドを実行する。"""
    github_base = "https://github.com/rayfiyo/fifty-nlp/tree/main"
    try:
        relative_run_dir = run_dir.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        relative_run_dir = f"result/{model_name}"
    cn_payload = f'{model_name} {github_base}/{relative_run_dir}'
    commands = [
        f'cn "{cn_payload}"',
        "git pull",
        "git add result/",
        f'git commit -m "add: {model_name} のログ追加"',
        "git push",
    ]

    for command in commands:
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as err:
            logger.error(
                "Post-run command failed (%s) exit=%s",
                command,
                err.returncode,
            )
            if err.stdout:
                logger.error("Post-run stdout (%s): %s", command, err.stdout.strip())
            if err.stderr:
                logger.error("Post-run stderr (%s): %s", command, err.stderr.strip())
            continue
        except Exception:
            logger.exception("Post-run command error (%s)", command)
            continue

        if completed.stdout:
            logger.info("Post-run stdout (%s): %s", command, completed.stdout.strip())
        if completed.stderr:
            logger.info("Post-run stderr (%s): %s", command, completed.stderr.strip())


def _is_cuda_error(err: BaseException) -> bool:
    keywords = ("CUDA error", "CUDA out of memory", "cuDNN", "CUDNN", "cuda")
    current: BaseException | None = err
    while current is not None:
        message = str(current)
        if any(keyword in message for keyword in keywords):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_cuda_misaligned_address(err: BaseException) -> bool:
    keywords = ("CUDA error: misaligned address", "misaligned address")
    current: BaseException | None = err
    while current is not None:
        message = str(current)
        if any(keyword in message for keyword in keywords):
            return True
        current = current.__cause__ or current.__context__
    return False


def _run_in_new_thread(task: Callable[[], None]) -> None:
    """別スレッドでタスクを実行し、例外を呼び出し元へ伝搬する。"""
    errors: list[BaseException] = []

    def _runner() -> None:
        try:
            task()
        except BaseException as exc:  # pragma: no cover - 例外を伝搬するため
            errors.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]


def main(device: torch.device | str = "cpu") -> None:
    """
    メイン関数:
    - config.type が配列なら先頭から順に実行
    - データ読み込み、モデル初期化、学習・評価を run_type ごとに実行
    """
    if not isinstance(device, torch.device):
        device = resolve_device(str(device))

    logger.info("run_types: %s", run_types)
    for run_type in run_types:
        logger.info("Starting run_type=%s", run_type)
        model_name = _resolve_model_dir_name(run_type)
        run_dir = None
        try:
            run_dir = run_single_experiment(run_type, device)
        except Exception as err:
            logger.exception("run_type=%s の学習でエラーが発生しました", run_type)
            if _is_cuda_error(err):
                logger.error(
                    "CUDA エラーのためプロセスを終了します: run_type=%s", run_type
                )
                raise
        if run_dir is None:
            run_dir = PROJECT_ROOT / "result" / model_name
        _run_post_model_commands(model_name, run_dir)


if __name__ == "__main__":
    device_spec = config.get("experiment", {}).get("device")
    exit_code = 0
    try:
        resolved_device = resolve_device(device_spec)
    except (Exception, MemoryError) as err:
        logger.error(f"Device resolution failed: {err}", exc_info=True)
        exit_code = 1
    else:
        try:
            main(device=resolved_device)
        except (Exception, MemoryError) as err:
            # MemoryError や CUDA OOM を含む致命的例外をここで処理する。
            logger.error(f"Fatal error occurred: {err}", exc_info=True)
            exit_code = 1
    finally:
        _run_summary_results_script()

    sys.exit(exit_code)
