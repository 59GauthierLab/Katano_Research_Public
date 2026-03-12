#!/usr/bin/env python3
"""LR スイープ実行スクリプト。

config.yml を読み込み、指定した run_type ごとに複数の学習率で
短い学習ジョブを回し、結果を CSV にまとめる。
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
import logging
import subprocess
from copy import deepcopy
from pathlib import Path
import sys
import warnings
from typing import Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# scheduler.step(epoch=...) 警告を無視（挙動はチェインAPI使用で変化なし）
warnings.filterwarnings(
    "ignore",
    message="The epoch parameter in `scheduler.step\\(\\)`.*",
    category=UserWarning,
)

from config.training_config import build_model_config, build_training_hyperparams
from src.main import (
    apply_subset_if_needed,
    configure_logging,
    create_optimizer_and_scheduler,
    ensure_learning_curve_file,
    initialize_model,
    load_config,
    load_dataset_splits,
    resolve_device,
    resolve_label_map,
    save_label_map,
    set_seed,
    _resolve_model_dir_name,
    _unwrap_compiled,
    capture_interpretability,
    log_parameter_scale,
    save_checkpoint,
    load_checkpoint,
)
from src.training import (
    InterpretabilitySettings,
    eval_model,
    prepare_eval_subset,
    run_training_loop,
)
from src.utils.plot_lr_search import generate_lr_vs_acc_plot


def _run_post_lr_search_commands(run_type: str, sweep_root: Path) -> None:
    """LR 探索完了後に post-run コマンドを実行する。"""
    logger = logging.getLogger(__name__)
    github_base = "https://github.com/rayfiyo/fifty-nlp/tree/main"
    try:
        relative_dir = sweep_root.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        relative_dir = f"result/{run_type}/lr_search"
    cn_payload = f"{run_type} lr_search {github_base}/{relative_dir}"
    commands = [
        f'cn "{cn_payload}"',
        "git pull",
        "git add result/",
        f'git commit -m "add: {run_type} のlr_searchログ追加"',
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FiFTy の学習率を探索するスクリプト",
    )

    lr_group = parser.add_mutually_exclusive_group(required=False)
    lr_group.add_argument(
        "-l",
        "--learning-rates",
        dest="learning_rates",
        nargs="+",
        type=float,
        help="明示的に試す学習率を指定（空白区切り）",
    )
    lr_group.add_argument(
        "--logspace",
        nargs=3,
        metavar=("START", "END", "NUM"),
        type=float,
        help="幾何間隔で START→END まで NUM 個生成（例: 1e-4 1e-2 5）",
    )

    parser.add_argument(
        "--run-types",
        nargs="+",
        help="対象の run_type を指定（未指定なら config.yml の type を使用）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="各 LR で回すエポック数（未指定なら config.yml の値）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="バッチサイズを上書き（未指定なら config.yml の値）",
    )
    parser.add_argument(
        "--n-subset",
        type=int,
        default=None,
        help="学習データをこの件数にサブセット（未指定なら config.yml の値）",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="ウォームアップエポック数を上書き",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=None,
        help="検証を実施するエポック間隔を上書き",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="利用デバイスを指定（cpu / cuda / auto / gpu）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="乱数シード（未指定なら config.yml の experiment.seed）",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="lr_search",
        help="結果ディレクトリ名に付与するタグ",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="torch.compile を有効化（短時間検証ではデフォルト無効）",
    )
    parser.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_const",
        const=True,
        default=None,
        help="early_stopping.enable を強制的に有効化（未指定なら config.yml を優先）",
    )
    parser.add_argument(
        "--no-early-stopping",
        dest="early_stopping",
        action="store_const",
        const=False,
        help="early_stopping.enable を強制的に無効化（未指定なら config.yml を優先）",
    )

    return parser.parse_args()


def _collect_learning_rates(args: argparse.Namespace) -> list[float]:
    if args.learning_rates:
        return [float(lr) for lr in args.learning_rates]

    if args.logspace:
        start, end, num = args.logspace
        count = max(1, int(num))
        if start <= 0 or end <= 0:
            raise ValueError("logspace は正の値のみ指定してください")
        return list(np.geomspace(start, end, count))

    return [1e-4, 3e-4, 6e-4, 1e-3]


def _normalize_run_types(raw: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for name in raw:
        if name is None:
            continue
        for part in str(name).split(","):
            candidate = part.strip().lower()
            if candidate:
                normalized.append(candidate)
    return normalized


def _format_lr_dirname(lr: float) -> str:
    return f"lr_{lr:.3g}".replace("+", "").replace("-", "m")


def _build_training_params(
    *,
    base_config: dict,
    run_type: str,
    lr: float,
    train_y,
    epochs: int | None,
    batch_size: int | None,
    n_subset: int | None,
    warmup_epochs: int | None,
    val_interval: int | None,
    early_stopping: bool | None,
):
    training_root = deepcopy(base_config.get("training", {}))
    common_cfg = training_root.setdefault("common", {})

    if epochs is not None:
        common_cfg["epochs"] = epochs
    if batch_size is not None:
        common_cfg["batch_size"] = batch_size
    if n_subset is not None:
        common_cfg["n_subset"] = n_subset
    if warmup_epochs is not None:
        common_cfg["warmup_epochs"] = warmup_epochs
    if val_interval is not None:
        common_cfg["val_interval"] = val_interval
    if early_stopping is not None:
        early_cfg = common_cfg.get("early_stopping")
        if not isinstance(early_cfg, dict):
            early_cfg = {}
        early_cfg["enable"] = bool(early_stopping)
        common_cfg["early_stopping"] = early_cfg

    specific_cfg = training_root.get(run_type, {})
    specific_cfg["lr"] = lr
    training_root[run_type] = specific_cfg

    return build_training_hyperparams(run_type, training_root, train_y)


def _read_best_metrics(curve_path: Path) -> dict[str, float | int | None]:
    best_loss = math.inf
    best_loss_epoch: int | None = None
    best_acc = -math.inf
    best_acc_epoch: int | None = None
    best_macro = -math.inf
    best_macro_epoch: int | None = None
    last_lr: float | None = None

    if not curve_path.exists():
        return {
            "best_val_loss": None,
            "best_val_loss_epoch": None,
            "best_val_acc": None,
            "best_val_acc_epoch": None,
            "best_val_macro_f1": None,
            "best_val_macro_f1_epoch": None,
            "final_lr": None,
        }

    with curve_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                epoch_idx = int(row["epoch"])
            except (TypeError, ValueError):
                continue

            def _to_float(val: str) -> float:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return math.nan

            val_loss = _to_float(row.get("val_loss", ""))
            val_acc = _to_float(row.get("val_acc", ""))
            val_macro = _to_float(row.get("val_macro_f1", ""))
            lr_val = _to_float(row.get("lr", ""))
            last_lr = lr_val if math.isfinite(lr_val) else last_lr

            if math.isfinite(val_loss) and val_loss < best_loss:
                best_loss = val_loss
                best_loss_epoch = epoch_idx

            if math.isfinite(val_acc) and val_acc > best_acc:
                best_acc = val_acc
                best_acc_epoch = epoch_idx

            if math.isfinite(val_macro) and val_macro > best_macro:
                best_macro = val_macro
                best_macro_epoch = epoch_idx

    def _sanitize(value: float, sentinel: float, default: float | None) -> float | None:
        if value == sentinel or math.isnan(value):
            return default
        return value

    return {
        "best_val_loss": _sanitize(best_loss, math.inf, None),
        "best_val_loss_epoch": best_loss_epoch,
        "best_val_acc": _sanitize(best_acc, -math.inf, None),
        "best_val_acc_epoch": best_acc_epoch,
        "best_val_macro_f1": _sanitize(best_macro, -math.inf, None),
        "best_val_macro_f1_epoch": best_macro_epoch,
        "final_lr": last_lr,
    }


def _append_summary(summary_path: Path, row: dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()

    with summary_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "run_type",
                "lr",
                "run_dir",
                "best_val_loss",
                "best_val_loss_epoch",
                "best_val_acc",
                "best_val_acc_epoch",
                "best_val_macro_f1",
                "best_val_macro_f1_epoch",
                "val_acc_at_end",
                "val_macro_f1_at_end",
                "final_lr",
                "epochs",
                "batch_size",
                "n_subset",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = _parse_args()
    config = load_config()

    experiment_cfg = config.get("experiment", {})
    device_spec = args.device or experiment_cfg.get("device")
    device = resolve_device(device_spec)
    seed = int(args.seed or experiment_cfg.get("seed", 42))
    set_seed(seed)

    raw_run_types = args.run_types or config.get("type", ["cnn"])
    run_types = _normalize_run_types(raw_run_types)
    if not run_types:
        raise ValueError("run_type が指定されていません")

    lr_candidates = _collect_learning_rates(args)

    base_result_root = Path(experiment_cfg.get("result_dir", "result")).expanduser()
    if not base_result_root.is_absolute():
        base_result_root = (PROJECT_ROOT / base_result_root).resolve()

    datasets = load_dataset_splits(config["data"])
    n_classes = int(datasets.train_y.max()) + 1
    label_map, jpeg_label_id, label_map_path, scenario_id = resolve_label_map(
        config["data"],
        n_classes,
    )
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    summary_rows: list[dict[str, object]] = []

    for run_type in run_types:
        model_cfg = build_model_config(run_type, config.get("model", {}))
        logger = logging.getLogger(__name__)

        sweep_root = (
            base_result_root
            / _resolve_model_dir_name(run_type)
            / "lr_search"
            / f"{timestamp}_{args.tag}"
        )
        summary_path = sweep_root / "lr_search_summary.csv"

        for lr in lr_candidates:
            run_dir = sweep_root / _format_lr_dirname(lr)
            run_dir.mkdir(parents=True, exist_ok=True)

            configure_logging(run_dir)
            logger.info("Starting LR sweep run: run_type=%s lr=%.6g", run_type, lr)
            save_label_map(
                run_dir=run_dir,
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

            training_params = _build_training_params(
                base_config=config,
                run_type=run_type,
                lr=lr,
                train_y=datasets.train_y,
                epochs=args.epochs,
                batch_size=args.batch_size,
                n_subset=args.n_subset,
                warmup_epochs=args.warmup_epochs,
                val_interval=args.val_interval,
                early_stopping=args.early_stopping,
            )

            train_x, train_y = apply_subset_if_needed(
                datasets.train_x,
                datasets.train_y,
                training_params.n_subset,
                seed,
            )

            model = initialize_model(
                run_type,
                model_cfg,
                training_params.n_classes,
                device,
            )
            log_parameter_scale(model)
            if args.compile_model:
                model = torch.compile(model, mode="reduce-overhead")

            optimizer, scheduler = create_optimizer_and_scheduler(
                model,
                lr,
                training_params.weight_decay,
                training_params.epochs,
                training_params.eta_min,
                optimizer_name=training_params.optimizer,
                scheduler_name=training_params.scheduler,
                warmup_epochs=training_params.warmup_epochs,
                warmup_start_factor=training_params.warmup_start_factor,
            )

            learning_curve_path = ensure_learning_curve_file(run_dir)
            # 出力先の存在を明示的に保証する。
            learning_curve_path.parent.mkdir(parents=True, exist_ok=True)
            learning_curve_path.touch(exist_ok=True)

            if training_params.debug_cuda_sync and device.type == "cuda":
                if os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
                    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                logger.info("CUDA_LAUNCH_BLOCKING=1 (debug_cuda_sync enabled)")

            interpret_settings = InterpretabilitySettings(
                enabled=False,
                split="val",
                interval=training_params.val_interval,
                indices=[],
                source_x=None,
                source_y=None,
            )

            run_training_loop(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                resume_checkpoint=None,
                train_x=train_x,
                train_y=train_y,
                val_x=datasets.val_x,
                val_y=datasets.val_y,
                batch_size=training_params.batch_size,
                epochs=training_params.epochs,
                start_epoch=0,
                seed=seed,
                device=device,
                run_dir=run_dir,
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

            val_subset_seed = (
                seed
                if training_params.val_subset_seed is None
                else training_params.val_subset_seed
            )
            val_x_eval, val_y_eval, _, val_subset_ratio = prepare_eval_subset(
                val_x=datasets.val_x,
                val_y=datasets.val_y,
                subset_ratio=training_params.val_subset_ratio,
                subset_seed=val_subset_seed,
            )
            val_metrics = eval_model(
                model,
                val_x_eval,
                val_y_eval,
                training_params.batch_size,
                device,
                max_batches=training_params.val_max_batches,
                jpeg_label_id=jpeg_label_id,
            )
            if (
                training_params.val_subset_ratio is not None
                or training_params.val_max_batches is not None
            ):
                logger.info(
                    "LR search validation subset: samples=%d/%d ratio=%.3f",
                    val_metrics.samples,
                    len(datasets.val_y),
                    val_subset_ratio,
                )

            best = _read_best_metrics(learning_curve_path)

            try:
                run_dir_repr = str(run_dir.relative_to(PROJECT_ROOT))
            except ValueError:
                run_dir_repr = str(run_dir)

            summary_row = {
                "run_type": run_type,
                "lr": lr,
                "run_dir": run_dir_repr,
                "best_val_loss": best["best_val_loss"],
                "best_val_loss_epoch": best["best_val_loss_epoch"],
                "best_val_acc": best["best_val_acc"],
                "best_val_acc_epoch": best["best_val_acc_epoch"],
                "best_val_macro_f1": best["best_val_macro_f1"],
                "best_val_macro_f1_epoch": best["best_val_macro_f1_epoch"],
                "val_acc_at_end": val_metrics.acc,
                "val_macro_f1_at_end": val_metrics.macro_f1,
                "final_lr": best["final_lr"],
                "epochs": training_params.epochs,
                "batch_size": training_params.batch_size,
                "n_subset": training_params.n_subset,
            }

            summary_rows.append(summary_row)
            _append_summary(summary_path, summary_row)
            logger.info("Finished: lr=%.6g val_acc=%.4f", lr, val_metrics.acc)

        # run_type ごとの探索完了後に、サマリ CSV から LR-Accuracy グラフを生成する。
        try:
            out_path = generate_lr_vs_acc_plot(
                summary_path,
                output_dir=sweep_root,
                verbose=False,
            )
        except Exception:  # pragma: no cover - plotting is auxiliary
            logger.exception("Failed to generate LR search summary plot")
        else:
            logger.info("LR search plot saved: %s", out_path.name)

        _run_post_lr_search_commands(run_type, sweep_root)

    if summary_rows:
        print("LR sweep summary:")
        for row in summary_rows:
            print(
                f"- {row['run_type']} lr={row['lr']:.6g} "
                f"best_val_loss={row['best_val_loss']} "
                f"best_val_acc={row['best_val_acc']}"
            )


if __name__ == "__main__":
    main()
