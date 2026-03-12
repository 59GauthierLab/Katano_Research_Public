#!/usr/bin/env python3

"""Summarize the latest experiment results across models.

CSV 出力ヘッダーの意味:
- Model: 実験で使用したモデル名。
- RunDir: 結果が保存された実行ディレクトリの相対パス。
- Timestamp: 実験タイムスタンプ。
- BestEpoch: 検証指標が最良となったエポック番号。
- Parameters: 主要ハイパーパラメータの一覧文字列。
- ValTop1: 検証セットで記録した最高 Top-1 精度。
- ValTop3: 検証セットで記録した最高 Top-3 精度。
- ValLoss: 検証セットの損失値。
- ValMacroF1: 検証セットのマクロ平均 F1 スコア。
- TestTop1: テストセットの Top-1 精度 (未評価時は空欄)。
- TestTop3: テストセットの Top-3 精度 (未評価時は空欄)。
- TestLoss: テストセットの損失値 (未評価時は空欄)。
- TestMacroF1: テストセットのマクロ平均 F1 スコア (未評価時は空欄)。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_PARAMS = (
    "batch_size",
    "epochs",
    "lr",
    "weight_decay",
    "label_smoothing",
    "clip_grad_norm",
    "eta_min",
    "val_interval",
    "n_subset",
)

MODEL_ORDER = ("CNN", "LSTM", "GRU", "TRANSFORMER")

HP_PATTERN = re.compile(
    r"(?:config\.training_config|__main__) INFO: (?P<name>[a-z_]+): (?P<value>.+)",
)
TEST_ACC_PATTERN = re.compile(r"Full test accuracy: (?P<acc>[-0-9.eE+]+)")
TEST_LOSS_PATTERN = re.compile(r"loss=(?P<loss>[-0-9.eE+]+)")
TEST_TOP3_PATTERN = re.compile(r"top3=(?P<top3>[-0-9.eE+]+)")
TEST_MACRO_PATTERN = re.compile(r"macro_f1=(?P<macro>[-0-9.eE+]+)")
TRAIN_PATTERN = re.compile(
    r"Epoch (?P<epoch>\d+): train_loss=(?P<train_loss>[-0-9.eE+]+), "
    r"train_acc=(?P<train_acc>[-0-9.eE+]+), "
    r"train_acc_top3=(?P<train_acc_top3>[-0-9.eE+]+), lr=(?P<lr>[-0-9.eE+]+)",
)
VAL_PATTERN = re.compile(
    r"Epoch (?P<epoch>\d+): Validation: .*? val_loss=(?P<val_loss>[-0-9.eE+]+), "
    r"val_acc=(?P<val_acc>[-0-9.eE+]+), acc_top3=(?P<val_acc_top3>[-0-9.eE+]+)"
    r"(?:, macro_f1=(?P<val_macro_f1>[-0-9.eE+]+))?"
    r"(?:, jpeg_acc=(?P<val_jpeg_acc>[-0-9.eE+]+) \\(support=(?P<val_jpeg_support>\\d+)\\))?",
)
VAL_LEGACY_PATTERN = re.compile(
    r"Epoch (?P<epoch>\d+): Validation: .* acc=(?P<val_acc>[-0-9.eE+]+)"
)


@dataclass
class RunRecord:
    model: str
    run_dir: Path
    timestamp: str | None
    params: dict[str, float | str]
    best_epoch: int | None
    val_metrics: dict[str, float | None]
    test_metrics: dict[str, float | None]


@dataclass(frozen=True)
class MetricLine:
    column: str
    label: str
    linestyle: str
    alpha: float = 1.0


@dataclass(frozen=True)
class MetricPlotConfig:
    ylabel: str
    lines: tuple[MetricLine, ...]
    ylim_mode: str = "auto"
    legend_loc: str = "lower right"
    legend_ncol: int = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and visualize latest experiment results.",
    )
    parser.add_argument(
        "--result-root",
        help="結果ディレクトリのルート (既定: <project>/result)",
    )
    parser.add_argument(
        "--output-dir",
        help="summary_results.csv と summary_learning_curve_*.png の出力先",
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        metavar="SPEC",
        help="Plots use specified learning_curve.csv files instead of latest runs. "
        "Each SPEC can be PATH or LABEL=PATH.",
    )
    parser.add_argument(
        "--mono-accuracy",
        action="store_true",
        help="白黒印刷向けの summary_learning_curve_accuracy_mono.png も出力します。",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _result_root(result_root: str | None) -> Path:
    return (
        Path(result_root).expanduser().resolve()
        if result_root
        else _project_root() / "result"
    )


def _output_dir(output_dir: str | None, result_root: Path) -> Path:
    return Path(output_dir).expanduser().resolve() if output_dir else result_root


def _parse_timestamp(run_dir_name: str) -> str | None:
    parts = run_dir_name.split("_")
    if len(parts) < 2:
        return None
    return f"{parts[0]} {parts[1].replace('-', ':', 2)}"


def _find_latest_run(model_dir: Path) -> Path | None:
    candidates = []
    for path in model_dir.iterdir():
        if not path.is_dir():
            continue
        if path.name == "lr_search":
            continue
        if (path / "learning_curve.csv").exists() or (path / "log.txt").exists():
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _maybe_float(value: str) -> float | str:
    cleaned = value.strip()
    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def _parse_log(
    log_path: Path,
) -> tuple[dict[str, float | str], dict[str, float | None]]:
    params: dict[str, float | str] = {}
    test_metrics: dict[str, float | None] = {
        "test_acc": None,
        "test_top3": None,
        "test_loss": None,
        "test_macro_f1": None,
    }

    if not log_path.exists():
        return params, test_metrics

    for line in log_path.read_text(encoding="utf-8").splitlines():
        hp_match = HP_PATTERN.search(line)
        if hp_match:
            name = hp_match.group("name")
            if name in TARGET_PARAMS:
                params[name] = _maybe_float(hp_match.group("value"))
            continue

        test_match = TEST_ACC_PATTERN.search(line)
        if test_match:
            test_metrics["test_acc"] = float(test_match.group("acc"))

            loss_match = TEST_LOSS_PATTERN.search(line)
            if loss_match:
                test_metrics["test_loss"] = float(loss_match.group("loss"))

            top3_match = TEST_TOP3_PATTERN.search(line)
            if top3_match:
                test_metrics["test_top3"] = float(top3_match.group("top3"))

            macro_match = TEST_MACRO_PATTERN.search(line)
            if macro_match:
                test_metrics["test_macro_f1"] = float(macro_match.group("macro"))

    return params, test_metrics


def _load_learning_curve(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "learning_curve.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    log_path = run_dir / "log.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"{csv_path} が存在せず、log.txt も見つかりません")

    rows: dict[int, dict[str, float]] = {}
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        train_match = TRAIN_PATTERN.search(line)
        if train_match:
            epoch = int(train_match.group("epoch"))
            row = rows.setdefault(epoch, {"epoch": float(epoch)})
            row["train_loss"] = float(train_match.group("train_loss"))
            row["train_acc"] = float(train_match.group("train_acc"))
            row["train_acc_top3"] = float(train_match.group("train_acc_top3"))
            row["lr"] = float(train_match.group("lr"))
            continue

        val_match = VAL_PATTERN.search(line)
        if val_match:
            epoch = int(val_match.group("epoch"))
            row = rows.setdefault(epoch, {"epoch": float(epoch)})
            row["val_loss"] = float(val_match.group("val_loss"))
            row["val_acc"] = float(val_match.group("val_acc"))
            row["val_acc_top3"] = float(val_match.group("val_acc_top3"))
            macro = val_match.group("val_macro_f1")
            if macro is not None:
                row["val_macro_f1"] = float(macro)
            continue

        legacy_match = VAL_LEGACY_PATTERN.search(line)
        if legacy_match:
            epoch = int(legacy_match.group("epoch"))
            row = rows.setdefault(epoch, {"epoch": float(epoch)})
            row["val_acc"] = float(legacy_match.group("val_acc"))

    if not rows:
        raise FileNotFoundError(
            f"{csv_path} が存在せず、ログから学習曲線を復元できません"
        )

    ordered = [rows[idx] for idx in sorted(rows)]
    return pd.DataFrame(ordered)


def _infer_label_from_path(csv_path: Path) -> str:
    for parent in csv_path.parents:
        candidate = parent.name.upper()
        if candidate in MODEL_ORDER:
            return candidate

    parent_name = csv_path.parent.name
    if parent_name:
        return parent_name

    return csv_path.stem or "curve"


def _load_curves_from_csv_specs(specs: Sequence[str]) -> dict[str, pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}

    for idx, spec in enumerate(specs, start=1):
        if "=" in spec:
            label, path_str = spec.split("=", 1)
            label = label.strip()
        else:
            label, path_str = None, spec

        csv_path = Path(path_str).expanduser()
        if not csv_path.is_file():
            raise FileNotFoundError(f"指定された CSV が見つかりません: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            raise RuntimeError(f"CSV の読み込みに失敗しました: {csv_path}") from exc

        inferred = label or _infer_label_from_path(csv_path)
        final_label = inferred
        suffix = 2
        while final_label in curves:
            final_label = f"{inferred}_{suffix}"
            suffix += 1

        curves[final_label] = df

    return curves


def _select_best_validation(
    df: pd.DataFrame,
) -> tuple[int | None, dict[str, float | None]]:
    if "val_acc" not in df.columns:
        return None, {
            "val_acc": None,
            "val_acc_top3": None,
            "val_loss": None,
            "val_macro_f1": None,
        }

    idx = df["val_acc"].idxmax()
    row = df.loc[idx]
    best_epoch_value = int(row["epoch"]) if "epoch" in row else None

    return best_epoch_value, {
        "val_acc": float(row.get("val_acc", float("nan"))),
        "val_acc_top3": (
            float(row.get("val_acc_top3", float("nan")))
            if "val_acc_top3" in df.columns
            else None
        ),
        "val_loss": (
            float(row.get("val_loss", float("nan")))
            if "val_loss" in df.columns
            else None
        ),
        "val_macro_f1": (
            float(row.get("val_macro_f1", float("nan")))
            if "val_macro_f1" in df.columns
            else None
        ),
    }


def _extract_run(model: str, run_dir: Path) -> tuple[RunRecord, pd.DataFrame]:
    df = _load_learning_curve(run_dir)
    params, test_metrics = _parse_log(run_dir / "log.txt")
    best_epoch, val_metrics = _select_best_validation(df)

    record = RunRecord(
        model=model,
        run_dir=run_dir,
        timestamp=_parse_timestamp(run_dir.name),
        params=params,
        best_epoch=best_epoch,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )

    return record, df


def _format_param_string(params: dict[str, float | str]) -> str:
    order = [key for key in TARGET_PARAMS if key in params]
    if not order:
        return ""

    formatted: list[str] = []
    for key in order:
        value = params[key]
        if isinstance(value, float):
            formatted.append(f"{key}={value:g}")
        else:
            formatted.append(f"{key}={value}")
    return ", ".join(formatted)


def _build_summary(records: Iterable[RunRecord], result_root: Path) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "Model": record.model,
            "RunDir": record.run_dir.relative_to(result_root).as_posix(),
            "Timestamp": record.timestamp or "",
            "BestEpoch": record.best_epoch or "",
            "Parameters": _format_param_string(record.params),
            "ValTop1": record.val_metrics.get("val_acc"),
            "ValTop3": record.val_metrics.get("val_acc_top3"),
            "ValLoss": record.val_metrics.get("val_loss"),
            "ValMacroF1": record.val_metrics.get("val_macro_f1"),
            "TestTop1": record.test_metrics.get("test_acc"),
            "TestTop3": record.test_metrics.get("test_top3"),
            "TestLoss": record.test_metrics.get("test_loss"),
            "TestMacroF1": record.test_metrics.get("test_macro_f1"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = [
        "ValTop1",
        "ValTop3",
        "ValLoss",
        "ValMacroF1",
        "TestTop1",
        "TestTop3",
        "TestLoss",
        "TestMacroF1",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float, errors="ignore")

    order = [model for model in MODEL_ORDER if model in df["Model"].tolist()]
    if order:
        df["Model"] = pd.Categorical(df["Model"], categories=order, ordered=True)
        df = df.sort_values("Model")
    else:
        df = df.sort_values("Model")

    return df.reset_index(drop=True)


def _save_summary_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.4f")


def _compute_ylim(
    min_value: float | None,
    max_value: float | None,
    mode: str,
) -> tuple[float, float] | None:
    if min_value is None or max_value is None:
        return None

    if mode == "accuracy":
        lower = max(0.0, min_value - 0.05)
        upper = min(1.0, max_value + 0.05)
        if upper - lower < 0.2:
            midpoint = (upper + lower) / 2
            lower = max(0.0, midpoint - 0.1)
            upper = min(1.0, midpoint + 0.1)
        return lower, upper

    range_value = max_value - min_value

    if mode == "non_negative":
        base = range_value if range_value > 0 else max(max_value, 1.0)
        lower = 0.0 if min_value >= 0 else min_value - 0.1 * base
        upper = max_value + 0.1 * base
        if upper <= lower:
            upper = lower + max(0.1, abs(lower) * 0.1)
        return lower, upper

    margin = 0.1 * (range_value if range_value > 0 else max(abs(max_value), 1.0))
    lower = min_value - margin
    upper = max_value + margin
    if upper <= lower:
        upper = lower + max(0.1, abs(lower) * 0.1)
    return lower, upper


def _plot_combined_metric(
    curves: dict[str, pd.DataFrame],
    output_path: Path,
    config: MetricPlotConfig,
) -> bool:
    if not curves:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "black",
            "axes.labelweight": "bold",
            "lines.linewidth": 1.8,
        }
    )

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    cmap = plt.get_cmap("tab10")
    keys = [key for key in MODEL_ORDER if key in curves] or sorted(curves)

    min_value: float | None = None
    max_value: float | None = None
    plotted = False

    for idx, model in enumerate(keys):
        df = curves[model]
        if "epoch" not in df.columns:
            continue

        epochs = pd.to_numeric(df["epoch"], errors="coerce")
        if epochs.isna().all():
            continue

        color = cmap(idx % cmap.N)

        for line in config.lines:
            if line.column not in df.columns:
                continue

            values = pd.to_numeric(df[line.column], errors="coerce")
            mask = (~epochs.isna()) & (~values.isna())
            if not mask.any():
                continue

            y_values = values[mask]
            x_values = epochs[mask]

            current_min = float(y_values.min())
            current_max = float(y_values.max())
            min_value = (
                current_min if min_value is None else min(min_value, current_min)
            )
            max_value = (
                current_max if max_value is None else max(max_value, current_max)
            )

            ax.plot(
                x_values,
                y_values,
                label=f"{model} {line.label}",
                color=color,
                linestyle=line.linestyle,
                alpha=line.alpha,
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ylim = _compute_ylim(min_value, max_value, config.ylim_mode)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(config.ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.margins(x=0.01)
    ax.legend(frameon=False, ncol=config.legend_ncol, loc=config.legend_loc)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_combined_metric_mono(
    curves: dict[str, pd.DataFrame],
    output_path: Path,
    config: MetricPlotConfig,
) -> bool:
    if not curves:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "black",
            "axes.labelweight": "bold",
            "lines.linewidth": 1.8,
        }
    )

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    keys = [key for key in MODEL_ORDER if key in curves] or sorted(curves)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    grays = [0.0, 0.2, 0.4, 0.6]  # 1.0 is #ffffff

    min_value: float | None = None
    max_value: float | None = None
    plotted = False

    for idx, model in enumerate(keys):
        df = curves[model]
        if "epoch" not in df.columns:
            continue

        epochs = pd.to_numeric(df["epoch"], errors="coerce")
        if epochs.isna().all():
            continue

        marker = markers[idx % len(markers)]
        gray = grays[idx % len(grays)]
        line_color = (gray, gray, gray)

        for line in config.lines:
            if line.column not in df.columns:
                continue

            values = pd.to_numeric(df[line.column], errors="coerce")
            mask = (~epochs.isna()) & (~values.isna())
            if not mask.any():
                continue

            y_values = values[mask]
            x_values = epochs[mask]

            current_min = float(y_values.min())
            current_max = float(y_values.max())
            min_value = (
                current_min if min_value is None else min(min_value, current_min)
            )
            max_value = (
                current_max if max_value is None else max(max_value, current_max)
            )

            markevery = max(1, int(len(x_values) / 12))
            marker_face = "none" if "Train" in line.label else "white"

            ax.plot(
                x_values,
                y_values,
                label=f"{model} {line.label}",
                color=line_color,
                linestyle=line.linestyle,
                alpha=line.alpha,
                marker=marker,
                markersize=4.2,
                markerfacecolor=marker_face,
                markeredgecolor=line_color,
                markeredgewidth=0.7,
                markevery=markevery,
                zorder=2,
            )

            marker_x = x_values.iloc[::markevery]
            marker_y = y_values.iloc[::markevery]
            ax.plot(
                marker_x,
                marker_y,
                linestyle="None",
                marker=marker,
                markersize=4.2,
                markerfacecolor=marker_face,
                markeredgecolor=line_color,
                markeredgewidth=0.7,
                color=line_color,
                label="_nolegend_",
                zorder=3,
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ylim = _compute_ylim(min_value, max_value, config.ylim_mode)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(config.ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.margins(x=0.01)
    ax.legend(frameon=False, ncol=config.legend_ncol, loc=config.legend_loc)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _load_confusion_labels(label_map_path: Path, num_classes: int) -> list[str]:
    if not label_map_path.exists():
        return [str(idx) for idx in range(num_classes)]

    payload = json.loads(label_map_path.read_text(encoding="utf-8"))
    raw_map = payload.get("label_map", payload)
    names: list[str] = []
    for idx in range(num_classes):
        value = raw_map.get(str(idx), raw_map.get(idx, str(idx)))
        names.append(Path(str(value)).name)
    return names


def _load_confusion_matrix(run_dir: Path) -> tuple[np.ndarray, list[str]] | None:
    matrix_path = run_dir / "confusion_matrix.npy"
    if not matrix_path.exists():
        return None

    confusion = np.load(matrix_path)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError(f"無効な混同行列です: {matrix_path} (shape={confusion.shape})")
    if np.any(confusion < 0):
        raise ValueError(f"負の値を含む混同行列です: {matrix_path}")

    labels = _load_confusion_labels(run_dir / "label_map.json", int(confusion.shape[0]))
    return confusion, labels


def _select_tick_positions(size: int, max_ticks: int = 20) -> list[int]:
    if size <= max_ticks:
        return list(range(size))

    step = max(1, math.ceil(size / max_ticks))
    ticks = list(range(0, size, step))
    if ticks[-1] != size - 1:
        ticks.append(size - 1)
    return ticks


def _plot_summary_confusion_matrix(
    confusion_by_model: dict[str, tuple[np.ndarray, list[str]]],
    output_path: Path,
) -> bool:
    if not confusion_by_model:
        return False

    keys = [key for key in MODEL_ORDER if key in confusion_by_model]
    keys.extend(sorted([key for key in confusion_by_model if key not in keys]))

    n_panels = len(keys)
    cols = 2 if n_panels <= 4 else 3
    rows = math.ceil(n_panels / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4.4, rows * 4.2), squeeze=False
    )

    vmax = max(float(confusion_by_model[key][0].max()) for key in keys)
    if vmax <= 0:
        vmax = 1.0

    image = None
    for idx, model in enumerate(keys):
        ax = axes[idx // cols][idx % cols]
        confusion, labels = confusion_by_model[model]
        image = ax.imshow(
            confusion, interpolation="nearest", cmap="Greys", vmin=0, vmax=vmax
        )

        n_classes = int(confusion.shape[0])
        ticks = _select_tick_positions(n_classes)
        tick_labels = [labels[tick] for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_title(model, fontsize=11)
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")

    for idx in range(n_panels, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(
        left=0.08, right=0.85, bottom=0.1, top=0.94, wspace=0.28, hspace=0.32
    )
    if image is not None:
        cax = fig.add_axes([0.88, 0.16, 0.02, 0.68])
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.set_label("Count", rotation=270, labelpad=12)

    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return True


PLOT_CONFIGS: dict[str, MetricPlotConfig] = {
    "accuracy": MetricPlotConfig(
        ylabel="Accuracy",
        lines=(
            MetricLine("val_acc", "Val", "-"),
            MetricLine("train_acc", "Train", "--", 0.75),
        ),
        ylim_mode="accuracy",
        legend_loc="lower right",
        legend_ncol=2,
    ),
    "loss": MetricPlotConfig(
        ylabel="Loss",
        lines=(
            MetricLine("val_loss", "Val", "-"),
            MetricLine("train_loss", "Train", "--", 0.75),
        ),
        ylim_mode="non_negative",
        legend_loc="upper right",
        legend_ncol=2,
    ),
    "lr": MetricPlotConfig(
        ylabel="Learning Rate",
        lines=(MetricLine("lr", "LR", "-"),),
        ylim_mode="non_negative",
        legend_loc="upper right",
        legend_ncol=1,
    ),
    "top3": MetricPlotConfig(
        ylabel="Top-3 Accuracy",
        lines=(
            MetricLine("val_acc_top3", "Val Top-3", "-"),
            MetricLine("train_acc_top3", "Train Top-3", "--", 0.75),
        ),
        ylim_mode="accuracy",
        legend_loc="lower right",
        legend_ncol=2,
    ),
}


def main() -> None:
    args = _parse_args()
    result_root = _result_root(args.result_root)
    output_dir = _output_dir(args.output_dir, result_root)
    if not result_root.exists():
        print("result ディレクトリが見つかりません。", file=sys.stderr)
        sys.exit(1)

    csv_curves: dict[str, pd.DataFrame] | None = None
    if args.csv:
        try:
            csv_curves = _load_curves_from_csv_specs(args.csv)
        except (FileNotFoundError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        if not csv_curves:
            print(
                "指定された CSV からロード可能な学習曲線がありません。",
                file=sys.stderr,
            )
            sys.exit(1)

    latest_runs: dict[str, Path] = {}
    for model_dir in result_root.iterdir():
        if not model_dir.is_dir():
            continue
        latest = _find_latest_run(model_dir)
        if latest is not None:
            latest_runs[model_dir.name.upper()] = latest

    if not latest_runs and not args.csv:
        print("実行済みモデルが見つかりません。", file=sys.stderr)
        sys.exit(1)

    records: list[RunRecord] = []
    curves: dict[str, pd.DataFrame] = {}
    confusion_by_model: dict[str, tuple[np.ndarray, list[str]]] = {}
    for model, run_dir in sorted(latest_runs.items()):
        try:
            record, df = _extract_run(model, run_dir)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
        else:
            records.append(record)
            curves[model] = df

        try:
            confusion_entry = _load_confusion_matrix(run_dir)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            confusion_entry = None
        if confusion_entry is not None:
            confusion_by_model[model] = confusion_entry

    if not records and not args.csv:
        print("有効な実行結果が解析できませんでした。", file=sys.stderr)
        sys.exit(1)

    summary_df = _build_summary(records, result_root)
    summary_path = output_dir / "summary_results.csv"
    summary_saved = False
    if not summary_df.empty:
        _save_summary_table(summary_df, summary_path)
        summary_saved = True

    curves_to_plot = csv_curves if csv_curves is not None else curves
    generated: list[Path] = []
    skipped: list[str] = []
    if not curves_to_plot:
        target_desc = "指定された CSV" if args.csv else "最新実行結果"
        print(f"Warning: {target_desc} から学習曲線を取得できませんでした。", file=sys.stderr)
        sys.exit(1)

    for suffix, config in PLOT_CONFIGS.items():
        plot_path = output_dir / f"summary_learning_curve_{suffix}.png"
        if _plot_combined_metric(curves_to_plot, plot_path, config):
            generated.append(plot_path)
        else:
            skipped.append(plot_path.name)

    if args.mono_accuracy:
        mono_path = output_dir / "summary_learning_curve_accuracy_mono.png"
        if _plot_combined_metric_mono(
            curves_to_plot, mono_path, PLOT_CONFIGS["accuracy"]
        ):
            generated.append(mono_path)
        else:
            skipped.append(mono_path.name)

    confusion_path = output_dir / "summary_confusion_matrix.png"
    if _plot_summary_confusion_matrix(confusion_by_model, confusion_path):
        generated.append(confusion_path)
    else:
        skipped.append(confusion_path.name)

    if summary_saved:
        print(f"Saved summary CSV: {summary_path}")
    elif not args.csv:
        print("Warning: 集計結果を出力できませんでした。", file=sys.stderr)

    for path in generated:
        if path.name == "summary_confusion_matrix.png":
            print(f"Saved confusion summary: {path}")
        else:
            print(f"Saved learning curve: {path}")

    for name in skipped:
        print(f"Warning: {name} の生成対象データが見つかりませんでした。", file=sys.stderr)


if __name__ == "__main__":
    main()
