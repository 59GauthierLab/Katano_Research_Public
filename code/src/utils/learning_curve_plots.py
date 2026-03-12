#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Learning Curve グラフ生成スクリプト（PNG 専用）である。

- 欠損列は警告を出してスキップする。
- 論文掲載を想定した白黒スタイルを用いる。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt


def _detect_csv(base_dir: Path) -> Path:
    candidates = list(base_dir.glob("*.csv"))
    lc = base_dir / "learning_curve.csv"
    if lc.exists():
        return lc
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("CSVファイルが見つかりません。")
    raise FileNotFoundError("複数のCSVが見つかりました。明示的に指定してください。")


def generate_learning_curve_plots(
    csv_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    verbose: bool = False,
) -> list[Path]:
    """Generate learning-curve plots from the specified CSV."""

    csv_path = Path(csv_path).resolve()
    if output_dir is None:
        output_dir = csv_path.parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Using CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if verbose:
        print(f"列: {list(df.columns)}")

    plt.rcParams.update(  # type: ignore[misc]
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "figure.dpi": 300,
        }
    )

    required_base = {"epoch", "val_loss"}
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise ValueError(f"致命的: {missing_base} が存在しません。")

    expected_optional = [
        "train_loss",
        "train_acc",
        "val_acc",
        "lr",
        "train_acc_top3",
        "val_acc_top3",
    ]
    missing_optional = [c for c in expected_optional if c not in df.columns]
    if missing_optional:
        warnings.warn(f"次の列が存在しません: {', '.join(missing_optional)}")

    generated: list[Path] = []

    def _save(fig, filename: str) -> None:
        path = output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)
        if verbose:
            print(f"Saved: {filename}")

    if {"train_loss", "val_loss"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(df["epoch"], df["train_loss"], "--", color="black", label="Training")
        ax.plot(df["epoch"], df["val_loss"], "-", color="black", label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False)
        ax.grid(True, linestyle=":", linewidth=0.5)
        plt.tight_layout()
        _save(fig, "learning_curve_loss.png")
    else:
        warnings.warn(
            "train_loss または val_loss が不足しているため、損失グラフをスキップします。"
        )

    if {"train_acc", "val_acc"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(df["epoch"], df["train_acc"], "--", color="black", label="Training")
        ax.plot(df["epoch"], df["val_acc"], "-", color="black", label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
        ax.grid(True, linestyle=":", linewidth=0.5)
        plt.tight_layout()
        _save(fig, "learning_curve_accuracy.png")
    elif "val_acc" in df.columns:
        warnings.warn("train_acc がないため、Validation のみ描画します。")
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(df["epoch"], df["val_acc"], "-", color="black", label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
        ax.grid(True, linestyle=":", linewidth=0.5)
        plt.tight_layout()
        _save(fig, "learning_curve_accuracy.png")

    if {"train_acc_top3", "val_acc_top3"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(
            df["epoch"],
            df["train_acc_top3"],
            "--",
            color="black",
            label="Training Top-3",
        )
        ax.plot(
            df["epoch"],
            df["val_acc_top3"],
            "-",
            color="black",
            label="Validation Top-3",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Top-3 Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
        ax.grid(True, linestyle=":", linewidth=0.5)
        plt.tight_layout()
        _save(fig, "learning_curve_top3.png")
    elif "val_acc_top3" in df.columns:
        warnings.warn("train_acc_top3 がないため、Validation のみ描画します。")

    if "lr" in df.columns:
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(df["epoch"], df["lr"], "-", color="black")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_yscale("log")
        ax.grid(True, linestyle=":", linewidth=0.5)
        plt.tight_layout()
        _save(fig, "learning_curve_lr.png")

    if verbose:
        print("Completed: generated all PNG files.")

    return generated


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate learning-curve plots from CSV."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        help="Path to learning_curve.csv (default: auto-detect in current directory)",
    )
    parser.add_argument(
        "--out",
        dest="output",
        help="Directory to store generated figures (default: CSV directory)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> list[Path]:
    args = _parse_args(argv)
    base_dir = Path.cwd()
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = base_dir / csv_path
    else:
        csv_path = _detect_csv(base_dir)

    output_dir = Path(args.output).resolve() if args.output else None
    return generate_learning_curve_plots(
        csv_path,
        output_dir=output_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main(sys.argv[1:])
