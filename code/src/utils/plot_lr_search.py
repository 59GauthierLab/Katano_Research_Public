#!/usr/bin/env python3
"""lr_search_summary.csv から LR vs Accuracy のグラフを生成するCLI。

想定入力:
- 引数に lr_search の結果ディレクトリを与える
  例: result/CNN/lr_search/2026-01-27_17-24-55_lr_search/
- 上記配下の lr_search_summary.csv を読み取り、同ディレクトリにPNGを出力
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

# ヘッドレス環境でも描画可能なように `Agg` を使用する。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ACC_COL = "val_acc_at_end"
DEFAULT_OUTPUT_NAME = "lr_search_acc_vs_lr.png"


def _resolve_paths(input_path: str | Path) -> tuple[Path, Path]:
    """入力から (csv_path, output_dir) を解決する。"""

    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        if path.name != "lr_search_summary.csv":
            raise FileNotFoundError(
                f"lr_search_summary.csv を期待しましたが、指定は {path.name} です。"
            )
        return path, path.parent

    csv_path = path / "lr_search_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")
    return csv_path, path


def _configure_plot_style() -> None:
    plt.rcParams.update(  # type: ignore[misc]
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "figure.dpi": 200,
        }
    )


def generate_lr_vs_acc_plot(
    csv_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    acc_col: str = DEFAULT_ACC_COL,
    output_name: str = DEFAULT_OUTPUT_NAME,
    title: str | None = None,
    verbose: bool = False,
) -> Path:
    """lr_search_summary.csv から LR vs Accuracy のグラフを生成する。"""

    csv_path = Path(csv_path).expanduser().resolve()
    if output_dir is None:
        output_dir = csv_path.parent
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Using CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required_lr = {"lr"}
    missing_lr = required_lr - set(df.columns)
    if missing_lr:
        raise ValueError(
            f"必要な列が不足しています: {sorted(missing_lr)} / 利用可能: {list(df.columns)}"
        )

    if acc_col not in df.columns:
        raise ValueError(
            f"指定された acc 列が見つかりません: {acc_col} / 利用可能: {list(df.columns)}"
        )

    has_best = "best_val_acc" in df.columns

    cols = ["lr", acc_col] + (
        ["best_val_acc"] if has_best and acc_col != "best_val_acc" else []
    )
    plot_df = df[cols].dropna(subset=["lr", acc_col]).copy()
    if plot_df.empty:
        raise ValueError("lr と主系列 acc の有効な行がありません（dropna 後に空です）。")

    plot_df["lr"] = plot_df["lr"].astype(float)
    plot_df[acc_col] = plot_df[acc_col].astype(float)
    if "best_val_acc" in plot_df.columns:
        plot_df["best_val_acc"] = plot_df["best_val_acc"].astype(float)
    plot_df = plot_df.sort_values("lr")

    _configure_plot_style()

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    # 主系列: `val_acc_at_end`（既定）を実線で描画する。
    ax.plot(
        plot_df["lr"],
        plot_df[acc_col],
        "-o",
        color="black",
        markersize=4,
        label=acc_col,
    )

    # 補助系列: `best_val_acc` が存在する場合は破線で重ねる。
    if "best_val_acc" in plot_df.columns and acc_col != "best_val_acc":
        best_df = plot_df.dropna(subset=["best_val_acc"])
        if not best_df.empty:
            ax.plot(
                best_df["lr"],
                best_df["best_val_acc"],
                "--",
                color="dimgray",
                linewidth=1.6,
                label="best_val_acc",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Accuracy")

    # 縦軸はデータ範囲に合わせつつ [0, 1] に制限する。
    y_values = [plot_df[acc_col]]
    if "best_val_acc" in plot_df.columns and acc_col != "best_val_acc":
        y_values.append(plot_df["best_val_acc"].dropna())
    y_all = pd.concat(y_values)
    y_min = float(y_all.min())
    y_max = float(y_all.max())
    margin = max(0.01, (y_max - y_min) * 0.08)
    lower = max(0.0, y_min - margin)
    upper = min(1.0, y_max + margin)
    if lower >= upper:
        lower, upper = max(0.0, y_min - 0.05), min(1.0, y_max + 0.05)
    ax.set_ylim(lower, upper)

    ax.grid(True, linestyle=":", linewidth=0.6)
    if "best_val_acc" in plot_df.columns and acc_col != "best_val_acc":
        ax.legend(frameon=False)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    out_path = output_dir / output_name
    fig.savefig(out_path)
    plt.close(fig)

    if verbose:
        print(f"Saved plot: {out_path}")

    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "lr_search_summary.csv から LR(横軸) と Accuracy(縦軸) のグラフを生成します。"
        )
    )
    parser.add_argument(
        "path",
        help=(
            "lr_search の結果ディレクトリ、または lr_search_summary.csv のパス。"
        ),
    )
    parser.add_argument(
        "--acc-col",
        default=DEFAULT_ACC_COL,
        help=(
            "縦軸に使う Accuracy 列名（既定: val_acc_at_end）。"
        ),
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"出力PNGファイル名（既定: {DEFAULT_OUTPUT_NAME}）。",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="グラフタイトル（任意）。",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="詳細ログを表示します。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        csv_path, output_dir = _resolve_paths(args.path)
        out_path = generate_lr_vs_acc_plot(
            csv_path,
            output_dir=output_dir,
            acc_col=args.acc_col,
            output_name=args.output_name,
            title=args.title,
            verbose=args.verbose,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Failed: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Completed: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
