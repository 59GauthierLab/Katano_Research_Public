#!/usr/bin/env python3
"""run ディレクトリの confusion_matrix.npy から PNG を生成する CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

# ヘッドレス環境でも安定して保存できるよう `Agg` を使用する。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_NAME = "confusion_matrix.npy"
DEFAULT_LABEL_MAP_NAME = "label_map.json"
DEFAULT_OUTPUT_NAME = "confusion_matrix.png"


def _resolve_paths(path_arg: str | Path) -> tuple[Path, Path, Path]:
    base = Path(path_arg).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"指定パスが存在しません: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"ディレクトリを指定してください: {base}")

    npy_path = base / DEFAULT_INPUT_NAME
    if not npy_path.exists():
        raise FileNotFoundError(f"{DEFAULT_INPUT_NAME} が見つかりません: {npy_path}")
    label_map_path = base / DEFAULT_LABEL_MAP_NAME
    if not label_map_path.exists():
        raise FileNotFoundError(f"{DEFAULT_LABEL_MAP_NAME} が見つかりません: {label_map_path}")
    return npy_path, label_map_path, base / DEFAULT_OUTPUT_NAME


def _load_label_names(path: Path, num_classes: int) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_map = payload.get("label_map", payload)
    names: list[str] = []
    for idx in range(num_classes):
        value = raw_map.get(str(idx), raw_map.get(idx, str(idx)))
        names.append(Path(str(value)).name)
    return names


def _validate_confusion(confusion: np.ndarray) -> None:
    if confusion.ndim != 2:
        raise ValueError(f"2次元配列ではありません: ndim={confusion.ndim}")
    if confusion.shape[0] != confusion.shape[1]:
        raise ValueError(f"正方行列ではありません: shape={confusion.shape}")
    if np.any(confusion < 0):
        raise ValueError("負の値を含むため混同行列として不正です")


def save_confusion_matrix_png(
    run_dir: str | Path,
    *,
    cmap: str = "Greys",
    dpi: int = 200,
) -> Path:
    npy_path, label_map_path, out_path = _resolve_paths(run_dir)
    confusion = np.load(npy_path)
    _validate_confusion(confusion)

    n_classes = int(confusion.shape[0])
    label_names = _load_label_names(label_map_path, n_classes)
    fig_size = max(6.0, min(16.0, 0.35 * n_classes + 2.0))

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(confusion, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ticks = np.arange(n_classes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    ax.tick_params(axis="x", labelsize=8, rotation=90)
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "指定ディレクトリの confusion_matrix.npy を読み込み、"
            "同ディレクトリに confusion_matrix.png を生成します。"
        )
    )
    parser.add_argument(
        "run_dir",
        help=(
            "実行結果ディレクトリ。"
            "例: docs/result_02_48epochの全結果/GRU/2026-01-15_22-13-06_e08e82e/"
        ),
    )
    parser.add_argument(
        "--cmap",
        default="Greys",
        help="matplotlib のカラーマップ名（既定: Greys）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="出力PNGの解像度（既定: 200）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        out_path = save_confusion_matrix_png(
            args.run_dir,
            cmap=args.cmap,
            dpi=args.dpi,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Failed: {exc}", file=sys.stderr)
        return 1

    print(f"Generated: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
