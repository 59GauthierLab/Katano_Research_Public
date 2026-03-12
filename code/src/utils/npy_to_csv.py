"""Convert a .npy matrix to a CSV file next to it."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _default_fmt(array: np.ndarray) -> str:
    """Pick a human-friendly default format based on dtype."""
    if np.issubdtype(array.dtype, np.integer):
        return "%d"
    return "%.18g"


def convert_npy_to_csv(npy_path: Path, *, delimiter: str = ",") -> Path:
    """Convert a .npy file to .csv in the same directory."""
    if npy_path.suffix != ".npy":
        raise ValueError(f".npy ファイルを指定してください: {npy_path}")
    if not npy_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {npy_path}")

    array = np.load(npy_path)
    csv_path = npy_path.with_suffix(".csv")
    np.savetxt(csv_path, array, delimiter=delimiter, fmt=_default_fmt(array))
    return csv_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=".npy を同じディレクトリの .csv に変換します。",
    )
    parser.add_argument(
        "npy_path",
        type=Path,
        help="変換対象の .npy ファイルへのパス",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV の区切り文字（既定: ,）",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    csv_path = convert_npy_to_csv(args.npy_path, delimiter=args.delimiter)
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()

