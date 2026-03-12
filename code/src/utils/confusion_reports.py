#!/usr/bin/env python3
"""Generate confusion-matrix based reports for a run directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _load_label_map(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    label_map = payload.get("label_map", payload)
    result: dict[int, str] = {}
    for key, value in label_map.items():
        result[int(key)] = str(value)
    return result


def _load_confusion(path: Path) -> np.ndarray:
    conf = np.load(path)
    if conf.ndim != 2 or conf.shape[0] != conf.shape[1]:
        raise ValueError(f"confusion matrix shape invalid: {conf.shape}")
    if conf.dtype != np.int64:
        raise ValueError(f"confusion dtype must be int64: {conf.dtype}")
    if np.any(conf < 0):
        raise ValueError("confusion matrix contains negative values")
    return conf


def _write_csv(path: Path, headers: list[str], rows: Iterable[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom > 0.0 else 0.0


def generate_reports(
    *,
    confusion_path: Path,
    label_map_path: Path,
    output_dir: Path,
    bottom_k: int = 10,
    top_k: int = 10,
) -> None:
    label_map = _load_label_map(label_map_path)
    conf = _load_confusion(confusion_path)
    num_classes = conf.shape[0]

    supports = conf.sum(axis=1)
    total = int(conf.sum())
    if int(supports.sum()) != total:
        raise ValueError(
            f"support.sum() mismatch: {supports.sum()} != {total}"
        )

    per_class_rows: list[list[object]] = []
    per_class_records: list[dict[str, object]] = []
    for class_idx in range(num_classes):
        tp = int(conf[class_idx, class_idx])
        fp = int(conf[:, class_idx].sum() - tp)
        fn = int(conf[class_idx, :].sum() - tp)
        support = int(supports[class_idx])
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = (
            _safe_div(2.0 * precision * recall, precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        label_name = label_map.get(class_idx, str(class_idx))
        per_class_records.append(
            {
                "label_id": class_idx,
                "label_name": label_name,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        per_class_rows.append(
            [
                class_idx,
                label_name,
                support,
                tp,
                fp,
                fn,
                f"{precision:.6f}",
                f"{recall:.6f}",
                f"{f1:.6f}",
            ]
        )

    _write_csv(
        output_dir / "per_class_metrics.csv",
        [
            "label_id",
            "label_name",
            "support",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ],
        per_class_rows,
    )

    bottom_sorted = sorted(
        per_class_records,
        key=lambda item: (float(item["f1"]), int(item["label_id"])),
    )
    bottom_rows: list[list[object]] = []
    for rank, record in enumerate(bottom_sorted[:bottom_k], start=1):
        bottom_rows.append(
            [
                rank,
                record["label_id"],
                record["label_name"],
                record["support"],
                record["tp"],
                record["fp"],
                record["fn"],
                f"{float(record['precision']):.6f}",
                f"{float(record['recall']):.6f}",
                f"{float(record['f1']):.6f}",
            ]
        )

    _write_csv(
        output_dir / "per_class_f1_bottom10.csv",
        [
            "rank",
            "label_id",
            "label_name",
            "support",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ],
        bottom_rows,
    )

    conf_off = conf.copy()
    np.fill_diagonal(conf_off, 0)
    flat = conf_off.ravel()
    sorted_indices = np.argsort(flat)[::-1]

    top_rows: list[list[object]] = []
    rank = 1
    for flat_index in sorted_indices:
        count = int(flat[flat_index])
        if count <= 0:
            break
        true_id = int(flat_index // num_classes)
        pred_id = int(flat_index % num_classes)
        support = int(supports[true_id])
        rate_in_true = _safe_div(count, support)
        top_rows.append(
            [
                rank,
                true_id,
                label_map.get(true_id, str(true_id)),
                pred_id,
                label_map.get(pred_id, str(pred_id)),
                count,
                f"{rate_in_true:.6f}",
            ]
        )
        rank += 1
        if rank > top_k:
            break

    _write_csv(
        output_dir / "top_confusion_pairs.csv",
        [
            "rank",
            "true_id",
            "true_name",
            "pred_id",
            "pred_name",
            "count",
            "rate_in_true",
        ],
        top_rows,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-class metrics and confusion pair reports.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory containing confusion_matrix.npy and label_map.json.",
    )
    parser.add_argument(
        "--confusion",
        type=Path,
        help="Path to confusion_matrix.npy.",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        help="Path to label_map.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for CSV files.",
    )
    parser.add_argument("--bottom-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.run_dir is None and (
        args.confusion is None or args.label_map is None
    ):
        raise SystemExit("run-dir か confusion/label-map の指定が必要です")

    run_dir = args.run_dir
    confusion_path = args.confusion
    label_map_path = args.label_map
    output_dir = args.output_dir

    if run_dir is not None:
        if confusion_path is None:
            confusion_path = run_dir / "confusion_matrix.npy"
        if label_map_path is None:
            label_map_path = run_dir / "label_map.json"
        if output_dir is None:
            output_dir = run_dir

    if confusion_path is None or label_map_path is None or output_dir is None:
        raise SystemExit("confusion, label_map, output_dir が指定されていません")

    generate_reports(
        confusion_path=confusion_path,
        label_map_path=label_map_path,
        output_dir=output_dir,
        bottom_k=args.bottom_k,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
