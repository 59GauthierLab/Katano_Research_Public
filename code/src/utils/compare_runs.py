#!/usr/bin/env python3
"""Compare per-class F1 between two runs (GRU vs CNN)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_label_map(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    label_map = payload.get("label_map", payload)
    result: dict[int, str] = {}
    for key, value in label_map.items():
        result[int(key)] = str(value)
    return result


def _load_metrics(path: Path) -> dict[int, dict[str, float | int]]:
    metrics: dict[int, dict[str, float | int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_id = int(row["label_id"])
            metrics[label_id] = {
                "f1": float(row["f1"]),
                "support": int(float(row["support"])),
            }
    return metrics


def _write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate top improved classes between GRU and CNN runs.",
    )
    parser.add_argument("--cnn-run-dir", type=Path)
    parser.add_argument("--gru-run-dir", type=Path)
    parser.add_argument("--cnn", type=Path, help="CNN per_class_metrics.csv")
    parser.add_argument("--gru", type=Path, help="GRU per_class_metrics.csv")
    parser.add_argument("--label-map", type=Path, help="label_map.json path")
    parser.add_argument("--output", type=Path, help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cnn_metrics_path = args.cnn
    gru_metrics_path = args.gru
    label_map_path = args.label_map
    output_path = args.output

    if args.cnn_run_dir is not None:
        if cnn_metrics_path is None:
            cnn_metrics_path = args.cnn_run_dir / "per_class_metrics.csv"
        if label_map_path is None:
            label_map_path = args.cnn_run_dir / "label_map.json"

    if args.gru_run_dir is not None:
        if gru_metrics_path is None:
            gru_metrics_path = args.gru_run_dir / "per_class_metrics.csv"
        if label_map_path is None:
            label_map_path = args.gru_run_dir / "label_map.json"
        if output_path is None:
            output_path = args.gru_run_dir / "improved_vs_cnn_top10.csv"

    if cnn_metrics_path is None or gru_metrics_path is None or label_map_path is None:
        raise SystemExit("cnn/gru metrics と label_map の指定が必要です")

    if output_path is None:
        output_path = Path("improved_vs_cnn_top10.csv")

    label_map = _load_label_map(label_map_path)
    cnn_metrics = _load_metrics(cnn_metrics_path)
    gru_metrics = _load_metrics(gru_metrics_path)

    if set(cnn_metrics.keys()) != set(gru_metrics.keys()):
        raise ValueError("CNN/GRU の label_id 集合が一致しません")

    rows: list[list[object]] = []
    deltas: list[tuple[float, int]] = []
    for label_id, cnn in cnn_metrics.items():
        gru = gru_metrics[label_id]
        delta = float(gru["f1"]) - float(cnn["f1"])
        deltas.append((delta, label_id))

    deltas.sort(key=lambda item: item[0], reverse=True)

    for rank, (delta, label_id) in enumerate(deltas[: args.top_k], start=1):
        cnn_f1 = float(cnn_metrics[label_id]["f1"])
        gru_f1 = float(gru_metrics[label_id]["f1"])
        support = int(cnn_metrics[label_id]["support"])
        rows.append(
            [
                rank,
                label_id,
                label_map.get(label_id, str(label_id)),
                f"{delta:.6f}",
                f"{cnn_f1:.6f}",
                f"{gru_f1:.6f}",
                support,
            ]
        )

    _write_csv(
        output_path,
        [
            "rank",
            "label_id",
            "label_name",
            "delta_f1",
            "cnn_f1",
            "gru_f1",
            "support",
        ],
        rows,
    )


if __name__ == "__main__":
    main()
