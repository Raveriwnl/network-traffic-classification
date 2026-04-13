#!/usr/bin/env python3
# python huawei_bin_preprocess.py --input-dir ../datasets/raw/huawei --output-dir ../datasets/processed/huawei
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-6


def build_flow_tensor(flow_df: pd.DataFrame, bins: int, window_ms: float) -> np.ndarray:
    bin_ms = window_ms / bins

    times = flow_df["arrive_time"].to_numpy(dtype=np.float64)
    times = np.clip(times, 0.0, window_ms - EPS)

    pkt_len = flow_df["pkt_len"].to_numpy(dtype=np.float64)
    direction = flow_df["direction"].to_numpy(dtype=np.int8)

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    pkt_len = pkt_len[order]
    direction = direction[order]

    bin_idx = np.floor(times / bin_ms).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, bins - 1)

    counts = np.bincount(bin_idx, minlength=bins).astype(np.float64)
    len_sum = np.bincount(bin_idx, weights=pkt_len, minlength=bins).astype(np.float64)

    len_mean = np.zeros(bins, dtype=np.float64)
    np.divide(len_sum, counts, out=len_mean, where=counts > 0)

    len_sq_sum = np.bincount(bin_idx, weights=pkt_len**2, minlength=bins).astype(np.float64)
    len_var = np.zeros(bins, dtype=np.float64)
    np.divide(len_sq_sum, counts, out=len_var, where=counts > 0)
    len_var = np.clip(len_var - len_mean**2, a_min=0.0, a_max=None)
    len_std = np.sqrt(len_var)

    # Global IAT between the last packet of adjacent bins (0 when either bin is empty).
    last_pkt_time = np.full(bins, -1.0, dtype=np.float64)
    np.maximum.at(last_pkt_time, bin_idx, times)
    last_pkt_iat = np.zeros(bins, dtype=np.float64)
    valid_adjacent = (last_pkt_time[:-1] >= 0.0) & (last_pkt_time[1:] >= 0.0)
    last_pkt_iat[1:][valid_adjacent] = last_pkt_time[1:][valid_adjacent] - last_pkt_time[:-1][valid_adjacent]

    offset_sum = np.bincount(bin_idx, weights=times, minlength=bins).astype(np.float64)
    offset_mean_norm = np.zeros(bins, dtype=np.float64)
    np.divide(offset_sum, counts, out=offset_mean_norm, where=counts > 0)
    offset_mean_norm /= window_ms

    uplink_mask = (direction == 1).astype(np.float64)
    downlink_mask = 1.0 - uplink_mask
    uplink_count = np.bincount(bin_idx, weights=uplink_mask, minlength=bins).astype(np.float64)
    downlink_count = np.bincount(bin_idx, weights=downlink_mask, minlength=bins).astype(np.float64)

    uplink_len_sum = np.bincount(bin_idx, weights=pkt_len * uplink_mask, minlength=bins).astype(np.float64)
    downlink_len_sum = np.bincount(bin_idx, weights=pkt_len * downlink_mask, minlength=bins).astype(np.float64)

    uplink_ratio = np.zeros(bins, dtype=np.float64)
    np.divide(uplink_count, counts, out=uplink_ratio, where=counts > 0)

    features = np.stack(
        [
            counts,
            len_mean,
            len_std,
            last_pkt_iat,
            offset_mean_norm,
            uplink_ratio,
            uplink_len_sum,
            downlink_len_sum,
            uplink_count,
            downlink_count,
        ],
        axis=1,
    )
    return features.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Huawei 5s window bin aggregation preprocessing")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("datasets/raw/huawei"),
        help="Input directory containing Huawei CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/processed/huawei"),
        help="Output directory for processed tensors.",
    )
    parser.add_argument("--bins", type=int, default=1000, help="Number of fixed bins in 5s window.")
    parser.add_argument("--window-ms", type=float, default=5000.0, help="Window length in milliseconds.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    label_path = input_dir / "network_traffic_classfication_sample_label.csv"
    label_df = pd.read_csv(label_path)
    label_map = dict(zip(label_df["flow_id"].astype(int), label_df["classification"].astype(str)))

    packet_files = sorted(input_dir.glob("network_traffic_classfication_packet_sequence-*.csv"))
    if not packet_files:
        raise ValueError(f"No packet sequence files found under {input_dir}")

    x_tensors: list[np.ndarray] = []
    flow_ids: list[int] = []
    labels: list[str] = []
    source_files: list[str] = []

    for csv_file in packet_files:
        df = pd.read_csv(csv_file)
        required_cols = {"flow_id", "arrive_time", "direction", "pkt_len"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{csv_file} missing required columns: {sorted(missing)}")

        for flow_id, flow_df in df.groupby("flow_id", sort=True):
            flow_id = int(flow_id)
            tensor = build_flow_tensor(flow_df, bins=args.bins, window_ms=args.window_ms)
            label = label_map.get(flow_id)
            if label is None:
                # Fallback to file-derived class name if label file misses this flow.
                label = csv_file.stem.split("-")[-1]

            x_tensors.append(tensor)
            flow_ids.append(flow_id)
            labels.append(label)
            source_files.append(csv_file.name)

    if not x_tensors:
        raise ValueError("No flows were parsed from packet sequence files.")

    X = np.stack(x_tensors, axis=0)
    classes = sorted(set(labels))
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_id[c] for c in labels], dtype=np.int64)

    np.savez_compressed(
        output_dir / "huawei_5s_1000bins_features.npz",
        X=X,
        y=y,
        flow_id=np.array(flow_ids, dtype=np.int64),
        label=np.array(labels, dtype=object),
        classes=np.array(classes, dtype=object),
        feature_names=np.array(
            [
                "pkt_count",
                "pkt_len_mean",
                "pkt_len_std",
                "last_pkt_global_iat",
                "time_offset_mean_norm",
                "uplink_ratio",
                "uplink_pkt_len_sum",
                "downlink_pkt_len_sum",
                "uplink_pkt_count",
                "downlink_pkt_count",
            ],
            dtype=object,
        ),
    )

    metadata_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(flow_ids), dtype=np.int64),
            "flow_id": flow_ids,
            "label": labels,
            "label_id": y,
            "source_file": source_files,
        }
    )
    metadata_df.to_csv(output_dir / "huawei_5s_1000bins_metadata.csv", index=False)

    with (output_dir / "huawei_5s_1000bins_schema.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "window_ms": args.window_ms,
                "bins": args.bins,
                "bin_ms": args.window_ms / args.bins,
                "shape": {
                    "samples": int(X.shape[0]),
                    "bins": int(X.shape[1]),
                    "features": int(X.shape[2]),
                },
                "features": [
                    "pkt_count",
                    "pkt_len_mean",
                    "pkt_len_std",
                    "last_pkt_global_iat",
                    "time_offset_mean_norm",
                    "uplink_ratio",
                    "uplink_pkt_len_sum",
                    "downlink_pkt_len_sum",
                    "uplink_pkt_count",
                    "downlink_pkt_count",
                ],
                "classes": classes,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved tensor: {output_dir / 'huawei_5s_1000bins_features.npz'}")
    print(f"Saved metadata: {output_dir / 'huawei_5s_1000bins_metadata.csv'}")
    print(f"Saved schema: {output_dir / 'huawei_5s_1000bins_schema.json'}")
    print(f"Tensor shape X={X.shape}, y={y.shape}, classes={classes}")


if __name__ == "__main__":
    main()
