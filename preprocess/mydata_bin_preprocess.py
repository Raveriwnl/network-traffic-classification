#!/usr/bin/env python3
# python mydata_bin_preprocess.py --input-dir ../datasets/raw/mydata --output-dir ../datasets/processed/mydata
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from huawei_bin_preprocess import build_flow_tensor


DEFAULT_CLASSES = ("video", "shortvideo")
FEATURE_NAMES = [
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
]
TARGET_LABEL_ALIASES = {
    "shortvideo": "short_video",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter mydata video/shortvideo flows and split them into 5s samples"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("datasets/raw/mydata"),
        help="Input directory containing mydata class folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/processed/mydata"),
        help="Output directory for processed tensors.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class folders to process.",
    )
    parser.add_argument("--bins", type=int, default=1000, help="Number of fixed bins in a 5s sample.")
    parser.add_argument("--window-ms", type=float, default=5000.0, help="Sample window length in milliseconds.")
    parser.add_argument(
        "--exclude-ports",
        type=str,
        default="53",
        help="Comma-separated ports to exclude when selecting actual flows.",
    )
    return parser.parse_args()


def normalize_target_label(class_name: str) -> str:
    return TARGET_LABEL_ALIASES.get(class_name, class_name)


def normalize_optional_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def parse_int_set(raw_value: str) -> set[int]:
    values = {int(item.strip()) for item in raw_value.split(",") if item.strip()}
    return values


def iter_capture_pairs(class_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for metadata_path in sorted(class_dir.glob("network_traffic_classfication_flow_metadata-*.csv")):
        packet_path = class_dir / metadata_path.name.replace("flow_metadata", "packet_sequence")
        if not packet_path.exists():
            raise ValueError(f"Missing packet sequence file for metadata file: {metadata_path}")
        pairs.append((metadata_path, packet_path))
    if not pairs:
        raise ValueError(f"No flow metadata files found under {class_dir}")
    return pairs


def filter_actual_flows(metadata_df: pd.DataFrame, class_name: str, exclude_ports: set[int]) -> pd.DataFrame:
    required_cols = {
        "flow_id",
        "src_port",
        "dst_port",
        "duration_ms",
        "packet_count",
        "uplink_bytes",
        "downlink_bytes",
        "candidate_labels",
        "is_target_candidate",
    }
    missing = required_cols - set(metadata_df.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

    target_label = normalize_target_label(class_name)
    selected = metadata_df.copy()
    selected["candidate_labels"] = selected["candidate_labels"].fillna("")

    has_target_label = selected["candidate_labels"].str.split("|", regex=False).apply(
        lambda labels: target_label in labels if isinstance(labels, list) else False
    )
    uses_excluded_port = selected["src_port"].isin(exclude_ports) | selected["dst_port"].isin(exclude_ports)

    return selected[(selected["is_target_candidate"] == 1) & has_target_label & ~uses_excluded_port].copy()


def split_flow_into_segments(flow_df: pd.DataFrame, window_ms: float) -> list[tuple[int, int, pd.DataFrame]]:
    ordered = flow_df.sort_values("arrive_time", kind="mergesort").reset_index(drop=True).copy()
    segment_ids = np.floor(ordered["arrive_time"].to_numpy(dtype=np.float64) / window_ms).astype(np.int32)
    ordered["segment_id"] = segment_ids

    segments: list[tuple[int, int, pd.DataFrame]] = []
    for segment_id, segment_df in ordered.groupby("segment_id", sort=True):
        segment_start_ms = int(round(int(segment_id) * window_ms))
        rebased = segment_df[["flow_id", "arrive_time", "direction", "pkt_len"]].copy()
        rebased["arrive_time"] = rebased["arrive_time"] - segment_start_ms
        segments.append((int(segment_id), segment_start_ms, rebased))
    return segments


def output_prefix(class_names: list[str], bins: int) -> str:
    return f"mydata_{'_'.join(class_names)}_5s_{bins}bins"


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = [item.strip() for item in args.classes.split(",") if item.strip()]
    if not class_names:
        raise ValueError("--classes cannot be empty")

    exclude_ports = parse_int_set(args.exclude_ports)

    x_tensors: list[np.ndarray] = []
    labels: list[str] = []
    sample_rows: list[dict[str, object]] = []
    selected_flow_frames: list[pd.DataFrame] = []
    source_packet_files: list[str] = []
    source_flow_ids: list[int] = []
    segment_indices: list[int] = []

    for class_name in class_names:
        class_dir = input_dir / class_name
        if not class_dir.exists():
            raise ValueError(f"Class directory does not exist: {class_dir}")

        for metadata_path, packet_path in iter_capture_pairs(class_dir):
            metadata_df = pd.read_csv(metadata_path)
            packet_df = pd.read_csv(packet_path)

            required_packet_cols = {"flow_id", "arrive_time", "direction", "pkt_len"}
            missing_packet_cols = required_packet_cols - set(packet_df.columns)
            if missing_packet_cols:
                raise ValueError(f"Packet CSV missing required columns: {sorted(missing_packet_cols)}")

            selected_flows = filter_actual_flows(metadata_df, class_name=class_name, exclude_ports=exclude_ports)
            if selected_flows.empty:
                continue

            capture_name = metadata_path.stem.removeprefix("network_traffic_classfication_flow_metadata-")
            selected_flows = selected_flows.copy()
            selected_flows.insert(0, "class_name", class_name)
            selected_flows.insert(1, "capture_name", capture_name)
            selected_flows.insert(2, "source_metadata_file", metadata_path.name)
            selected_flows.insert(3, "source_packet_file", packet_path.name)
            selected_flow_frames.append(selected_flows)

            selected_flow_ids = set(selected_flows["flow_id"].astype(int).tolist())
            selected_packets = packet_df[packet_df["flow_id"].isin(selected_flow_ids)].copy()
            packet_groups = {
                int(flow_id): flow_packets.sort_values("arrive_time", kind="mergesort").reset_index(drop=True)
                for flow_id, flow_packets in selected_packets.groupby("flow_id", sort=True)
            }

            for flow_row in selected_flows.itertuples(index=False):
                flow_id = int(flow_row.flow_id)
                flow_packets = packet_groups.get(flow_id)
                if flow_packets is None or flow_packets.empty:
                    continue

                for segment_index, segment_start_ms, segment_packets in split_flow_into_segments(
                    flow_packets, window_ms=args.window_ms
                ):
                    tensor = build_flow_tensor(segment_packets, bins=args.bins, window_ms=args.window_ms)
                    segment_end_ms = segment_start_ms + int(segment_packets["arrive_time"].max())

                    x_tensors.append(tensor)
                    labels.append(class_name)
                    source_packet_files.append(packet_path.name)
                    source_flow_ids.append(flow_id)
                    segment_indices.append(segment_index)

                    sample_rows.append(
                        {
                            "sample_index": len(sample_rows),
                            "class_name": class_name,
                            "capture_name": capture_name,
                            "source_packet_file": packet_path.name,
                            "source_metadata_file": metadata_path.name,
                            "original_flow_id": flow_id,
                            "segment_index": segment_index,
                            "segment_start_ms": segment_start_ms,
                            "segment_end_ms": segment_end_ms,
                            "segment_packet_count": int(len(segment_packets)),
                            "segment_duration_ms": int(segment_packets["arrive_time"].max()),
                            "original_duration_ms": int(flow_row.duration_ms),
                            "original_packet_count": int(flow_row.packet_count),
                            "uplink_bytes": int(flow_row.uplink_bytes),
                            "downlink_bytes": int(flow_row.downlink_bytes),
                            "candidate_labels": normalize_optional_text(flow_row.candidate_labels),
                            "tls_sni": normalize_optional_text(getattr(flow_row, "tls_sni", "")),
                            "http_host": normalize_optional_text(getattr(flow_row, "http_host", "")),
                            "dns_queries": normalize_optional_text(getattr(flow_row, "dns_queries", "")),
                        }
                    )

    if not x_tensors:
        raise ValueError("No actual flows were selected from the specified classes.")

    class_set = set(labels)
    classes = [class_name for class_name in class_names if class_name in class_set]
    class_to_id = {class_name: index for index, class_name in enumerate(classes)}

    X = np.stack(x_tensors, axis=0)
    y = np.array([class_to_id[label] for label in labels], dtype=np.int64)

    prefix = output_prefix(class_names=classes, bins=args.bins)
    features_path = output_dir / f"{prefix}_features.npz"
    metadata_path = output_dir / f"{prefix}_metadata.csv"
    schema_path = output_dir / f"{prefix}_schema.json"
    selected_flows_path = output_dir / f"{prefix}_selected_flows.csv"

    np.savez_compressed(
        features_path,
        X=X,
        y=y,
        label=np.array(labels, dtype=object),
        classes=np.array(classes, dtype=object),
        feature_names=np.array(FEATURE_NAMES, dtype=object),
        source_packet_file=np.array(source_packet_files, dtype=object),
        original_flow_id=np.array(source_flow_ids, dtype=np.int64),
        segment_index=np.array(segment_indices, dtype=np.int64),
    )

    metadata_df = pd.DataFrame(sample_rows)
    metadata_df["label_id"] = y
    metadata_df.to_csv(metadata_path, index=False)

    selected_flows_df = pd.concat(selected_flow_frames, axis=0, ignore_index=True)
    selected_flows_df.to_csv(selected_flows_path, index=False)

    flow_counts = selected_flows_df.groupby("class_name")["flow_id"].count().to_dict()
    sample_counts = metadata_df.groupby("class_name")["sample_index"].count().to_dict()
    split_counts = metadata_df.groupby("class_name")["segment_index"].apply(lambda values: int((values > 0).sum())).to_dict()

    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "classes": classes,
                "window_ms": args.window_ms,
                "bins": args.bins,
                "bin_ms": args.window_ms / args.bins,
                "filter_rules": {
                    "is_target_candidate": 1,
                    "candidate_labels_must_include": [normalize_target_label(class_name) for class_name in classes],
                    "exclude_ports": sorted(exclude_ports),
                    "segmentation": "split each selected flow into contiguous 5s windows; each sample keeps only packets inside one window",
                },
                "shape": {
                    "samples": int(X.shape[0]),
                    "bins": int(X.shape[1]),
                    "features": int(X.shape[2]),
                },
                "selected_flows_per_class": {key: int(value) for key, value in flow_counts.items()},
                "samples_per_class": {key: int(value) for key, value in sample_counts.items()},
                "split_samples_per_class": {key: int(value) for key, value in split_counts.items()},
                "features": FEATURE_NAMES,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved tensor: {features_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved selected flows: {selected_flows_path}")
    print(f"Saved schema: {schema_path}")
    print(f"Tensor shape X={X.shape}, y={y.shape}, classes={classes}")


if __name__ == "__main__":
    main()