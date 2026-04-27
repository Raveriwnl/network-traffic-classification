#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


LABEL_FILE = "network_traffic_classfication_sample_label.csv"
PACKET_GLOB = "network_traffic_classfication_packet_sequence-*.csv"
REQUIRED_COLUMNS = {"flow_id", "arrive_time", "direction", "pkt_len"}
VISUALIZATION_UPLINK_DIRECTION_VALUE = 0
CLASS_NAME_MAP = {
    "cloud_game": "云游戏",
    "live": "直播（观众端）",
    "meeting": "会议",
    "message": "消息",
    "openlive": "直播（主播端）",
    "phone_game": "手机游戏",
    "short_video": "短视频",
    "video": "长视频",
}
CHINESE_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
]
TITLE_FONT_CANDIDATES = [
    "SimHei",
    "STHeiti",
    "Heiti SC",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "KaiTi",
    "STKaiti",
    "Kaiti SC",
    "AR PL UKai CN",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Huawei raw traffic CSV files")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("datasets/raw/huawei"),
        help="Directory containing Huawei raw CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualization/outputs/huawei_raw"),
        help="Directory for generated figures and summary tables.",
    )
    parser.add_argument(
        "--packet-bins",
        type=int,
        default=40,
        help="Number of histogram bins for packet length.",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=100,
        help="Number of histogram bins for arrival time.",
    )
    parser.add_argument(
        "--series-bins",
        type=int,
        default=100,
        help="Number of time bins for representative flow series plots.",
    )
    return parser.parse_args()


def infer_class_from_filename(csv_path: Path) -> str:
    return csv_path.stem.split("-")[-1]


def get_class_display_name(class_name: str) -> str:
    return CLASS_NAME_MAP.get(class_name, class_name)


def is_uplink_direction(direction_values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return direction_values == VISUALIZATION_UPLINK_DIRECTION_VALUE


def get_preferred_title_font() -> font_manager.FontProperties | None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in TITLE_FONT_CANDIDATES:
        if font_name in available_fonts:
            return font_manager.FontProperties(family=font_name, size=14)
    return None


def load_label_map(raw_dir: Path) -> tuple[pd.DataFrame, dict[int, str]]:
    label_path = raw_dir / LABEL_FILE
    label_df = pd.read_csv(label_path)
    if {"flow_id", "classification"} - set(label_df.columns):
        raise ValueError(f"{label_path} must contain flow_id and classification columns")

    label_df = label_df.copy()
    label_df["flow_id"] = label_df["flow_id"].astype(np.int64)
    label_df["classification"] = label_df["classification"].astype(str)
    return label_df, dict(zip(label_df["flow_id"], label_df["classification"]))


def read_packet_file(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=["flow_id", "arrive_time", "direction", "pkt_len"],
        dtype={
            "flow_id": np.int64,
            "arrive_time": np.float32,
            "direction": np.int8,
            "pkt_len": np.int16,
        },
    )
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")
    return df


def analyze_raw_data(raw_dir: Path, packet_bins: int, time_bins: int) -> dict[str, object]:
    label_df, label_map = load_label_map(raw_dir)
    packet_files = sorted(raw_dir.glob(PACKET_GLOB))
    if not packet_files:
        raise ValueError(f"No packet files found under {raw_dir}")

    packet_len_edges = np.linspace(0.0, 1500.0, packet_bins + 1, dtype=np.float64)
    arrival_time_edges = np.linspace(0.0, 5000.0, time_bins + 1, dtype=np.float64)

    packet_count_by_class: defaultdict[str, int] = defaultdict(int)
    packet_len_hist: defaultdict[str, np.ndarray] = defaultdict(lambda: np.zeros(packet_bins, dtype=np.int64))
    arrival_time_hist: defaultdict[str, np.ndarray] = defaultdict(lambda: np.zeros(time_bins, dtype=np.int64))
    flow_frames: list[pd.DataFrame] = []

    total_packets = 0
    for csv_path in packet_files:
        df = read_packet_file(csv_path)
        fallback_class = infer_class_from_filename(csv_path)
        df["classification"] = df["flow_id"].map(label_map).fillna(fallback_class)
        df["is_uplink"] = is_uplink_direction(df["direction"]).astype(np.int8)

        total_packets += int(len(df))
        class_packet_counts = df["classification"].value_counts()
        for class_name, count in class_packet_counts.items():
            packet_count_by_class[str(class_name)] += int(count)

        for class_name, group_df in df.groupby("classification", sort=False):
            class_key = str(class_name)
            packet_len_hist[class_key] += np.histogram(group_df["pkt_len"].to_numpy(), bins=packet_len_edges)[0]
            arrival_time_hist[class_key] += np.histogram(group_df["arrive_time"].to_numpy(), bins=arrival_time_edges)[0]

        flow_summary = (
            df.groupby("flow_id", sort=True)
            .agg(
                classification=("classification", "first"),
                packet_count=("pkt_len", "size"),
                duration_ms=("arrive_time", "max"),
                mean_pkt_len=("pkt_len", "mean"),
                uplink_packets=("is_uplink", "sum"),
            )
            .reset_index()
        )
        flow_summary["uplink_ratio"] = flow_summary["uplink_packets"] / flow_summary["packet_count"]
        flow_summary["downlink_packets"] = flow_summary["packet_count"] - flow_summary["uplink_packets"]
        flow_frames.append(flow_summary)

    if not flow_frames:
        raise ValueError("No flows were extracted from Huawei raw data")

    flow_df = pd.concat(flow_frames, ignore_index=True)
    class_order = sorted(set(label_df["classification"].tolist()) | set(flow_df["classification"].tolist()))

    flow_count_by_class = (
        flow_df["classification"].value_counts().reindex(class_order, fill_value=0).astype(int).to_dict()
    )
    packet_count_by_class_final = {
        class_name: int(packet_count_by_class.get(class_name, 0)) for class_name in class_order
    }

    return {
        "label_df": label_df,
        "label_map": label_map,
        "flow_df": flow_df,
        "class_order": class_order,
        "packet_len_edges": packet_len_edges,
        "arrival_time_edges": arrival_time_edges,
        "packet_len_hist": dict(packet_len_hist),
        "arrival_time_hist": dict(arrival_time_hist),
        "flow_count_by_class": flow_count_by_class,
        "packet_count_by_class": packet_count_by_class_final,
        "packet_file_paths": packet_files,
        "packet_files": [path.name for path in packet_files],
        "total_packets": total_packets,
        "total_flows": int(flow_df["flow_id"].nunique()),
    }


def configure_style() -> list[str]:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese_fonts = [font_name for font_name in CHINESE_FONT_CANDIDATES if font_name in available_fonts]

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#fffaf2",
            "axes.facecolor": "#fffdf8",
            "axes.edgecolor": "#3a312a",
            "axes.labelcolor": "#2f2923",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.color": "#2f2923",
            "ytick.color": "#2f2923",
            "grid.color": "#d8cfc1",
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "savefig.facecolor": "#fffaf2",
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.unicode_minus": False,
        }
    )
    if chinese_fonts:
        plt.rcParams["font.sans-serif"] = chinese_fonts + ["DejaVu Sans"]
    return [
        "#33658a",
        "#86bbd8",
        "#758e4f",
        "#f6ae2d",
        "#f26419",
        "#7a306c",
        "#2a9d8f",
        "#b56576",
    ]


def build_metric_summary(flow_df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    grouped = flow_df.groupby("classification")
    for class_name, class_df in grouped:
        metric_summary: dict[str, dict[str, float]] = {}
        for metric_name in ["packet_count", "duration_ms", "mean_pkt_len", "uplink_ratio"]:
            series = class_df[metric_name]
            metric_summary[metric_name] = {
                "mean": round(float(series.mean()), 3),
                "median": round(float(series.median()), 3),
                "min": round(float(series.min()), 3),
                "max": round(float(series.max()), 3),
            }
        summary[str(class_name)] = metric_summary
    return summary


def select_representative_flows(flow_df: pd.DataFrame, class_order: list[str]) -> pd.DataFrame:
    working_df = flow_df.copy()
    working_df["packet_count_log"] = np.log1p(working_df["packet_count"])
    working_df["duration_ms_log"] = np.log1p(working_df["duration_ms"])

    feature_columns = ["packet_count_log", "duration_ms_log", "mean_pkt_len", "uplink_ratio"]
    representative_rows: list[pd.Series] = []

    for class_name in class_order:
        class_df = working_df.loc[working_df["classification"] == class_name].copy()
        if class_df.empty:
            continue

        center = class_df[feature_columns].median()
        scale = (class_df[feature_columns] - center).abs().median()
        scale = scale.replace(0.0, 1.0).fillna(1.0)

        standardized_distance = (class_df[feature_columns] - center).abs().divide(scale, axis=1)
        class_df["representative_score"] = standardized_distance.sum(axis=1)
        representative_row = class_df.sort_values(["representative_score", "flow_id"], ascending=[True, True]).iloc[0]
        representative_rows.append(representative_row)

    representative_df = pd.DataFrame(representative_rows).reset_index(drop=True)
    representative_df["classification_zh"] = representative_df["classification"].map(get_class_display_name)
    return representative_df[
        [
            "classification",
            "classification_zh",
            "flow_id",
            "representative_score",
            "packet_count",
            "duration_ms",
            "mean_pkt_len",
            "uplink_ratio",
            "uplink_packets",
            "downlink_packets",
        ]
    ]


def load_selected_flow_packets(
    packet_files: list[Path],
    label_map: dict[int, str],
    selected_flow_ids: set[int],
) -> pd.DataFrame:
    selected_frames: list[pd.DataFrame] = []
    for csv_path in packet_files:
        df = read_packet_file(csv_path)
        selected_df = df.loc[df["flow_id"].isin(selected_flow_ids)].copy()
        if selected_df.empty:
            continue
        fallback_class = infer_class_from_filename(csv_path)
        selected_df["classification"] = selected_df["flow_id"].map(label_map).fillna(fallback_class)
        selected_df["source_file"] = csv_path.name
        selected_frames.append(selected_df)

    if not selected_frames:
        raise ValueError("Representative flows were selected, but no packet rows were found for them")

    packet_df = pd.concat(selected_frames, ignore_index=True)
    return packet_df.sort_values(["classification", "flow_id", "arrive_time", "direction", "pkt_len"])


def save_dataset_overview(output_dir: Path, flow_counts: dict[str, int], packet_counts: dict[str, int], colors: list[str]) -> None:
    title_font = get_preferred_title_font()
    classes = list(flow_counts.keys())
    class_labels = [get_class_display_name(class_name) for class_name in classes]
    indices = np.arange(len(classes))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    axes[0].bar(indices, [flow_counts[name] for name in classes], color=colors[: len(classes)], edgecolor="#2f2923")
    axes[0].set_title("各类别流数量", fontproperties=title_font)
    axes[0].set_ylabel("流数量")
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels(class_labels, rotation=30, ha="right")
    axes[0].grid(axis="y")

    axes[1].bar(indices, [packet_counts[name] for name in classes], color=colors[: len(classes)], edgecolor="#2f2923")
    axes[1].set_title("各类别包数量", fontproperties=title_font)
    axes[1].set_ylabel("包数量")
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(class_labels, rotation=30, ha="right")
    axes[1].grid(axis="y")

    fig.suptitle("Huawei 原始数据集概览", fontsize=15, fontweight="bold", fontproperties=title_font)
    fig.savefig(output_dir / "huawei_class_overview.png", dpi=180)
    plt.close(fig)


def save_flow_statistics(output_dir: Path, flow_df: pd.DataFrame, class_order: list[str], colors: list[str]) -> None:
    title_font = get_preferred_title_font()
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    metrics = [
        ("packet_count", "每条流的包数量", True),
        ("duration_ms", "流持续时间（毫秒）", False),
        ("uplink_ratio", "上行包占比", False),
    ]
    class_labels = [get_class_display_name(class_name) for class_name in class_order]

    for axis, (column, title, log_scale) in zip(axes, metrics):
        data = [flow_df.loc[flow_df["classification"] == class_name, column].to_numpy() for class_name in class_order]
        boxplot = axis.boxplot(
            data,
            vert=False,
            tick_labels=class_labels,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#1d1d1d", "linewidth": 1.4},
        )
        for patch, color in zip(boxplot["boxes"], colors[: len(class_order)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        axis.set_title(title, fontproperties=title_font)
        axis.grid(axis="x")
        if log_scale:
            axis.set_xscale("log")

    fig.suptitle("流级原始统计分布", fontsize=15, fontweight="bold", fontproperties=title_font)
    fig.savefig(output_dir / "huawei_flow_statistics.png", dpi=180)
    plt.close(fig)


def save_packet_length_profiles(
    output_dir: Path,
    class_order: list[str],
    packet_len_edges: np.ndarray,
    packet_len_hist: dict[str, np.ndarray],
    colors: list[str],
) -> None:
    title_font = get_preferred_title_font()
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    centers = 0.5 * (packet_len_edges[:-1] + packet_len_edges[1:])

    for axis, class_name, color in zip(axes.flat, class_order, colors[: len(class_order)]):
        counts = packet_len_hist.get(class_name, np.zeros_like(centers, dtype=np.int64))
        density = counts / counts.sum() if counts.sum() else counts
        axis.fill_between(centers, density, color=color, alpha=0.25)
        axis.plot(centers, density, color=color, linewidth=2.0)
        axis.set_title(get_class_display_name(class_name), fontproperties=title_font)
        axis.grid(axis="y")

    for axis in axes[-1]:
        axis.set_xlabel("包长（字节）")
    for axis in axes[:, 0]:
        axis.set_ylabel("密度")

    fig.suptitle("各类别包长分布", fontsize=15, fontweight="bold", fontproperties=title_font)
    fig.savefig(output_dir / "huawei_packet_length_profiles.png", dpi=180)
    plt.close(fig)


def save_arrival_time_heatmap(
    output_dir: Path,
    class_order: list[str],
    arrival_time_edges: np.ndarray,
    arrival_time_hist: dict[str, np.ndarray],
) -> None:
    title_font = get_preferred_title_font()
    heatmap_rows = []
    for class_name in class_order:
        counts = arrival_time_hist.get(class_name)
        if counts is None:
            counts = np.zeros(len(arrival_time_edges) - 1, dtype=np.int64)
        normalized = counts / counts.sum() if counts.sum() else counts.astype(np.float64)
        heatmap_rows.append(normalized)

    heatmap = np.vstack(heatmap_rows)
    fig, ax = plt.subplots(figsize=(16, 5.5))
    image = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
    ax.set_title("各类别包到达时间密度热力图", fontproperties=title_font)
    ax.set_yticks(np.arange(len(class_order)))
    ax.set_yticklabels([get_class_display_name(class_name) for class_name in class_order])

    tick_positions = np.linspace(0, len(arrival_time_edges) - 2, 6, dtype=int)
    tick_labels = [f"{arrival_time_edges[idx]:.0f}" for idx in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("5 秒窗口内到达时间（毫秒）")
    fig.colorbar(image, ax=ax, shrink=0.9, label="归一化密度")
    fig.savefig(output_dir / "huawei_arrival_time_heatmap.png", dpi=180)
    plt.close(fig)


def save_flow_feature_scatter(output_dir: Path, flow_df: pd.DataFrame, class_order: list[str], colors: list[str]) -> None:
    title_font = get_preferred_title_font()
    fig, ax = plt.subplots(figsize=(11, 7))
    for class_name, color in zip(class_order, colors[: len(class_order)]):
        class_flows = flow_df.loc[flow_df["classification"] == class_name]
        ax.scatter(
            class_flows["duration_ms"],
            class_flows["packet_count"],
            s=np.clip(class_flows["mean_pkt_len"] / 8.0, 10.0, 180.0),
            alpha=0.45,
            color=color,
            edgecolors="none",
            label=get_class_display_name(class_name),
        )

    ax.set_title("流持续时间与包数量关系", fontproperties=title_font)
    ax.set_xlabel("持续时间（毫秒）")
    ax.set_ylabel("包数量")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(output_dir / "huawei_flow_feature_scatter.png", dpi=180)
    plt.close(fig)


def save_representative_flow_timeseries(
    output_dir: Path,
    representative_df: pd.DataFrame,
    packet_df: pd.DataFrame,
    class_order: list[str],
    colors: list[str],
    series_bins: int,
) -> None:
    title_font = get_preferred_title_font()
    representative_dir = output_dir / "representative_flows"
    representative_dir.mkdir(parents=True, exist_ok=True)
    bin_edges = np.linspace(0.0, 5000.0, series_bins + 1, dtype=np.float64)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    combined_output = output_dir / "huawei_representative_flow_timeseries.png"
    if combined_output.exists():
        combined_output.unlink()

    for class_name, color in zip(class_order, colors[: len(class_order)]):
        representative_row = representative_df.loc[representative_df["classification"] == class_name]
        if representative_row.empty:
            continue

        fig, axis = plt.subplots(figsize=(12, 5.5))
        flow_id = int(representative_row.iloc[0]["flow_id"])
        class_name_zh = str(representative_row.iloc[0]["classification_zh"])
        flow_packets = packet_df.loc[packet_df["flow_id"] == flow_id]
        uplink_mask = is_uplink_direction(flow_packets["direction"].to_numpy(dtype=np.int8))
        packet_lengths = flow_packets["pkt_len"].to_numpy(dtype=np.float64)
        packet_times = flow_packets["arrive_time"].to_numpy(dtype=np.float64)

        uplink_bytes = np.histogram(packet_times[uplink_mask], bins=bin_edges, weights=packet_lengths[uplink_mask])[0]
        downlink_bytes = np.histogram(packet_times[~uplink_mask], bins=bin_edges, weights=packet_lengths[~uplink_mask])[0]

        axis.axhline(0.0, color="#3a312a", linewidth=0.9)
        axis.fill_between(centers, 0.0, uplink_bytes, step="mid", color=color, alpha=0.28)
        axis.plot(centers, uplink_bytes, color=color, linewidth=1.6, label="上行字节")
        axis.fill_between(centers, 0.0, -downlink_bytes, step="mid", color="#5c6770", alpha=0.2)
        axis.plot(centers, -downlink_bytes, color="#5c6770", linewidth=1.2, label="下行字节")
        axis.set_title(f"{class_name_zh}代表性 Flow 时序曲线\nflow_id={flow_id}", fontproperties=title_font)
        axis.set_xlabel("5 秒窗口内时间（毫秒）")
        axis.set_ylabel("每个时间箱中的字节数（上行为正，下行为负）")
        axis.grid(axis="y")
        axis.legend(frameon=False, ncol=2, loc="upper right")
        fig.savefig(representative_dir / f"{class_name}_representative_flow_timeseries.png", dpi=180)
        plt.close(fig)


def save_outputs(analysis: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = configure_style()

    flow_df = analysis["flow_df"].copy()
    class_order = analysis["class_order"]
    representative_df = select_representative_flows(flow_df, class_order)
    representative_packets = load_selected_flow_packets(
        analysis["packet_file_paths"],
        analysis["label_map"],
        set(representative_df["flow_id"].astype(int).tolist()),
    )

    save_dataset_overview(
        output_dir,
        analysis["flow_count_by_class"],
        analysis["packet_count_by_class"],
        colors,
    )
    save_flow_statistics(output_dir, flow_df, class_order, colors)
    save_packet_length_profiles(
        output_dir,
        class_order,
        analysis["packet_len_edges"],
        analysis["packet_len_hist"],
        colors,
    )
    save_arrival_time_heatmap(
        output_dir,
        class_order,
        analysis["arrival_time_edges"],
        analysis["arrival_time_hist"],
    )
    save_flow_feature_scatter(output_dir, flow_df, class_order, colors)
    save_representative_flow_timeseries(
        output_dir,
        representative_df,
        representative_packets,
        class_order,
        colors,
        analysis["series_bins"],
    )

    flow_df.sort_values(["classification", "flow_id"]).to_csv(output_dir / "huawei_flow_summary.csv", index=False)
    representative_df.sort_values(["classification", "flow_id"]).to_csv(
        output_dir / "huawei_representative_flows.csv", index=False
    )

    summary = {
        "packet_files": analysis["packet_files"],
        "classes": class_order,
        "class_display_names": {class_name: get_class_display_name(class_name) for class_name in class_order},
        "total_flows": int(analysis["total_flows"]),
        "total_packets": int(analysis["total_packets"]),
        "flow_count_by_class": analysis["flow_count_by_class"],
        "packet_count_by_class": analysis["packet_count_by_class"],
        "flow_metric_summary": build_metric_summary(flow_df),
        "representative_flows": representative_df.sort_values(["classification", "flow_id"]).to_dict(orient="records"),
    }
    with (output_dir / "huawei_visualization_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    analysis = analyze_raw_data(args.raw_dir, packet_bins=args.packet_bins, time_bins=args.time_bins)
    analysis["series_bins"] = args.series_bins
    save_outputs(analysis, args.output_dir)
    print(f"Saved Huawei raw data figures to {args.output_dir}")


if __name__ == "__main__":
    main()