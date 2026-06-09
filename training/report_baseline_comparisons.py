from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from ml_baselines import FINAL_FEATURES, evaluate_model, get_huawei_raw_dir


DEFAULT_PACKET_COUNTS = (10, 20, 30)
DEFAULT_K_VALUES = (1, 3, 5)
DEFAULT_DIST_BINS = (16, 32)
SIZE_RANGE_MAX = 1600.0
TIME_RANGE_MAX_MS = 5000.0
IPT_CLIP_MS = 1000.0
IPT_SCALE = 0.1


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def build_flow_records(repo_root: str | Path | None = None) -> pd.DataFrame:
    raw_dir = get_huawei_raw_dir(repo_root)
    label_path = raw_dir / "network_traffic_classfication_sample_label.csv"
    label_df = pd.read_csv(label_path)
    label_map = dict(zip(label_df["flow_id"].astype(int), label_df["classification"].astype(str)))

    packet_files = sorted(raw_dir.glob("network_traffic_classfication_packet_sequence-*.csv"))
    if not packet_files:
        raise ValueError(f"No Huawei packet sequence files found under {raw_dir}")

    rows: list[dict[str, object]] = []
    for csv_file in packet_files:
        flow_df = pd.read_csv(csv_file)
        required_cols = {"flow_id", "arrive_time", "direction", "pkt_len"}
        missing = required_cols - set(flow_df.columns)
        if missing:
            raise ValueError(f"{csv_file.name} missing required columns: {sorted(missing)}")

        flow_df = flow_df.sort_values(["flow_id", "arrive_time"], kind="mergesort")
        for flow_id, grouped_df in flow_df.groupby("flow_id", sort=True):
            times = grouped_df["arrive_time"].to_numpy(dtype=np.float32)
            packet_lengths = grouped_df["pkt_len"].to_numpy(dtype=np.float32)
            direction_raw = grouped_df["direction"].to_numpy(dtype=np.int8)
            direction_pm = np.where(direction_raw == 1, 1.0, -1.0).astype(np.float32)

            if len(times) == 0:
                continue

            min_time = float(times.min())
            max_time = float(times.max())
            total_length = float(packet_lengths.sum())

            rows.append(
                {
                    "flow_id": int(flow_id),
                    "label": label_map.get(int(flow_id), csv_file.stem.split("-")[-1]),
                    "source_file": csv_file.name,
                    "times_ms": times,
                    "packet_sizes": packet_lengths,
                    "direction_pm": direction_pm,
                    "min_time_ms": min_time,
                    "max_time_ms": max_time,
                    "flow_duration_ms": max_time - min_time,
                    "num_packets": int(len(grouped_df)),
                    "avg_packet_size_bytes": float(packet_lengths.mean()),
                    "total_length_bytes": total_length,
                    "total_payload_bytes": total_length,
                    "uplink_pkt_count": int((direction_raw == 1).sum()),
                    "downlink_pkt_count": int((direction_raw == 0).sum()),
                }
            )

    if not rows:
        raise ValueError(f"No flows were parsed from packet sequence files under {raw_dir}")

    records_df = pd.DataFrame(rows).sort_values(["label", "source_file", "flow_id"]).reset_index(drop=True)
    return records_df


def add_label_ids(records_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    class_names = sorted(records_df["label"].astype(str).unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(class_names)}
    enriched = records_df.copy()
    enriched["label_id"] = enriched["label"].map(label_to_id).astype(np.int64)
    return enriched, class_names


def build_sequence_matrix(
    records_df: pd.DataFrame,
    max_packets: int,
    ipt_clip_ms: float = IPT_CLIP_MS,
    ipt_scale: float = IPT_SCALE,
) -> np.ndarray:
    matrix = np.zeros((len(records_df), int(max_packets), 3), dtype=np.float32)

    for row_idx, row in enumerate(records_df.itertuples(index=False)):
        times = np.asarray(row.times_ms, dtype=np.float32)
        sizes = np.asarray(row.packet_sizes, dtype=np.float32)
        direction = np.asarray(row.direction_pm, dtype=np.float32)
        packet_count = min(int(max_packets), len(times))
        if packet_count == 0:
            continue

        times = times[:packet_count]
        sizes = sizes[:packet_count]
        direction = direction[:packet_count]

        ipt = np.zeros(packet_count, dtype=np.float32)
        if packet_count > 1:
            ipt[1:] = np.diff(times)
        ipt = np.clip(ipt, a_min=0.0, a_max=float(ipt_clip_ms)) * float(ipt_scale)

        matrix[row_idx, :packet_count, 0] = sizes
        matrix[row_idx, :packet_count, 1] = ipt
        matrix[row_idx, :packet_count, 2] = direction

    return matrix.reshape(len(records_df), -1)


def build_dist_matrix(
    records_df: pd.DataFrame,
    bins: int,
    size_range_max: float = SIZE_RANGE_MAX,
    time_range_max_ms: float = TIME_RANGE_MAX_MS,
) -> np.ndarray:
    bins = int(bins)
    size_edges = np.linspace(0.0, float(size_range_max), bins + 1, dtype=np.float32)
    time_edges = np.linspace(0.0, float(time_range_max_ms), bins + 1, dtype=np.float32)
    features = np.zeros((len(records_df), bins * 4), dtype=np.float32)

    for row_idx, row in enumerate(records_df.itertuples(index=False)):
        times = np.asarray(row.times_ms, dtype=np.float32)
        sizes = np.asarray(row.packet_sizes, dtype=np.float32)
        direction = np.asarray(row.direction_pm, dtype=np.float32)

        if len(times) == 0:
            continue

        rel_times = times - float(times[0])
        rel_times = np.clip(rel_times, a_min=0.0, a_max=float(time_range_max_ms))
        sizes = np.clip(sizes, a_min=0.0, a_max=float(size_range_max))

        up_mask = direction > 0
        down_mask = ~up_mask
        total_packets = max(1, len(times))

        size_up = np.histogram(sizes[up_mask], bins=size_edges)[0].astype(np.float32) / total_packets
        size_down = np.histogram(sizes[down_mask], bins=size_edges)[0].astype(np.float32) / total_packets
        time_up = np.histogram(rel_times[up_mask], bins=time_edges)[0].astype(np.float32) / total_packets
        time_down = np.histogram(rel_times[down_mask], bins=time_edges)[0].astype(np.float32) / total_packets

        features[row_idx] = np.concatenate([size_up, size_down, time_up, time_down], axis=0)

    return features


def build_stats_matrix(records_df: pd.DataFrame) -> np.ndarray:
    return records_df.loc[:, FINAL_FEATURES].to_numpy(dtype=np.float32)


def split_indices_random(
    labels: np.ndarray,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, np.ndarray]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    indices = np.arange(len(labels), dtype=np.int64)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=float(test_ratio),
        random_state=int(seed),
        stratify=labels,
    )

    val_within_train_val = float(val_ratio) / (float(train_ratio) + float(val_ratio))
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_within_train_val,
        random_state=int(seed),
        stratify=labels[train_val_idx],
    )
    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def split_indices_blocked(
    records_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, np.ndarray]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    ordered = records_df.reset_index().sort_values(["label", "source_file", "flow_id"], kind="mergesort")
    for _, group_df in ordered.groupby("label", sort=True):
        group_indices = group_df["index"].to_numpy(dtype=np.int64)
        sample_count = len(group_indices)
        if sample_count < 3:
            raise ValueError("Blocked split requires at least 3 samples per label.")

        train_count = int(round(sample_count * float(train_ratio)))
        val_count = int(round(sample_count * float(val_ratio)))
        train_count = min(max(train_count, 1), sample_count - 2)
        val_count = min(max(val_count, 1), sample_count - train_count - 1)
        test_count = sample_count - train_count - val_count
        if test_count <= 0:
            val_count = max(1, val_count - 1)
            test_count = sample_count - train_count - val_count

        train_parts.append(group_indices[:train_count])
        val_parts.append(group_indices[train_count : train_count + val_count])
        test_parts.append(group_indices[train_count + val_count : train_count + val_count + test_count])

    return {
        "train": np.sort(np.concatenate(train_parts)),
        "val": np.sort(np.concatenate(val_parts)),
        "test": np.sort(np.concatenate(test_parts)),
    }


def compute_exact_match_rate(train_matrix: np.ndarray, test_matrix: np.ndarray) -> float:
    train_keys = {row.tobytes() for row in train_matrix}
    if not len(test_matrix):
        return 0.0
    hits = sum(row.tobytes() in train_keys for row in test_matrix)
    return float(hits / len(test_matrix))


def train_random_forest(
    train_x: np.ndarray,
    train_y: np.ndarray,
    seed: int,
    *,
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    max_features: str | float = "sqrt",
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        class_weight="balanced_subsample",
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    return model


def build_distribution_rf_kwargs(bins: int) -> dict[str, object]:
    if int(bins) <= 16:
        return {
            "n_estimators": 120,
            "max_depth": 8,
            "min_samples_leaf": 8,
            "max_features": 0.5,
        }
    return {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    }


def to_millions(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / 1_000_000.0, 6)


def estimate_random_forest_param_count(model: RandomForestClassifier) -> int:
    total = 0
    for estimator in getattr(model, "estimators_", []):
        tree = estimator.tree_
        total += int(tree.children_left.size)
        total += int(tree.children_right.size)
        total += int(tree.feature.size)
        total += int(tree.threshold.size)
        total += int(tree.value.size)
    return total


def estimate_knn_param_count(model: KNeighborsClassifier) -> int:
    fit_x = getattr(model, "_fit_X", None)
    fit_y = getattr(model, "_y", None)
    total = 0
    if fit_x is not None:
        total += int(np.asarray(fit_x).size)
    if fit_y is not None:
        total += int(np.asarray(fit_y).size)
    return total


def estimate_model_complexity(model: RandomForestClassifier | KNeighborsClassifier) -> tuple[int | None, int | None]:
    if isinstance(model, RandomForestClassifier):
        return estimate_random_forest_param_count(model), 0
    if isinstance(model, KNeighborsClassifier):
        fit_x = getattr(model, "_fit_X", None)
        macs = None
        if fit_x is not None:
            fit_x = np.asarray(fit_x)
            if fit_x.ndim == 2:
                # L1 k-NN has no true multiplies; use distance accumulation work as a per-sample proxy.
                macs = int(fit_x.shape[0] * fit_x.shape[1])
        return estimate_knn_param_count(model), macs
    return None, None


def summarize_metrics(split_name: str, experiment_name: str, representation: str, model_name: str, metrics: dict[str, object], val_metrics: dict[str, object], extra: dict[str, object] | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "split": split_name,
        "experiment": experiment_name,
        "representation": representation,
        "model": model_name,
        "param_count": None,
        "param_count_million": None,
        "macs": None,
        "macs_million": None,
        "val_accuracy": round(float(val_metrics["accuracy"]), 6),
        "val_macro_precision": round(float(val_metrics["macro_precision"]), 6),
        "val_macro_recall": round(float(val_metrics["macro_recall"]), 6),
        "val_macro_f1": round(float(val_metrics["macro_f1"]), 6),
        "test_accuracy": round(float(metrics["accuracy"]), 6),
        "test_balanced_accuracy": round(float(metrics["balanced_accuracy"]), 6),
        "test_macro_precision": round(float(metrics["macro_precision"]), 6),
        "test_macro_recall": round(float(metrics["macro_recall"]), 6),
        "test_macro_f1": round(float(metrics["macro_f1"]), 6),
    }
    if extra:
        row.update(extra)
    row["param_count_million"] = to_millions(row.get("param_count"))
    row["macs_million"] = to_millions(row.get("macs"))
    return row


def run_experiments_for_split(
    records_df: pd.DataFrame,
    class_names: list[str],
    split_name: str,
    split_indices: dict[str, np.ndarray],
    seed: int,
    packet_counts: list[int],
    k_values: list[int],
    dist_bins: list[int],
    include_stats_baseline: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    labels = records_df["label_id"].to_numpy(dtype=np.int64)
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    stats_matrix = build_stats_matrix(records_df)
    summary_rows: list[dict[str, object]] = []
    reports: dict[str, object] = {}

    if include_stats_baseline:
        stats_model = train_random_forest(stats_matrix[train_idx], labels[train_idx], seed=seed)
        stats_val_metrics = evaluate_model(stats_model, stats_matrix[val_idx], labels[val_idx], class_names)
        stats_test_metrics = evaluate_model(stats_model, stats_matrix[test_idx], labels[test_idx], class_names)
        stats_param_count, stats_macs = estimate_model_complexity(stats_model)
        experiment_name = "stats_rf"
        summary_rows.append(
            summarize_metrics(
                split_name=split_name,
                experiment_name=experiment_name,
                representation="stats",
                model_name="random_forest",
                metrics=stats_test_metrics,
                val_metrics=stats_val_metrics,
                extra={
                    "param_count": stats_param_count,
                    "macs": stats_macs,
                    "feature_dim": int(stats_matrix.shape[1]),
                    "train_size": int(len(train_idx)),
                    "val_size": int(len(val_idx)),
                    "test_size": int(len(test_idx)),
                },
            )
        )
        reports[experiment_name] = {
            "val_report": stats_val_metrics["report"],
            "test_report": stats_test_metrics["report"],
        }

    for max_packets in packet_counts:
        sequence_matrix = build_sequence_matrix(records_df, max_packets=max_packets)
        exact_match_rate = compute_exact_match_rate(sequence_matrix[train_idx], sequence_matrix[test_idx])
        for neighbor_count in k_values:
            knn = KNeighborsClassifier(
                n_neighbors=int(neighbor_count),
                metric="manhattan",
                algorithm="brute",
                n_jobs=-1,
            )
            knn.fit(sequence_matrix[train_idx], labels[train_idx])
            val_metrics = evaluate_model(knn, sequence_matrix[val_idx], labels[val_idx], class_names)
            test_metrics = evaluate_model(knn, sequence_matrix[test_idx], labels[test_idx], class_names)
            knn_param_count, knn_macs = estimate_model_complexity(knn)
            experiment_name = f"seq_knn_p{max_packets}_k{neighbor_count}"
            summary_rows.append(
                summarize_metrics(
                    split_name=split_name,
                    experiment_name=experiment_name,
                    representation="packet_sequence",
                    model_name="knn_l1",
                    metrics=test_metrics,
                    val_metrics=val_metrics,
                    extra={
                        "param_count": knn_param_count,
                        "macs": knn_macs,
                        "packet_count": int(max_packets),
                        "k": int(neighbor_count),
                        "feature_dim": int(sequence_matrix.shape[1]),
                        "test_exact_match_rate": round(exact_match_rate, 6),
                    },
                )
            )
            reports[experiment_name] = {
                "val_report": val_metrics["report"],
                "test_report": test_metrics["report"],
            }

    for bins in dist_bins:
        dist_matrix = build_dist_matrix(records_df, bins=bins)
        dist_model = train_random_forest(
            dist_matrix[train_idx],
            labels[train_idx],
            seed=seed,
            **build_distribution_rf_kwargs(bins),
        )
        val_metrics = evaluate_model(dist_model, dist_matrix[val_idx], labels[val_idx], class_names)
        test_metrics = evaluate_model(dist_model, dist_matrix[test_idx], labels[test_idx], class_names)
        dist_param_count, dist_macs = estimate_model_complexity(dist_model)
        experiment_name = f"dist_rf_b{bins}"
        summary_rows.append(
            summarize_metrics(
                split_name=split_name,
                experiment_name=experiment_name,
                representation="distribution",
                model_name="random_forest",
                metrics=test_metrics,
                val_metrics=val_metrics,
                extra={
                    "param_count": dist_param_count,
                    "macs": dist_macs,
                    "bins": int(bins),
                    "feature_dim": int(dist_matrix.shape[1]),
                    **{f"rf_{key}": value for key, value in build_distribution_rf_kwargs(bins).items()},
                },
            )
        )
        reports[experiment_name] = {
            "val_report": val_metrics["report"],
            "test_report": test_metrics["report"],
        }

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["test_accuracy", "test_macro_f1", "val_accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    metadata = {
        "split": split_name,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "class_names": class_names,
        "reports": reports,
    }
    return summary_df, metadata


def save_outputs(output_dir: Path, split_name: str, summary_df: pd.DataFrame, metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"summary_{split_name}.csv"
    metadata_path = output_dir / f"details_{split_name}.json"
    summary_df.to_csv(summary_path, index=False)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run report-aligned Huawei baseline comparison experiments.")
    parser.add_argument(
        "--split",
        choices=["random", "blocked", "both"],
        default="both",
        help="Which split protocol to run. 'blocked' keeps per-class flow_id order as a stricter proxy split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random split and RF training.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Training ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio.")
    parser.add_argument(
        "--packet-counts",
        type=str,
        default=",".join(str(value) for value in DEFAULT_PACKET_COUNTS),
        help="Comma-separated packet counts for sequence k-NN baselines.",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default=",".join(str(value) for value in DEFAULT_K_VALUES),
        help="Comma-separated k values for sequence k-NN baselines.",
    )
    parser.add_argument(
        "--dist-bins",
        type=str,
        default=",".join(str(value) for value in DEFAULT_DIST_BINS),
        help="Comma-separated histogram bin counts for distribution RF baselines.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip the stats_rf baseline and only run sequence/distribution baselines.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "training" / "experiment_results" / "report_baselines",
        help="Directory where summary CSV and detailed JSON outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet_counts = parse_int_list(args.packet_counts)
    k_values = parse_int_list(args.k_values)
    dist_bins = parse_int_list(args.dist_bins)

    records_df, class_names = add_label_ids(build_flow_records(REPO_ROOT))
    labels = records_df["label_id"].to_numpy(dtype=np.int64)

    split_plan: list[tuple[str, dict[str, np.ndarray]]] = []
    if args.split in {"random", "both"}:
        split_plan.append(
            (
                "random",
                split_indices_random(
                    labels=labels,
                    seed=args.seed,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                ),
            )
        )
    if args.split in {"blocked", "both"}:
        split_plan.append(
            (
                "blocked",
                split_indices_blocked(
                    records_df=records_df,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                ),
            )
        )

    for split_name, split_indices in split_plan:
        summary_df, metadata = run_experiments_for_split(
            records_df=records_df,
            class_names=class_names,
            split_name=split_name,
            split_indices=split_indices,
            seed=args.seed,
            packet_counts=packet_counts,
            k_values=k_values,
            dist_bins=dist_bins,
            include_stats_baseline=not args.skip_stats,
        )
        save_outputs(args.output_dir, split_name, summary_df, metadata)
        print(f"[{split_name}] top experiments:")
        print(summary_df.head(5).to_string(index=False))
        print()


if __name__ == "__main__":
    main()