from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, TensorDataset

from report_baseline_comparisons import (
    IPT_CLIP_MS,
    IPT_SCALE,
    add_label_ids,
    build_flow_records,
    split_indices_blocked,
    split_indices_random,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_PACKETS = 30


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_models(raw: str) -> list[str]:
    model_names = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = {"dnn", "multimodal"}
    invalid = sorted(set(model_names) - allowed)
    if invalid:
        raise ValueError(f"Unsupported model names: {invalid}")
    if not model_names:
        raise ValueError("Expected at least one model name.")
    return model_names


def compute_basic_stats(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros(12, dtype=np.float32)

    mean = float(values.mean())
    std = float(values.std())
    var = float(values.var())
    centered = values - mean
    if std > 1e-6:
        normalized = centered / std
        skew = float(np.mean(normalized**3))
        kurt = float(np.mean(normalized**4) - 3.0)
    else:
        skew = 0.0
        kurt = 0.0

    return np.array(
        [
            float(values.min()),
            float(values.max()),
            mean,
            std,
            var,
            float(values.sum()),
            float(np.median(values)),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 75)),
            skew,
            kurt,
            float(values.size),
        ],
        dtype=np.float32,
    )


def build_dnn_feature_matrix(records_df) -> tuple[np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    feature_names: list[str] = []
    stat_names = [
        "min",
        "max",
        "mean",
        "std",
        "var",
        "sum",
        "median",
        "q25",
        "q75",
        "skew",
        "kurtosis",
        "count",
    ]
    groups = [
        "uplink_pkt_len",
        "downlink_pkt_len",
        "uplink_iat",
        "downlink_iat",
    ]
    for group in groups:
        for stat_name in stat_names:
            feature_names.append(f"{group}_{stat_name}")

    for row in records_df.itertuples(index=False):
        sizes = np.asarray(row.packet_sizes, dtype=np.float32)
        times = np.asarray(row.times_ms, dtype=np.float32)
        direction = np.asarray(row.direction_pm, dtype=np.float32)

        up_sizes = sizes[direction > 0]
        down_sizes = sizes[direction < 0]

        up_times = times[direction > 0]
        down_times = times[direction < 0]
        up_iat = np.diff(up_times) if up_times.size > 1 else np.empty(0, dtype=np.float32)
        down_iat = np.diff(down_times) if down_times.size > 1 else np.empty(0, dtype=np.float32)

        row_features = np.concatenate(
            [
                compute_basic_stats(up_sizes),
                compute_basic_stats(down_sizes),
                compute_basic_stats(up_iat),
                compute_basic_stats(down_iat),
            ],
            axis=0,
        )
        features.append(row_features)

    return np.stack(features, axis=0).astype(np.float32), feature_names


def build_pstats_tensor(records_df, max_packets: int = DEFAULT_MAX_PACKETS) -> np.ndarray:
    tensor = np.zeros((len(records_df), int(max_packets), 3), dtype=np.float32)
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

        iat = np.zeros(packet_count, dtype=np.float32)
        if packet_count > 1:
            iat[1:] = np.diff(times)
        iat = np.clip(iat, a_min=0.0, a_max=float(IPT_CLIP_MS)) * float(IPT_SCALE)

        tensor[row_idx, :packet_count, 0] = sizes
        tensor[row_idx, :packet_count, 1] = iat
        tensor[row_idx, :packet_count, 2] = direction
    return tensor


def fit_stats_scaler(train_x: np.ndarray) -> RobustScaler:
    scaler = RobustScaler(with_centering=True, with_scaling=True)
    scaler.fit(train_x)
    return scaler


def apply_stats_scaler(x: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    return scaler.transform(x).astype(np.float32)


def fit_sequence_channel_stats(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flattened = train_x.reshape(-1, train_x.shape[-1]).astype(np.float32)
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_sequence_channel_stats(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


class FourLayerDNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.conv(x))))


class MultimodalCNNMLP(nn.Module):
    def __init__(self, seq_channels: int, stats_dim: int, num_classes: int):
        super().__init__()
        self.seq_block1 = ConvBNAct(seq_channels, 16, kernel_size=3, dropout=0.15)
        self.seq_pool1 = nn.MaxPool1d(kernel_size=2)

        self.seq_block2 = ConvBNAct(16, 32, kernel_size=3, dropout=0.15)
        self.seq_pool2 = nn.MaxPool1d(kernel_size=2)
        self.seq_dropout = nn.Dropout1d(0.2)

        self.stats_branch = nn.Sequential(
            nn.LayerNorm(stats_dim),
            nn.Linear(stats_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, seq_x: torch.Tensor, stats_x: torch.Tensor) -> torch.Tensor:
        seq_x = seq_x.transpose(1, 2)
        seq_x = self.seq_block1(seq_x)
        seq_x = self.seq_pool1(seq_x)
        seq_x = self.seq_block2(seq_x)
        seq_x = self.seq_pool2(seq_x)
        seq_x = self.seq_dropout(seq_x).transpose(1, 2)
        seq_features = seq_x.mean(dim=1)

        stats_features = self.stats_branch(stats_x)
        fused = torch.cat([seq_features, stats_features], dim=1)
        return self.classifier(fused)


def create_loader(*tensors: torch.Tensor, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    dataset = TensorDataset(*tensors)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def batch_to_device(batch, device: torch.device):
    return [item.to(device, non_blocking=device.type == "cuda") for item in batch]


def run_epoch(model, loader: DataLoader, criterion, optimizer, device: torch.device) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        *inputs, target = batch
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(*inputs)
            loss = criterion(logits, target)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = target.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((preds == target).sum().item())
        total_count += int(batch_size)

    return total_loss / total_count, total_correct / total_count


def predict(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            *inputs, target = batch
            logits = model(*inputs)
            preds = logits.argmax(dim=1)
            all_true.append(target.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
    }


def train_model(model, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int, learning_rate: float, weight_decay: float, patience: int) -> tuple[nn.Module, dict[str, list[float]], float, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    best_val_loss = float("inf")
    wait = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for _epoch in range(int(epochs)):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        if val_acc > best_val_acc or (np.isclose(val_acc, best_val_acc) and val_loss < best_val_loss):
            best_val_acc = float(val_acc)
            best_val_loss = float(val_loss)
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break

    model.load_state_dict(best_state)
    return model, history, best_val_loss, best_val_acc


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def to_millions(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / 1_000_000.0, 6)


def estimate_model_macs(model: nn.Module, *example_inputs: torch.Tensor) -> int | None:
    if not example_inputs:
        return None

    profile_model = copy.deepcopy(model).to("cpu")
    cpu_inputs = [example_input.detach().to("cpu") for example_input in example_inputs]
    was_training = profile_model.training
    profile_model.eval()

    try:
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU], with_flops=True, record_shapes=False) as profiler:
                profile_model(*cpu_inputs)
    except Exception:
        return None
    finally:
        profile_model.train(was_training)

    total_flops = sum(int(getattr(event, "flops", 0) or 0) for event in profiler.key_averages())
    if total_flops <= 0:
        return None
    return int(total_flops / 2)


def build_split_tensors(stats_x: np.ndarray, seq_x: np.ndarray, labels: np.ndarray, split_indices: dict[str, np.ndarray]) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for split_name, split_idx in split_indices.items():
        result[split_name] = {
            "stats": torch.from_numpy(stats_x[split_idx]).float(),
            "seq": torch.from_numpy(seq_x[split_idx]).float(),
            "y": torch.from_numpy(labels[split_idx]).long(),
        }
    return result


def summarize_result(split_name: str, experiment_name: str, representation: str, model_name: str, param_count: int, macs: int | None, val_metrics: dict[str, object], test_metrics: dict[str, object], extra: dict[str, object] | None = None) -> dict[str, object]:
    row = {
        "split": split_name,
        "experiment": experiment_name,
        "representation": representation,
        "model": model_name,
        "param_count": int(param_count),
        "param_count_million": to_millions(param_count),
        "macs": macs,
        "macs_million": to_millions(macs),
        "val_accuracy": round(float(val_metrics["accuracy"]), 6),
        "val_macro_precision": round(float(val_metrics["macro_precision"]), 6),
        "val_macro_recall": round(float(val_metrics["macro_recall"]), 6),
        "val_macro_f1": round(float(val_metrics["macro_f1"]), 6),
        "test_accuracy": round(float(test_metrics["accuracy"]), 6),
        "test_balanced_accuracy": round(float(test_metrics["balanced_accuracy"]), 6),
        "test_macro_precision": round(float(test_metrics["macro_precision"]), 6),
        "test_macro_recall": round(float(test_metrics["macro_recall"]), 6),
        "test_macro_f1": round(float(test_metrics["macro_f1"]), 6),
    }
    if extra:
        row.update(extra)
    return row


def run_split(
    records_df,
    class_names: list[str],
    split_name: str,
    split_indices: dict[str, np.ndarray],
    seed: int,
    max_packets: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    model_names: list[str],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    seed_everything(seed)
    labels = records_df["label_id"].to_numpy(dtype=np.int64)
    stats_x, stats_feature_names = build_dnn_feature_matrix(records_df)
    seq_x = build_pstats_tensor(records_df, max_packets=max_packets)

    train_idx = split_indices["train"]
    stats_scaler = fit_stats_scaler(stats_x[train_idx])
    stats_x = apply_stats_scaler(stats_x, stats_scaler)
    seq_mean, seq_std = fit_sequence_channel_stats(seq_x[train_idx])
    seq_x = apply_sequence_channel_stats(seq_x, seq_mean, seq_std)

    tensors = build_split_tensors(stats_x, seq_x, labels, split_indices)
    summary_rows: list[dict[str, object]] = []
    details: dict[str, object] = {
        "split": split_name,
        "class_names": class_names,
        "stats_feature_names": stats_feature_names,
        "models": {},
    }

    if "dnn" in model_names:
        dnn_model = FourLayerDNN(input_dim=stats_x.shape[1], num_classes=len(class_names)).to(device)
        train_loader = create_loader(tensors["train"]["stats"], tensors["train"]["y"], batch_size=batch_size, shuffle=True, device=device)
        val_loader = create_loader(tensors["val"]["stats"], tensors["val"]["y"], batch_size=batch_size, shuffle=False, device=device)
        test_loader = create_loader(tensors["test"]["stats"], tensors["test"]["y"], batch_size=batch_size, shuffle=False, device=device)
        dnn_model, history, _best_val_loss, _best_val_acc = train_model(
            dnn_model,
            train_loader,
            val_loader,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
        )
        val_true, val_pred = predict(dnn_model, val_loader, device)
        test_true, test_pred = predict(dnn_model, test_loader, device)
        val_metrics = evaluate_predictions(val_true, val_pred, class_names)
        test_metrics = evaluate_predictions(test_true, test_pred, class_names)
        param_count = count_trainable_params(dnn_model)
        macs = estimate_model_macs(
            dnn_model,
            torch.zeros(1, stats_x.shape[1], dtype=torch.float32),
        )
        summary_rows.append(
            summarize_result(
                split_name=split_name,
                experiment_name="dnn_4layer_48feat",
                representation="stats48",
                model_name="dnn_4layer",
                param_count=param_count,
                macs=macs,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                extra={"feature_dim": int(stats_x.shape[1])},
            )
        )
        details["models"]["dnn_4layer_48feat"] = {
            "param_count": param_count,
            "param_count_million": to_millions(param_count),
            "macs": macs,
            "macs_million": to_millions(macs),
            "history": history,
            "val_report": val_metrics["report"],
            "test_report": test_metrics["report"],
        }

    if "multimodal" in model_names:
        multimodal_model = MultimodalCNNMLP(seq_channels=seq_x.shape[-1], stats_dim=stats_x.shape[1], num_classes=len(class_names)).to(device)
        train_loader = create_loader(tensors["train"]["seq"], tensors["train"]["stats"], tensors["train"]["y"], batch_size=batch_size, shuffle=True, device=device)
        val_loader = create_loader(tensors["val"]["seq"], tensors["val"]["stats"], tensors["val"]["y"], batch_size=batch_size, shuffle=False, device=device)
        test_loader = create_loader(tensors["test"]["seq"], tensors["test"]["stats"], tensors["test"]["y"], batch_size=batch_size, shuffle=False, device=device)
        multimodal_model, history, _best_val_loss, _best_val_acc = train_model(
            multimodal_model,
            train_loader,
            val_loader,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
        )
        val_true, val_pred = predict(multimodal_model, val_loader, device)
        test_true, test_pred = predict(multimodal_model, test_loader, device)
        val_metrics = evaluate_predictions(val_true, val_pred, class_names)
        test_metrics = evaluate_predictions(test_true, test_pred, class_names)
        param_count = count_trainable_params(multimodal_model)
        macs = estimate_model_macs(
            multimodal_model,
            torch.zeros(1, max_packets, seq_x.shape[-1], dtype=torch.float32),
            torch.zeros(1, stats_x.shape[1], dtype=torch.float32),
        )
        summary_rows.append(
            summarize_result(
                split_name=split_name,
                experiment_name=f"multimodal_cnn_mlp_p{max_packets}_stats48",
                representation="pstats30_plus_stats48",
                model_name="multimodal_cnn_mlp",
                param_count=param_count,
                macs=macs,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                extra={"packet_count": int(max_packets), "feature_dim": int(stats_x.shape[1])},
            )
        )
        details["models"][f"multimodal_cnn_mlp_p{max_packets}_stats48"] = {
            "param_count": param_count,
            "param_count_million": to_millions(param_count),
            "macs": macs,
            "macs_million": to_millions(macs),
            "history": history,
            "val_report": val_metrics["report"],
            "test_report": test_metrics["report"],
        }

    return summary_rows, details


def save_outputs(output_dir: Path, split_name: str, summary_rows: list[dict[str, object]], details: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"summary_{split_name}.csv"
    details_path = output_dir / f"details_{split_name}.json"

    import pandas as pd

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["test_accuracy", "test_macro_f1", "val_accuracy"],
        ascending=False,
    )
    summary_df.to_csv(summary_path, index=False)
    details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run report-aligned neural baseline comparisons on Huawei data.")
    parser.add_argument("--split", choices=["random", "blocked", "both"], default="both")
    parser.add_argument("--models", type=str, default="dnn,multimodal", help="Comma-separated: dnn,multimodal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-packets", type=int, default=DEFAULT_MAX_PACKETS)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "training" / "experiment_results" / "report_neural_baselines",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = parse_models(args.models)

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
        summary_rows, details = run_split(
            records_df=records_df,
            class_names=class_names,
            split_name=split_name,
            split_indices=split_indices,
            seed=args.seed,
            max_packets=args.max_packets,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            model_names=model_names,
            device=device,
        )
        save_outputs(args.output_dir, split_name, summary_rows, details)
        print(f"[{split_name}] neural baselines complete")


if __name__ == "__main__":
    main()