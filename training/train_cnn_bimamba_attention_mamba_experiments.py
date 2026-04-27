import copy
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cnn_bimamba_attention import build_cnn_bimamba_attention_model


COMMON_CONFIG = {
    "seed": 42,
    "dataset_path": "datasets/processed/huawei/huawei_5s_1000bins_features.npz",
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "batch_size": 64,
    "epochs": 120,
    "learning_rate": 6e-4,
    "weight_decay": 3e-4,
    "scheduler_factor": 0.5,
    "scheduler_patience": 6,
    "scheduler_min_lr": 1e-5,
    "early_stop_patience": 18,
    "label_smoothing": 0.04,
    "mixup_alpha": 0.20,
    "mixup_prob": 0.40,
    "input_noise_std": 0.004,
    "time_mask_prob": 0.10,
    "time_mask_ratio": 0.04,
    "grad_clip_norm": 1.0,
    "use_sam": True,
    "sam_rho": 0.05,
    "use_ema": True,
    "ema_decay": 0.999,
    "eval_with_ema": True,
    "save_ema_checkpoint": True,
    "best_metric": "val_acc",
    "summary_metric": "test_acc",
    "report_clean_train_metrics": True,
    "num_workers": 0,
    "results_dir": "training/experiment_results/cnn_bimamba_attention_mamba",
    "log1p_features": [
        "pkt_count",
        "pkt_len_mean",
        "pkt_len_std",
        "last_pkt_global_iat",
        "uplink_pkt_len_sum",
        "downlink_pkt_len_sum",
        "uplink_pkt_count",
        "downlink_pkt_count",
    ],
}


EXPERIMENTS = [
    {
        "name": "round4_base_wide_l1_s32_k5_e3_lr45",
        "model_kwargs": {
            "num_mamba_layers": 1,
            "mamba_d_state": 32,
            "mamba_d_conv": 5,
            "mamba_expand": 3,
            "mamba_dropout": 0.16,
            "stem_dropout": 0.10,
            "head_dropout": 0.28,
            "drop_path_rate": 0.10,
            "feature_dropout": 0.06,
            "pool_dropout": 0.06,
        },
        "config_overrides": {
            "learning_rate": 4.5e-4,
            "weight_decay": 2e-4,
            "scheduler_patience": 8,
            "label_smoothing": 0.02,
            "mixup_alpha": 0.15,
            "mixup_prob": 0.30,
            "input_noise_std": 0.003,
        },
    },
    {
        "name": "round4_lr40_wide_l1_s32_k5_e3",
        "model_kwargs": {
            "num_mamba_layers": 1,
            "mamba_d_state": 32,
            "mamba_d_conv": 5,
            "mamba_expand": 3,
            "mamba_dropout": 0.16,
            "stem_dropout": 0.10,
            "head_dropout": 0.28,
            "drop_path_rate": 0.10,
            "feature_dropout": 0.06,
            "pool_dropout": 0.06,
        },
        "config_overrides": {
            "learning_rate": 4.0e-4,
            "weight_decay": 2e-4,
            "scheduler_patience": 10,
            "label_smoothing": 0.02,
            "mixup_alpha": 0.15,
            "mixup_prob": 0.30,
            "input_noise_std": 0.003,
        },
    },
    {
        "name": "round4_lr50_wide_l1_s32_k5_e3",
        "model_kwargs": {
            "num_mamba_layers": 1,
            "mamba_d_state": 32,
            "mamba_d_conv": 5,
            "mamba_expand": 3,
            "mamba_dropout": 0.16,
            "stem_dropout": 0.10,
            "head_dropout": 0.28,
            "drop_path_rate": 0.10,
            "feature_dropout": 0.06,
            "pool_dropout": 0.06,
        },
        "config_overrides": {
            "learning_rate": 5.0e-4,
            "weight_decay": 2e-4,
            "scheduler_patience": 8,
            "label_smoothing": 0.02,
            "mixup_alpha": 0.15,
            "mixup_prob": 0.30,
            "input_noise_std": 0.003,
        },
    },
    {
        "name": "round4_low_reg_wide_l1_s32_k5_e3",
        "model_kwargs": {
            "num_mamba_layers": 1,
            "mamba_d_state": 32,
            "mamba_d_conv": 5,
            "mamba_expand": 3,
            "mamba_dropout": 0.14,
            "stem_dropout": 0.08,
            "head_dropout": 0.24,
            "drop_path_rate": 0.08,
            "feature_dropout": 0.04,
            "pool_dropout": 0.04,
        },
        "config_overrides": {
            "learning_rate": 5e-4,
            "weight_decay": 1.5e-4,
            "scheduler_patience": 8,
            "label_smoothing": 0.01,
            "mixup_alpha": 0.10,
            "mixup_prob": 0.25,
            "input_noise_std": 0.002,
        },
    },
    {
        "name": "round4_high_state_wide_l1_s40_k5_e3",
        "model_kwargs": {
            "num_mamba_layers": 1,
            "mamba_d_state": 40,
            "mamba_d_conv": 5,
            "mamba_expand": 3,
            "mamba_dropout": 0.16,
            "stem_dropout": 0.10,
            "head_dropout": 0.28,
            "drop_path_rate": 0.10,
            "feature_dropout": 0.06,
            "pool_dropout": 0.06,
        },
        "config_overrides": {
            "learning_rate": 4.5e-4,
            "weight_decay": 2e-4,
            "scheduler_patience": 8,
            "label_smoothing": 0.02,
            "mixup_alpha": 0.15,
            "mixup_prob": 0.30,
            "input_noise_std": 0.003,
        },
    },
    {
        "name": "round4_mid_state_wide_l1_s24_k5_e3",
        "model_kwargs": {
            "num_mamba_layers": 1,
            "mamba_d_state": 24,
            "mamba_d_conv": 5,
            "mamba_expand": 3,
            "mamba_dropout": 0.16,
            "stem_dropout": 0.10,
            "head_dropout": 0.28,
            "drop_path_rate": 0.10,
            "feature_dropout": 0.06,
            "pool_dropout": 0.06,
        },
        "config_overrides": {
            "learning_rate": 5e-4,
            "weight_decay": 2e-4,
            "scheduler_patience": 8,
            "label_smoothing": 0.02,
            "mixup_alpha": 0.15,
            "mixup_prob": 0.30,
            "input_noise_std": 0.003,
        },
    },
]


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_device(device_str=None):
    if device_str:
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available")
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_split_config(config):
    ratio_sum = config["train_ratio"] + config["val_ratio"] + config["test_ratio"]
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")


def load_dataset(config):
    data_path = resolve_path(config["dataset_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {data_path}")

    with np.load(data_path, allow_pickle=True) as data:
        x = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        classes = data["classes"] if "classes" in data.files else np.unique(y)
        feature_names = (
            data["feature_names"].astype(str)
            if "feature_names" in data.files
            else np.array([f"feature_{i}" for i in range(x.shape[-1])], dtype=str)
        )

    return data_path, x, y, classes, feature_names


def split_dataset(x, y, seed, config):
    test_ratio = float(config["test_ratio"])
    val_ratio = float(config["val_ratio"])
    remaining_ratio = 1.0 - test_ratio
    val_ratio_within_temp = val_ratio / remaining_ratio

    x_temp, x_test, y_temp, y_test = train_test_split(
        x,
        y,
        test_size=test_ratio,
        random_state=int(seed),
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=val_ratio_within_temp,
        random_state=int(seed),
        stratify=y_temp,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def build_preprocessor(feature_names, log1p_features):
    feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names.tolist())}
    missing_features = [name for name in log1p_features if name not in feature_name_to_idx]
    if missing_features:
        raise ValueError(f"Unknown log1p features: {missing_features}")

    return [feature_name_to_idx[name] for name in log1p_features]


def apply_log1p_transform(x3d, feature_indices):
    transformed = x3d.copy()
    if feature_indices:
        transformed[:, :, feature_indices] = np.log1p(
            np.clip(transformed[:, :, feature_indices], a_min=0.0, a_max=None)
        )
    return transformed.astype(np.float32)


def fit_scaler(x_train_log):
    num_features = x_train_log.shape[-1]
    scaler = RobustScaler(with_centering=True, with_scaling=True)
    scaler.fit(x_train_log.reshape(-1, num_features))
    return scaler


def apply_scaler(x3d, scaler):
    num_features = x3d.shape[-1]
    x2d = scaler.transform(x3d.reshape(-1, num_features))
    return x2d.reshape(x3d.shape).astype(np.float32)


def prepare_data(config):
    data_path, x, y, classes, feature_names = load_dataset(config)
    split_data = split_dataset(x, y, seed=config["seed"], config=config)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data

    log1p_feature_indices = build_preprocessor(feature_names, config["log1p_features"])
    x_train_log = apply_log1p_transform(x_train, log1p_feature_indices)
    x_val_log = apply_log1p_transform(x_val, log1p_feature_indices)
    x_test_log = apply_log1p_transform(x_test, log1p_feature_indices)

    scaler = fit_scaler(x_train_log)
    x_train_scaled = apply_scaler(x_train_log, scaler)
    x_val_scaled = apply_scaler(x_val_log, scaler)
    x_test_scaled = apply_scaler(x_test_log, scaler)

    return {
        "data_path": data_path,
        "classes": classes,
        "feature_names": feature_names,
        "log1p_feature_indices": log1p_feature_indices,
        "x_train": x_train_scaled,
        "x_val": x_val_scaled,
        "x_test": x_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def build_dataloaders(prepared, batch_size, num_workers, device):
    pin_memory = device.type == "cuda"
    train_dataset = TensorDataset(
        torch.from_numpy(prepared["x_train"]).float(),
        torch.from_numpy(prepared["y_train"]).long(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(prepared["x_val"]).float(),
        torch.from_numpy(prepared["y_val"]).long(),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(prepared["x_test"]).float(),
        torch.from_numpy(prepared["y_test"]).long(),
    )

    loader_kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        if grad_norm == 0:
            if zero_grad:
                self.zero_grad()
            return
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for param in group["params"]:
                if param.grad is None:
                    continue
                e_w = param.grad * scale.to(param)
                param.add_(e_w)
                self.state[param]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.sub_(self.state[param].get("e_w", 0.0))
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM requires closure-based step, use first_step/second_step.")
        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
        return loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    norms.append(param.grad.norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}


def apply_time_mask(xb, mask_prob=0.2, mask_ratio=0.08):
    if mask_prob <= 0 or mask_ratio <= 0:
        return xb
    if np.random.rand() > mask_prob:
        return xb

    batch_size, seq_len, _ = xb.shape
    mask_len = max(1, int(seq_len * mask_ratio))
    starts = torch.randint(0, seq_len - mask_len + 1, (batch_size,), device=xb.device)

    for index in range(batch_size):
        start = starts[index]
        xb[index, start : start + mask_len, :] = 0.0
    return xb


def mixup_batch(xb, yb, alpha=0.2):
    if alpha <= 0:
        return xb, yb, yb, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(xb.size(0), device=xb.device)
    mixed_xb = lam * xb + (1.0 - lam) * xb[index]
    y_a, y_b = yb, yb[index]
    return mixed_xb, y_a, y_b, lam


def run_epoch(model, loader, criterion, config, device, optimizer=None, apply_augment=False, apply_mixup=False, ema=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    use_sam = is_train and bool(config.get("use_sam", False)) and hasattr(optimizer, "first_step")

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=device.type == "cuda")
        yb = yb.to(device, non_blocking=device.type == "cuda")
        use_mixup = False

        if is_train:
            optimizer.zero_grad()

            if apply_augment:
                noise_std = float(config["input_noise_std"])
                if noise_std > 0:
                    xb = xb + torch.randn_like(xb) * noise_std
                xb = apply_time_mask(
                    xb,
                    mask_prob=float(config["time_mask_prob"]),
                    mask_ratio=float(config["time_mask_ratio"]),
                )

            if apply_mixup:
                use_mixup = np.random.rand() < float(config["mixup_prob"])
                if use_mixup:
                    xb, y_a, y_b, lam = mixup_batch(xb, yb, alpha=float(config["mixup_alpha"]))

        if is_train and use_sam:
            logits = model(xb)
            if use_mixup:
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, yb)
            loss.backward()

            clip_norm = float(config["grad_clip_norm"])
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.first_step(zero_grad=True)

            logits_second = model(xb)
            if use_mixup:
                loss_second = lam * criterion(logits_second, y_a) + (1.0 - lam) * criterion(logits_second, y_b)
            else:
                loss_second = criterion(logits_second, yb)
            loss_second.backward()

            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.second_step(zero_grad=True)

            logits_for_metrics = logits.detach()
            loss_for_log = loss.detach()
        else:
            with torch.set_grad_enabled(is_train):
                logits = model(xb)
                if is_train and use_mixup:
                    loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
                else:
                    loss = criterion(logits, yb)
                if is_train:
                    loss.backward()
                    clip_norm = float(config["grad_clip_norm"])
                    if clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                    optimizer.step()

            logits_for_metrics = logits.detach()
            loss_for_log = loss.detach()

        if is_train and ema is not None:
            ema.update(model)

        running_loss += loss_for_log.item() * yb.size(0)
        preds = logits_for_metrics.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return running_loss / total, correct / total


def predict(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.numpy())
    return np.concatenate(all_targets), np.concatenate(all_preds)


def get_current_lr(optimizer):
    if hasattr(optimizer, "base_optimizer"):
        return float(optimizer.base_optimizer.param_groups[0]["lr"])
    return float(optimizer.param_groups[0]["lr"])


def build_optimizer(model, config):
    optimizer_kwargs = {
        "lr": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
    }
    if bool(config.get("use_sam", False)):
        return SAM(model.parameters(), AdamW, rho=float(config["sam_rho"]), **optimizer_kwargs)
    return AdamW(model.parameters(), **optimizer_kwargs)


def build_checkpoint_payload(model, experiment, config, metrics):
    return {
        "state_dict": copy.deepcopy(model.state_dict()),
        "experiment": copy.deepcopy(experiment),
        "config": copy.deepcopy(config),
        "metrics": copy.deepcopy(metrics),
    }


def merge_results(existing_results, new_results):
    merged_by_name = {result["name"]: result for result in existing_results}
    for result in new_results:
        merged_by_name[result["name"]] = result
    return list(merged_by_name.values())


def load_existing_results(results_dir):
    summary_json_path = results_dir / "summary.json"
    if summary_json_path.exists():
        with summary_json_path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in existing summary file: {summary_json_path}")
        return data

    summary_csv_path = results_dir / "summary.csv"
    if not summary_csv_path.exists():
        return []

    numeric_fields = {
        "best_epoch": int,
        "best_val_loss": float,
        "best_val_acc": float,
        "test_loss": float,
        "test_acc": float,
        "num_params": int,
        "num_trainable_params": int,
    }
    with summary_csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        rows = []
        for row in reader:
            normalized = dict(row)
            for field_name, caster in numeric_fields.items():
                normalized[field_name] = caster(normalized[field_name])
            rows.append(normalized)
    return rows


def evaluate_checkpoint(checkpoint_path, prepared, experiment, config, device):
    _, _, test_loader = build_dataloaders(
        prepared,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        device=device,
    )

    model = build_cnn_bimamba_attention_model(
        input_shape=prepared["x_train"].shape[1:],
        num_classes=len(np.unique(prepared["y_train"])),
        **experiment["model_kwargs"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, config, device, optimizer=None)
    y_true, y_pred = predict(model, test_loader, device)

    target_names = [str(item) for item in prepared["classes"]]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
    return test_loss, test_acc, report


def train_single_experiment(experiment, prepared, base_config, device, results_dir, experiment_index):
    experiment_config = copy.deepcopy(base_config)
    experiment_config.update(copy.deepcopy(experiment.get("config_overrides", {})))
    experiment_name = experiment["name"]
    checkpoint_path = results_dir / f"{experiment_name}.pt"
    history_path = results_dir / f"{experiment_name}_history.json"

    seed_everything(int(base_config["seed"]) + int(experiment_index))
    train_loader, val_loader, _ = build_dataloaders(
        prepared,
        batch_size=experiment_config["batch_size"],
        num_workers=experiment_config["num_workers"],
        device=device,
    )

    model = build_cnn_bimamba_attention_model(
        input_shape=prepared["x_train"].shape[1:],
        num_classes=len(np.unique(prepared["y_train"])),
        **experiment["model_kwargs"],
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(experiment_config["label_smoothing"]))
    optimizer = build_optimizer(model, experiment_config)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(experiment_config["scheduler_factor"]),
        patience=int(experiment_config["scheduler_patience"]),
        min_lr=float(experiment_config["scheduler_min_lr"]),
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_metric_name = str(experiment_config["best_metric"])
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0
    ema = ModelEMA(model, decay=float(experiment_config["ema_decay"])) if bool(experiment_config["use_ema"]) else None
    eval_with_ema = bool(experiment_config["eval_with_ema"])
    save_ema_checkpoint = bool(experiment_config["save_ema_checkpoint"])

    print(f"\n=== Experiment: {experiment_name} ===")
    print("Model kwargs:", experiment["model_kwargs"])

    for epoch in range(1, int(experiment_config["epochs"]) + 1):
        train_loss_aug, _ = run_epoch(
            model,
            train_loader,
            criterion,
            experiment_config,
            device,
            optimizer=optimizer,
            apply_augment=True,
            apply_mixup=True,
            ema=ema,
        )

        if bool(experiment_config.get("report_clean_train_metrics", True)):
            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                criterion,
                experiment_config,
                device,
                optimizer=None,
            )
        else:
            train_loss, train_acc = train_loss_aug, float("nan")

        if eval_with_ema and ema is not None:
            ema.apply_shadow(model)
            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                criterion,
                experiment_config,
                device,
                optimizer=None,
            )
            ema.restore(model)
        else:
            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                criterion,
                experiment_config,
                device,
                optimizer=None,
            )

        scheduler.step(val_loss)
        current_lr = get_current_lr(optimizer)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(current_lr))

        print(
            f"Epoch {epoch:03d}/{int(experiment_config['epochs'])} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if best_metric_name == "val_loss":
            is_better = val_loss < best_val_loss
        else:
            is_better = val_acc > best_val_acc

        if is_better:
            best_epoch = epoch
            best_val_acc = float(val_acc)
            best_val_loss = float(val_loss)
            early_stop_counter = 0

            if save_ema_checkpoint and ema is not None:
                ema.apply_shadow(model)
                payload = build_checkpoint_payload(
                    model,
                    experiment,
                    experiment_config,
                    {"epoch": best_epoch, "val_loss": best_val_loss, "val_acc": best_val_acc},
                )
                torch.save(payload, checkpoint_path)
                ema.restore(model)
            else:
                payload = build_checkpoint_payload(
                    model,
                    experiment,
                    experiment_config,
                    {"epoch": best_epoch, "val_loss": best_val_loss, "val_acc": best_val_acc},
                )
                torch.save(payload, checkpoint_path)

            print(
                f"Saved best model to {checkpoint_path} "
                f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})"
            )
        else:
            early_stop_counter += 1
            if early_stop_counter >= int(experiment_config["early_stop_patience"]):
                print("Early stopping triggered.")
                break

    with history_path.open("w", encoding="utf-8") as file_obj:
        json.dump(history, file_obj, indent=2)

    test_loss, test_acc, report = evaluate_checkpoint(
        checkpoint_path,
        prepared,
        experiment,
        experiment_config,
        device,
    )

    num_params = sum(param.numel() for param in model.parameters())
    num_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    result = {
        "name": experiment_name,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "num_params": int(num_params),
        "num_trainable_params": int(num_trainable_params),
        "model_kwargs": copy.deepcopy(experiment["model_kwargs"]),
        "classification_report": report,
    }
    print(f"Test loss={test_loss:.4f} | Test acc={test_acc:.4f}")
    return result


def save_summary(results, results_dir, best_metric=None):
    summary_json_path = results_dir / "summary.json"
    summary_csv_path = results_dir / "summary.csv"

    existing_results = load_existing_results(results_dir)
    merged_results = merge_results(existing_results, results)
    if best_metric is not None:
        merged_results = rank_results(merged_results, best_metric=best_metric)

    with summary_json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(merged_results, file_obj, indent=2)

    csv_columns = [
        "name",
        "best_epoch",
        "best_val_loss",
        "best_val_acc",
        "test_loss",
        "test_acc",
        "num_params",
        "num_trainable_params",
        "checkpoint_path",
    ]
    with summary_csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=csv_columns)
        writer.writeheader()
        for result in merged_results:
            writer.writerow({column: result[column] for column in csv_columns})

    return merged_results, summary_json_path, summary_csv_path


def rank_results(results, best_metric):
    if best_metric == "test_acc":
        return sorted(results, key=lambda item: item["test_acc"], reverse=True)
    if best_metric == "val_acc":
        return sorted(results, key=lambda item: item["best_val_acc"], reverse=True)
    return sorted(results, key=lambda item: item["best_val_loss"])


def main():
    config = copy.deepcopy(COMMON_CONFIG)
    validate_split_config(config)
    seed_everything(config["seed"])
    device = resolve_device()
    results_dir = resolve_path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_data(config)
    print("Using device:", device)
    print("Dataset:", prepared["data_path"])
    print("Train/Val/Test:", prepared["x_train"].shape, prepared["x_val"].shape, prepared["x_test"].shape)
    print("Applied log1p to features:", config["log1p_features"])
    print("Results dir:", results_dir)

    results = []
    for experiment_index, experiment in enumerate(EXPERIMENTS):
        result = train_single_experiment(
            experiment=experiment,
            prepared=prepared,
            base_config=config,
            device=device,
            results_dir=results_dir,
            experiment_index=experiment_index,
        )
        results.append(result)

    ranked_results, summary_json_path, summary_csv_path = save_summary(
        results,
        results_dir,
        best_metric=config.get("summary_metric", config["best_metric"]),
    )

    print("\n=== Ranked Results ===")
    for rank, result in enumerate(ranked_results, start=1):
        print(
            f"{rank:02d}. {result['name']} | "
            f"val_loss={result['best_val_loss']:.4f} | "
            f"val_acc={result['best_val_acc']:.4f} | "
            f"test_acc={result['test_acc']:.4f} | "
            f"params={result['num_params']:,}"
        )

    print("\nSummary JSON:", summary_json_path)
    print("Summary CSV:", summary_csv_path)


if __name__ == "__main__":
    main()

#nohup python3 -u training/train_cnn_bimamba_attention_mamba_experiments.py > output.log 2>&1 &