import argparse
import importlib
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


DEFAULT_LOG1P_FEATURES = [
    "pkt_count",
    "pkt_len_mean",
    "pkt_len_std",
    "last_pkt_global_iat",
    "uplink_pkt_len_sum",
    "downlink_pkt_len_sum",
    "uplink_pkt_count",
    "downlink_pkt_count",
]


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

    for index, start in enumerate(starts):
        xb[index, start : start + mask_len, :] = 0.0
    return xb


def mixup_batch(xb, yb, alpha=0.2):
    if alpha <= 0:
        return xb, yb, yb, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(xb.size(0), device=xb.device)
    mixed_xb = lam * xb + (1 - lam) * xb[index]
    y_a, y_b = yb, yb[index]
    return mixed_xb, y_a, y_b, lam


def get_current_lr(optimizer):
    if hasattr(optimizer, "base_optimizer"):
        return float(optimizer.base_optimizer.param_groups[0]["lr"])
    return float(optimizer.param_groups[0]["lr"])


def resolve_training_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (TRAINING_DIR / path).resolve()


def load_checkpoint_state_dict(checkpoint_path, map_location):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def profile_model_macs(model_builder, input_shape, num_classes, model_kwargs, model_path):
    profile_model = model_builder(
        input_shape=input_shape,
        num_classes=num_classes,
        **model_kwargs,
    ).cpu()
    profile_model.load_state_dict(load_checkpoint_state_dict(model_path, map_location="cpu"))
    profile_model.eval()

    example_input = torch.randn(1, *input_shape, dtype=torch.float32)
    with profile(activities=[ProfilerActivity.CPU], with_flops=True, record_shapes=False, acc_events=True) as prof:
        with torch.no_grad():
            _ = profile_model(example_input)

    total_flops = sum(getattr(event, "flops", 0) for event in prof.key_averages())
    total_macs = total_flops / 2 if total_flops else float("nan")
    return total_flops, total_macs


def run_epoch(model, loader, criterion, device, config, optimizer=None, apply_augment=False, apply_mixup=False, ema=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    use_sam = is_train and bool(config.get("use_sam", False)) and hasattr(optimizer, "first_step")

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
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
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b) if use_mixup else criterion(logits, yb)
            loss.backward()

            clip_norm = float(config["grad_clip_norm"])
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.first_step(zero_grad=True)

            logits_second = model(xb)
            loss_second = (
                lam * criterion(logits_second, y_a) + (1 - lam) * criterion(logits_second, y_b)
                if use_mixup
                else criterion(logits_second, yb)
            )
            loss_second.backward()

            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.second_step(zero_grad=True)

            logits_for_metrics = logits.detach()
            loss_for_log = loss.detach()
        else:
            with torch.set_grad_enabled(is_train):
                logits = model(xb)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b) if is_train and use_mixup else criterion(logits, yb)
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


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--dataset-path")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--scheduler-factor", type=float)
    parser.add_argument("--scheduler-patience", type=int)
    parser.add_argument("--scheduler-min-lr", type=float)
    parser.add_argument("--early-stop-patience", type=int)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--mixup-alpha", type=float)
    parser.add_argument("--mixup-prob", type=float)
    parser.add_argument("--input-noise-std", type=float)
    parser.add_argument("--time-mask-prob", type=float)
    parser.add_argument("--time-mask-ratio", type=float)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--sam-rho", type=float)
    parser.add_argument("--ema-decay", type=float)
    parser.add_argument("--best-metric", choices=["val_loss", "val_acc"])
    parser.add_argument("--model-save-name")
    parser.add_argument("--eval-model-path")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--num-mamba-layers", type=int)
    parser.add_argument("--mamba-d-state", type=int)
    parser.add_argument("--mamba-d-conv", type=int)
    parser.add_argument("--mamba-expand", type=float)
    parser.add_argument("--mamba-dropout", type=float)
    parser.add_argument("--input-dropout", type=float)
    parser.add_argument("--stem-dropout", type=float)
    parser.add_argument("--head-dropout", type=float)
    parser.add_argument("--drop-path-rate", type=float)
    parser.add_argument("--feature-dropout", type=float)
    parser.add_argument("--pool-dropout", type=float)
    parser.add_argument("--disable-sam", action="store_true")
    parser.add_argument("--disable-ema", action="store_true")
    parser.add_argument("--disable-eval-with-ema", action="store_true")
    parser.add_argument("--disable-save-ema-checkpoint", action="store_true")
    parser.add_argument("--disable-clean-train-metrics", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def merge_config(default_config, args):
    config = deepcopy(default_config)
    model_kwargs = deepcopy(default_config.get("model_kwargs", {}))

    scalar_mappings = {
        "seed": args.seed,
        "cuda_device": args.cuda_device,
        "dataset_path": args.dataset_path,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_min_lr": args.scheduler_min_lr,
        "early_stop_patience": args.early_stop_patience,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "mixup_prob": args.mixup_prob,
        "input_noise_std": args.input_noise_std,
        "time_mask_prob": args.time_mask_prob,
        "time_mask_ratio": args.time_mask_ratio,
        "grad_clip_norm": args.grad_clip_norm,
        "sam_rho": args.sam_rho,
        "ema_decay": args.ema_decay,
        "best_metric": args.best_metric,
        "model_save_name": args.model_save_name,
        "eval_model_path": args.eval_model_path,
        "artifact_dir": args.artifact_dir,
    }
    for key, value in scalar_mappings.items():
        if value is not None:
            config[key] = value

    model_kwarg_mappings = {
        "d_model": args.d_model,
        "num_mamba_layers": args.num_mamba_layers,
        "mamba_d_state": args.mamba_d_state,
        "mamba_d_conv": args.mamba_d_conv,
        "mamba_expand": args.mamba_expand,
        "mamba_dropout": args.mamba_dropout,
        "input_dropout": args.input_dropout,
        "stem_dropout": args.stem_dropout,
        "head_dropout": args.head_dropout,
        "drop_path_rate": args.drop_path_rate,
        "feature_dropout": args.feature_dropout,
        "pool_dropout": args.pool_dropout,
    }
    for key, value in model_kwarg_mappings.items():
        if value is not None:
            model_kwargs[key] = value

    if args.disable_sam:
        config["use_sam"] = False
    if args.disable_ema:
        config["use_ema"] = False
    if args.disable_eval_with_ema:
        config["eval_with_ema"] = False
    if args.disable_save_ema_checkpoint:
        config["save_ema_checkpoint"] = False
    if args.disable_clean_train_metrics:
        config["report_clean_train_metrics"] = False

    config["model_kwargs"] = model_kwargs
    return config


def run_training(default_config, description, run_name):
    args = parse_args(description)
    config = merge_config(default_config, args)

    if not np.isclose(config["train_ratio"] + config["val_ratio"] + config["test_ratio"], 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        requested_cuda_device = config.get("cuda_device")
        if requested_cuda_device is not None:
            requested_cuda_device = int(requested_cuda_device)
            if requested_cuda_device >= torch.cuda.device_count():
                raise ValueError(
                    f"Requested cuda_device={requested_cuda_device}, but only {torch.cuda.device_count()} device(s) are visible"
                )
            torch.cuda.set_device(requested_cuda_device)
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = resolve_training_path(config["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {dataset_path}")

    artifacts_dir = resolve_training_path(config.get("artifact_dir") or f"experiment_results/{run_name}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_save_name = str(config["model_save_name"])
    best_model_path = TRAINING_DIR / "models" / model_save_name
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    if args.print_config:
        printable_config = deepcopy(config)
        printable_config["dataset_path"] = str(dataset_path)
        printable_config["artifact_dir"] = str(artifacts_dir)
        printable_config["best_model_path"] = str(best_model_path)
        print(json.dumps(printable_config, indent=2, ensure_ascii=False))

    print("Using device:", device)
    if torch.cuda.is_available():
        print("CUDA device index:", torch.cuda.current_device())
    print("Dataset:", dataset_path)
    print("Model:", f"{config['model_module']}.{config['model_builder']}")
    print("Model kwargs:", config["model_kwargs"])
    print("Checkpoint output:", best_model_path)
    print("Artifacts dir:", artifacts_dir)

    data = np.load(dataset_path, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    classes = data["classes"] if "classes" in data.files else np.unique(y)
    feature_names = (
        data["feature_names"].astype(str)
        if "feature_names" in data.files
        else np.array([f"feature_{index}" for index in range(x.shape[-1])], dtype=str)
    )

    print("X shape:", x.shape)
    print("y shape:", y.shape)
    print("num classes:", len(classes), classes)
    print("feature names:", feature_names.tolist())

    test_ratio = float(config["test_ratio"])
    val_ratio = float(config["val_ratio"])
    remaining_ratio = 1.0 - test_ratio
    val_ratio_within_temp = val_ratio / remaining_ratio

    x_temp, x_test, y_temp, y_test = train_test_split(
        x,
        y,
        test_size=test_ratio,
        random_state=seed,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=val_ratio_within_temp,
        random_state=seed,
        stratify=y_temp,
    )

    print("Train shape:", x_train.shape, y_train.shape)
    print("Val shape:", x_val.shape, y_val.shape)
    print("Test shape:", x_test.shape, y_test.shape)

    num_features = x_train.shape[-1]
    feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names.tolist())}
    missing_log1p_features = [name for name in config["log1p_features"] if name not in feature_name_to_idx]
    if missing_log1p_features:
        raise ValueError(f"Unknown log1p features: {missing_log1p_features}")
    log1p_feature_indices = [feature_name_to_idx[name] for name in config["log1p_features"]]

    def apply_log1p_transform(x3d):
        transformed = x3d.copy()
        if log1p_feature_indices:
            transformed[:, :, log1p_feature_indices] = np.log1p(
                np.clip(transformed[:, :, log1p_feature_indices], a_min=0.0, a_max=None)
            )
        return transformed.astype(np.float32)

    x_train_log = apply_log1p_transform(x_train)
    x_val_log = apply_log1p_transform(x_val)
    x_test_log = apply_log1p_transform(x_test)

    scaler = RobustScaler(with_centering=True, with_scaling=True)
    scaler.fit(x_train_log.reshape(-1, num_features))

    def apply_scaler(x3d):
        x2d = scaler.transform(x3d.reshape(-1, num_features))
        return x2d.reshape(x3d.shape).astype(np.float32)

    x_train_scaled = apply_scaler(x_train_log)
    x_val_scaled = apply_scaler(x_val_log)
    x_test_scaled = apply_scaler(x_test_log)

    model_module = importlib.import_module(config["model_module"])
    model_builder = getattr(model_module, config["model_builder"])
    num_classes = len(np.unique(y))
    model = model_builder(
        input_shape=x_train_scaled.shape[1:],
        num_classes=num_classes,
        **config["model_kwargs"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))

    if bool(config.get("use_sam", False)):
        optimizer = SAM(
            model.parameters(),
            AdamW,
            rho=float(config["sam_rho"]),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
        min_lr=float(config["scheduler_min_lr"]),
    )

    train_dataset = TensorDataset(torch.from_numpy(x_train_scaled).float(), torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(x_val_scaled).float(), torch.from_numpy(y_val).long())
    test_dataset = TensorDataset(torch.from_numpy(x_test_scaled).float(), torch.from_numpy(y_test).long())

    batch_size = int(config["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(model)
    print("Total parameters:", sum(param.numel() for param in model.parameters()))
    print("Loss:", criterion)
    print("Optimizer:", type(optimizer).__name__)
    print("Prepared datasets and dataloaders:")
    print("  train batches:", len(train_loader))
    print("  val batches:", len(val_loader))
    print("  test batches:", len(test_loader))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_metric_name = str(config["best_metric"])
    best_val_acc = -1.0
    best_val_loss = float("inf")
    early_stop_patience = int(config["early_stop_patience"])
    early_stop_counter = 0
    num_epochs = int(config["epochs"])
    report_clean_train = bool(config.get("report_clean_train_metrics", True))

    use_ema = bool(config.get("use_ema", False))
    ema = ModelEMA(model, decay=float(config.get("ema_decay", 0.999))) if use_ema else None
    eval_with_ema = bool(config.get("eval_with_ema", False))
    save_ema_ckpt = bool(config.get("save_ema_checkpoint", True))

    for epoch in range(1, num_epochs + 1):
        train_loss_aug, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            config,
            optimizer=optimizer,
            apply_augment=True,
            apply_mixup=True,
            ema=ema,
        )

        if report_clean_train:
            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                config,
                optimizer=None,
                apply_augment=False,
                apply_mixup=False,
                ema=None,
            )
        else:
            train_loss, train_acc = train_loss_aug, float("nan")

        if eval_with_ema and ema is not None:
            ema.apply_shadow(model)
            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                config,
                optimizer=None,
            )
            ema.restore(model)
        else:
            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                config,
                optimizer=None,
            )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = get_current_lr(optimizer)
        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        is_better = val_loss < best_val_loss if best_metric_name == "val_loss" else val_acc > best_val_acc
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            early_stop_counter = 0
            if save_ema_ckpt and ema is not None:
                ema.apply_shadow(model)
                torch.save(model.state_dict(), best_model_path)
                ema.restore(model)
            else:
                torch.save(model.state_dict(), best_model_path)
            print(
                f"Saved best model to {best_model_path} "
                f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})"
            )
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    eval_model_override = config.get("eval_model_path")
    eval_model_path = resolve_training_path(eval_model_override) if eval_model_override else best_model_path.resolve()
    if not eval_model_path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint file: {eval_model_path}")

    best_model = model_builder(
        input_shape=x_train_scaled.shape[1:],
        num_classes=num_classes,
        **config["model_kwargs"],
    ).to(device)
    state_dict = load_checkpoint_state_dict(eval_model_path, map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.eval()

    num_params = sum(param.numel() for param in best_model.parameters())
    num_trainable_params = sum(param.numel() for param in best_model.parameters() if param.requires_grad)
    total_flops, total_macs = profile_model_macs(
        model_builder=model_builder,
        input_shape=x_train_scaled.shape[1:],
        num_classes=num_classes,
        model_kwargs=config["model_kwargs"],
        model_path=eval_model_path,
    )

    print(f"Loaded checkpoint: {eval_model_path}")
    print(f"Parameters: {num_params:,} ({num_params / 1e6:.3f} M)")
    print(f"Trainable parameters: {num_trainable_params:,} ({num_trainable_params / 1e6:.3f} M)")
    if np.isfinite(total_macs):
        print(f"Approx. FLOPs per sample: {total_flops:,.0f}")
        print(f"Approx. MACs per sample: {total_macs:,.0f} ({total_macs / 1e6:.3f} M)")
    else:
        print("Approx. MACs per sample: unavailable")

    test_loss, test_acc = run_epoch(best_model, test_loader, criterion, device, config, optimizer=None)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = best_model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    target_names = [str(item) for item in classes]
    report_text = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\nClassification Report:\n")
    print(report_text)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    history_plot_path = artifacts_dir / "history.png"
    confusion_plot_path = artifacts_dir / "confusion_matrix.png"
    metrics_json_path = artifacts_dir / "metrics.json"
    report_txt_path = artifacts_dir / "classification_report.txt"

    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_loss"], label="train_loss")
        axes[0].plot(history["val_loss"], label="val_loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[1].plot(history["train_acc"], label="train_acc")
        axes[1].plot(history["val_acc"], label="val_acc")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(history_plot_path, dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d", colorbar=False)
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(confusion_plot_path, dpi=160)
        plt.close(fig)
    else:
        print("matplotlib is not installed; skipping history/confusion-matrix image export.")

    metrics = {
        "run_name": run_name,
        "best_model_path": str(best_model_path.resolve()),
        "eval_model_path": str(eval_model_path.resolve()),
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "num_params": num_params,
        "num_trainable_params": num_trainable_params,
        "total_flops": total_flops,
        "total_macs": total_macs,
        "classes": target_names,
    }
    metrics_json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report_txt_path.write_text(report_text, encoding="utf-8")

    if plt is not None:
        print("Saved history plot:", history_plot_path)
        print("Saved confusion matrix:", confusion_plot_path)
    print("Saved metrics JSON:", metrics_json_path)
    print("Saved classification report:", report_txt_path)

    return metrics
