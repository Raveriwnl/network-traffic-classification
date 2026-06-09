from huawei_training_runner import DEFAULT_LOG1P_FEATURES, run_training


DEFAULT_CONFIG = {
    "seed": 42,
    "cuda_device": None,
    "dataset_path": "../datasets/processed/huawei/huawei_5s_1000bins_features.npz",
    "model_module": "bimobilemamba_attention",
    "model_builder": "build_bimobilemamba_attention_model",
    "model_kwargs": {
        "d_model": 64,
        "num_mamba_layers": 1,
        "mamba_d_state": 16,
        "mamba_d_conv": 5,
        "mamba_expand": 2,
        "mamba_dropout": 0.20,
        "input_dropout": 0.10,
        "head_dropout": 0.35,
        "drop_path_rate": 0.10,
        "pool_dropout": 0.10,
    },
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
    "best_metric": "val_loss",
    "report_clean_train_metrics": True,
    "model_save_name": "huawei_bimobilemamba_attention_best.pt",
    "eval_model_path": None,
    "artifact_dir": "experiment_results/bimobilemamba_attention",
    "log1p_features": DEFAULT_LOG1P_FEATURES,
}


def main():
    run_training(
        default_config=DEFAULT_CONFIG,
        description="Train Huawei BiMobileMamba-Attention model without a CNN stem.",
        run_name="bimobilemamba_attention",
    )


if __name__ == "__main__":
    main()