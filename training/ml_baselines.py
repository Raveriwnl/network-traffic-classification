from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler


CANDIDATE_FEATURES = [
    "min_time_ms",
    "max_time_ms",
    "flow_duration_ms",
    "num_packets",
    "avg_packet_size_bytes",
    "total_length_bytes",
    "total_payload_bytes",
    "uplink_pkt_count",
    "downlink_pkt_count",
]

FINAL_FEATURES = [
    "flow_duration_ms",
    "num_packets",
    "avg_packet_size_bytes",
    "total_payload_bytes",
]

MODEL_SEARCH_SPACES = {
    "gaussian_nb": [
        {"var_smoothing": 1e-12},
        {"var_smoothing": 1e-10},
        {"var_smoothing": 1e-9},
        {"var_smoothing": 1e-8},
        {"var_smoothing": 1e-7},
    ],
    "random_forest": [
        {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
        {
            "n_estimators": 400,
            "max_depth": 18,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
        {
            "n_estimators": 500,
            "max_depth": 24,
            "min_samples_leaf": 2,
            "max_features": 0.8,
            "class_weight": "balanced_subsample",
        },
    ],
    "xgboost": [
        {
            "n_estimators": 120,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 180,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 120,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 3,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.08,
            "subsample": 1.0,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "reg_lambda": 1.5,
        },
    ],
}


def resolve_repo_root(start_path: str | Path | None = None) -> Path:
    path = Path(start_path) if start_path is not None else Path.cwd()
    if path.is_file():
        path = path.parent

    for candidate in [path, *path.parents]:
        if (candidate / "datasets").exists() and (candidate / "training").exists():
            return candidate

    raise FileNotFoundError(
        "Could not resolve repository root containing both 'datasets' and 'training'."
    )


def get_huawei_raw_dir(repo_root: str | Path | None = None) -> Path:
    root = resolve_repo_root(repo_root)
    raw_dir = root / "datasets" / "raw" / "huawei"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Huawei raw dataset directory not found: {raw_dir}")
    return raw_dir


def build_flow_feature_table(repo_root: str | Path | None = None) -> tuple[pd.DataFrame, list[str]]:
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

        for flow_id, grouped_df in flow_df.groupby("flow_id", sort=True):
            times = grouped_df["arrive_time"].to_numpy(dtype=np.float64)
            packet_lengths = grouped_df["pkt_len"].to_numpy(dtype=np.float64)
            direction = grouped_df["direction"].to_numpy(dtype=np.int8)

            min_time = float(times.min())
            max_time = float(times.max())
            total_length = float(packet_lengths.sum())

            rows.append(
                {
                    "flow_id": int(flow_id),
                    "label": label_map.get(int(flow_id), csv_file.stem.split("-")[-1]),
                    "source_file": csv_file.name,
                    "min_time_ms": min_time,
                    "max_time_ms": max_time,
                    "flow_duration_ms": max_time - min_time,
                    "num_packets": int(len(grouped_df)),
                    "avg_packet_size_bytes": float(packet_lengths.mean()),
                    "total_length_bytes": total_length,
                    # The Huawei CSV does not expose a separate payload field, so packet-length sum is the closest proxy.
                    "total_payload_bytes": total_length,
                    "uplink_pkt_count": int((direction == 1).sum()),
                    "downlink_pkt_count": int((direction == 0).sum()),
                }
            )

    feature_df = pd.DataFrame(rows).sort_values(["label", "flow_id"]).reset_index(drop=True)
    return feature_df, CANDIDATE_FEATURES.copy()


def compute_feature_analysis(
    feature_df: pd.DataFrame,
    candidate_features: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    candidate_features = candidate_features or CANDIDATE_FEATURES
    grouped = list(feature_df.groupby("label", sort=True))

    class_names = sorted(feature_df["label"].astype(str).unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(class_names)}
    y = feature_df["label"].map(label_to_id).to_numpy(dtype=np.int64)
    X = feature_df[candidate_features].to_numpy(dtype=np.float32)

    forest = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    forest.fit(X, y)

    corr = feature_df[candidate_features].corr().abs()
    analysis_rows: list[dict[str, object]] = []
    for feature_name, importance in zip(candidate_features, forest.feature_importances_):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_stat, p_value = f_oneway(
                *[group[feature_name].to_numpy(dtype=np.float64) for _, group in grouped]
            )

        corr_without_self = corr.loc[feature_name].drop(labels=[feature_name])
        top_corr_feature = corr_without_self.idxmax()
        top_corr_value = float(corr_without_self.max())
        analysis_rows.append(
            {
                "feature": feature_name,
                "rf_importance": float(importance),
                "f_stat": float(f_stat),
                "p_value": float(p_value),
                "top_corr_feature": top_corr_feature,
                "top_abs_corr": top_corr_value,
            }
        )

    return pd.DataFrame(analysis_rows).sort_values(
        ["rf_importance", "f_stat"], ascending=[False, False]
    ).reset_index(drop=True)


def paper_feature_elimination(
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    equal_ratio = float(
        (feature_df["total_length_bytes"] == feature_df["total_payload_bytes"]).mean()
    )
    total_corr = float(
        feature_df[["total_length_bytes", "total_payload_bytes"]].corr().iloc[0, 1]
    )

    elimination_rows: list[dict[str, object]] = []
    for feature_name in CANDIDATE_FEATURES:
        if feature_name in {"min_time_ms", "max_time_ms"}:
            elimination_rows.append(
                {
                    "feature": feature_name,
                    "action": "drop",
                    "reason": "Paper rule: remove explicit timestamps because flow_duration_ms already captures elapsed time.",
                }
            )
        elif feature_name in {"uplink_pkt_count", "downlink_pkt_count"}:
            elimination_rows.append(
                {
                    "feature": feature_name,
                    "action": "drop",
                    "reason": "Paper rule: remove direction counts and keep num_packets as the aggregate packet-count feature.",
                }
            )
        elif feature_name == "total_length_bytes":
            elimination_rows.append(
                {
                    "feature": feature_name,
                    "action": "drop",
                    "reason": (
                        "Paper correlation rule: redundant with total_payload_bytes "
                        f"(corr={total_corr:.4f}, equal_ratio={equal_ratio:.2%})."
                    ),
                }
            )
        else:
            elimination_rows.append(
                {
                    "feature": feature_name,
                    "action": "keep",
                    "reason": "Retained by the paper-inspired elimination policy.",
                }
            )

    kept_columns = ["flow_id", "label", "source_file", *FINAL_FEATURES]
    filtered_df = feature_df.loc[:, kept_columns].copy()
    elimination_df = pd.DataFrame(elimination_rows)
    return filtered_df, elimination_df, FINAL_FEATURES.copy()


def split_scale_dataset(
    feature_df: pd.DataFrame,
    feature_names: list[str] | None = None,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, object]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    feature_names = feature_names or FINAL_FEATURES
    class_names = sorted(feature_df["label"].astype(str).unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(class_names)}
    y = feature_df["label"].map(label_to_id).to_numpy(dtype=np.int64)
    X = feature_df[feature_names].to_numpy(dtype=np.float32)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=float(test_ratio),
        random_state=seed,
        stratify=y,
    )

    val_within_temp = float(val_ratio) / (1.0 - float(test_ratio))
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_within_temp,
        random_state=seed,
        stratify=y_temp,
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "class_names": class_names,
        "label_to_id": label_to_id,
        "feature_names": list(feature_names),
        "scaler": scaler,
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
    }


def build_model(
    model_key: str,
    params: dict[str, object],
    num_classes: int,
    seed: int = 42,
):
    if model_key == "gaussian_nb":
        return GaussianNB(**params)

    if model_key == "random_forest":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    if model_key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for the XGBoost notebook. Install it in the active environment first."
            ) from exc

        return XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            **params,
        )

    raise ValueError(f"Unsupported model_key: {model_key}")


def select_best_model(
    model_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    seed: int = 42,
    search_space: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object], object]:
    search_space = search_space or MODEL_SEARCH_SPACES[model_key]

    results: list[dict[str, object]] = []
    best_model = None
    best_params: dict[str, object] | None = None
    best_key = (-np.inf, -np.inf)

    for idx, params in enumerate(search_space, start=1):
        model = build_model(model_key=model_key, params=params, num_classes=num_classes, seed=seed)
        if model_key == "xgboost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        val_macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
        val_accuracy = float(accuracy_score(y_val, y_pred))
        results.append(
            {
                "trial": idx,
                "params": json.dumps(params, ensure_ascii=True, sort_keys=True),
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_accuracy,
            }
        )

        score_key = (val_macro_f1, val_accuracy)
        if score_key > best_key:
            best_key = score_key
            best_params = params
            best_model = model

    if best_params is None or best_model is None:
        raise ValueError(f"No model was trained for model_key={model_key}")

    results_df = pd.DataFrame(results).sort_values(
        ["val_macro_f1", "val_accuracy"], ascending=False
    ).reset_index(drop=True)
    return results_df, best_params, best_model


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    y_pred = model.predict(X)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "report": classification_report(
            y,
            y_pred,
            labels=labels,
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
        "confusion_matrix": cm,
        "predictions": y_pred,
    }


def plot_feature_importance(analysis_df: pd.DataFrame, title: str) -> tuple[plt.Figure, plt.Axes]:
    sorted_df = analysis_df.sort_values("rf_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_df["feature"], sorted_df["rf_importance"], color="#366092")
    ax.set_xlabel("Random Forest Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_correlation_matrix(
    feature_df: pd.DataFrame,
    feature_names: list[str],
    title: str,
) -> tuple[plt.Figure, plt.Axes]:
    corr = feature_df[feature_names].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(feature_names)), labels=feature_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(feature_names)), labels=feature_names)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


def plot_confusion(
    confusion: np.ndarray,
    class_names: list[str],
    title: str,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax