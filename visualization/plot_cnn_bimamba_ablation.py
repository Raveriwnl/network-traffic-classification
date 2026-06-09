from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib import font_manager


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "ablation"

METRIC_SPECS = [
    ("accuracy", "准确率"),
    ("precision", "精确率"),
    ("recall", "召回率"),
    ("f1", "F1值"),
]

MODEL_COLORS = {
    "CMamba": "#c97c5d",
    "CNN-Mamba-Attention": "#6f91b5",
    "CNN-BiMamba-AvgPool": "#7fb7ad",
    "CNN-BiLSTM-Attention": "#9b88c2",
    "Mamba-Attention": "#d98c8c",
    "CNN-Attention": "#d7b46a",
}


ABLATION_RESULTS = [
    {
        "model": "CMamba",
        "variant": "Final model",
        "accuracy": 0.9417,
        "precision": 0.9440,
        "recall": 0.9417,
        "f1": 0.9417,
        "params_m": 0.316073,
        "macs_m": 77.134944,
        "source": "cnn_bimamba_attention_mamba rerun",
    },
    {
        "model": "CNN-Mamba-Attention",
        "variant": "Single-direction Mamba",
        "accuracy": 0.9271,
        "precision": 0.9293,
        "recall": 0.9271,
        "f1": 0.9270,
        "params_m": 0.102697,
        "macs_m": 21.678944,
        "source": "training_cnn_mamba_attention.ipynb",
    },
    {
        "model": "CNN-BiMamba-AvgPool",
        "variant": "AvgPool replaces attention",
        "accuracy": 0.9146,
        "precision": 0.9169,
        "recall": 0.9146,
        "f1": 0.9147,
        "params_m": 0.179112,
        "macs_m": 41.190912,
        "source": "training_cnn_bimamba_avgpool.ipynb",
    },
    {
        "model": "CNN-BiLSTM-Attention",
        "variant": "BiLSTM replaces BiMamba",
        "accuracy": 0.9083,
        "precision": 0.9098,
        "recall": 0.9083,
        "f1": 0.9083,
        "params_m": 0.071849,
        "macs_m": 7.263072,
        "source": "training_cnn_bilstm_attention.ipynb",
    },
    {
        "model": "Mamba-Attention",
        "variant": "No CNN stem",
        "accuracy": 0.8750,
        "precision": 0.8815,
        "recall": 0.8750,
        "f1": 0.8747,
        "params_m": 0.174333,
        "macs_m": 42.846944,
        "source": "training_mamba_attention.ipynb",
    },
    {
        "model": "CNN-Attention",
        "variant": "No Mamba block",
        "accuracy": 0.8729,
        "precision": 0.8751,
        "recall": 0.8729,
        "f1": 0.8730,
        "params_m": 0.021353,
        "macs_m": 1.086944,
        "source": "training_cnn_attention.ipynb",
    },
]


def build_colors(names):
    return [MODEL_COLORS[name] for name in names]


def build_legend_handles(names):
    return [Patch(facecolor=MODEL_COLORS[name], edgecolor="none", label=name) for name in names]


def configure_matplotlib_fonts():
    candidate_font_files = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]

    selected_name = None
    for font_file in candidate_font_files:
        font_path = Path(font_file)
        if not font_path.exists():
            continue
        font_manager.fontManager.addfont(str(font_path))
        selected_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        if selected_name:
            break

    if selected_name is None:
        selected_name = "DejaVu Sans"

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [selected_name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def annotate_bars(ax, bars, value_format, offset=0.003):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            value_format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def annotate_horizontal_bars(ax, bars, value_format, offset):
    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            value_format(value),
            va="center",
            fontsize=9,
        )


def save_single_metric_figures(results, output_dir):
    names = [item["model"] for item in results]
    colors = build_colors(names)
    x = np.arange(len(results))
    file_paths = []
    legend_handles = build_legend_handles(names)

    for key, label in METRIC_SPECS:
        values = [item[key] for item in results]

        fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
        fig.patch.set_facecolor("white")
        bars = ax.bar(x, values, color=colors, edgecolor="none", width=0.64)
        annotate_bars(ax, bars, lambda value: f"{value:.4f}", offset=0.002)

        ax.set_title(f"CMamba 消融实验：{label}")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(0.84, 0.97)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(
            handles=legend_handles,
            title="变体模型",
            loc="upper right",
            frameon=True,
            fancybox=True,
            framealpha=0.92,
            edgecolor="#cbd5e1",
            fontsize=8.5,
            title_fontsize=9,
        )

        png_path = output_dir / f"cmamba_ablation_{key}.png"
        svg_path = output_dir / f"cmamba_ablation_{key}.svg"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)
        file_paths.extend([png_path, svg_path])

    return file_paths


def save_metric_grid_figure(results, output_dir):
    names = [item["model"] for item in results]
    colors = build_colors(names)
    x = np.arange(len(results))
    legend_handles = build_legend_handles(names)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax, (key, label) in zip(axes.flat, METRIC_SPECS):
        values = [item[key] for item in results]
        bars = ax.bar(x, values, color=colors, edgecolor="none", width=0.64)
        annotate_bars(ax, bars, lambda value: f"{value:.4f}", offset=0.0018)

        ax.set_title(label, fontsize=11.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
        ax.set_ylim(0.84, 0.97)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0, 0].set_ylabel("指标值")
    axes[1, 0].set_ylabel("指标值")

    axes[0, 1].legend(
        handles=legend_handles,
        title="变体模型",
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor="#cbd5e1",
        fontsize=8.2,
        title_fontsize=8.8,
    )

    fig.suptitle("CMamba 消融实验四指标汇总", fontsize=13.5, fontweight="bold")

    png_path = output_dir / "cmamba_ablation_metrics_grid.png"
    svg_path = output_dir / "cmamba_ablation_metrics_grid.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, svg_path]


def plot_ablation(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_fonts()

    metric_paths = save_single_metric_figures(results, OUTPUT_DIR)
    metric_grid_paths = save_metric_grid_figure(results, OUTPUT_DIR)

    names = [item["model"] for item in results]
    colors = build_colors(names)
    legend_handles = build_legend_handles(names)

    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib_fonts()
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.2), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax_metrics = axes[0, 0]
    ax_metrics.axis("off")
    ax_metrics.set_title("指标图已单独保存")
    metric_lines = [
        "原左上角合并指标图已拆分为四张单独图片，并额外导出了一张四指标合图：",
        "",
        "- cmamba_ablation_accuracy.png/.svg",
        "- cmamba_ablation_precision.png/.svg",
        "- cmamba_ablation_recall.png/.svg",
        "- cmamba_ablation_f1.png/.svg",
        "- cmamba_ablation_metrics_grid.png/.svg",
        "",
        "每张图都保持同一套变体配色，CMamba 继续作为主模型高亮显示。",
    ]
    ax_metrics.text(
        0.02,
        0.95,
        "\n".join(metric_lines),
        transform=ax_metrics.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        color="#334155",
        linespacing=1.6,
    )
    ax_metrics.legend(
        handles=legend_handles,
        title="变体模型",
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor="#cbd5e1",
        fontsize=8.5,
        title_fontsize=9,
    )

    ax_params = axes[0, 1]
    params = [item["params_m"] for item in results]
    param_bars = ax_params.barh(names, params, color=colors, edgecolor="none")
    annotate_horizontal_bars(ax_params, param_bars, lambda value: f"{value:.3f}M", offset=0.006)
    ax_params.set_title("参数量对比")
    ax_params.set_xlabel("参数量 (M)")
    ax_params.invert_yaxis()
    ax_params.grid(axis="x", linestyle="--", alpha=0.35)

    ax_macs = axes[1, 0]
    macs = [item["macs_m"] for item in results]
    mac_bars = ax_macs.barh(names, macs, color=colors, edgecolor="none")
    annotate_horizontal_bars(ax_macs, mac_bars, lambda value: f"{value:.2f}M", offset=1.0)
    ax_macs.set_title("单样本推理开销")
    ax_macs.set_xlabel("MACs (M)")
    ax_macs.invert_yaxis()
    ax_macs.grid(axis="x", linestyle="--", alpha=0.35)

    ax_tradeoff = axes[1, 1]
    for item, color in zip(results, colors):
        marker_size = 250 if item["model"] == "CMamba" else 150
        ax_tradeoff.scatter(
            item["macs_m"],
            item["accuracy"],
            s=marker_size,
            color=color,
            alpha=0.9,
            edgecolors="white",
            linewidths=1.5,
        )
        ax_tradeoff.annotate(
            item["model"],
            (item["macs_m"], item["accuracy"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    ax_tradeoff.set_title("准确率与 MACs 权衡")
    ax_tradeoff.set_xlabel("MACs (M)")
    ax_tradeoff.set_ylabel("准确率")
    ax_tradeoff.set_ylim(0.86, 0.95)
    ax_tradeoff.grid(linestyle="--", alpha=0.35)

    fig.suptitle("CMamba 在华为流量分类任务上的消融实验", fontsize=15.5, fontweight="bold")

    note = (
        "CMamba 使用 cnn_bimamba_attention_mamba 最终复评结果；"
        "其余变体性能指标来自 notebook 结果汇总，参数量与 MACs 来自各自 notebook 输出。"
    )
    fig.text(0.5, 0.012, note, ha="center", fontsize=8.8, color="#4b5563")

    png_path = OUTPUT_DIR / "cmamba_ablation_study.png"
    svg_path = OUTPUT_DIR / "cmamba_ablation_study.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path, metric_paths, metric_grid_paths


def main():
    png_path, svg_path, metric_paths, metric_grid_paths = plot_ablation(ABLATION_RESULTS)
    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")
    for path in metric_paths:
        print(f"Saved metric figure: {path}")
    for path in metric_grid_paths:
        print(f"Saved metric grid: {path}")


if __name__ == "__main__":
    main()