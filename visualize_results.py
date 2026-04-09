from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import json

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ==================== 工具函数 ====================
def set_zoomed_ylim(
    ax, values, pad_ratio=0.20, min_span=0.004, hard_min=0.0, hard_max=1.0
):
    """
    根据数据自动缩放 y 轴，避免总是显示 0~1。
    pad_ratio: 上下留白比例
    min_span : 最小显示跨度，防止数据太接近时图像过扁
    """
    values = np.asarray(values, dtype=float)
    vmin = float(values.min())
    vmax = float(values.max())
    span = max(vmax - vmin, min_span)
    pad = span * pad_ratio
    lower = max(hard_min, vmin - pad)
    upper = min(hard_max, vmax + pad)

    if upper - lower < min_span:
        center = (upper + lower) / 2
        lower = max(hard_min, center - min_span / 2)
        upper = min(hard_max, center + min_span / 2)

    ax.set_ylim(lower, upper)


def annotate_points(ax, xs, ys, fmt="{:.3f}", dy=8, fontsize=10):
    """在点的正上方用像素偏移标注，避免改变坐标轴范围。"""
    for x, y in zip(xs, ys):
        ax.annotate(
            fmt.format(y),
            xy=(x, y),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=True,
        )


def annotate_bars(ax, xs, ys, fmt="{:.2f}s", dy=3, fontsize=10):
    for x, y in zip(xs, ys):
        ax.annotate(
            fmt.format(y),
            xy=(x, y),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


# 加载实验结果
with open("./experiment_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# 创建输出目录
output_dir = Path("./figures")
output_dir.mkdir(exist_ok=True)


# ==================== 1. 学习率对比 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Learning Rate Comparison (SGD, Batch Size=32)", fontsize=16, fontweight="bold"
)

learning_rates = []
final_test_accs = []
final_train_accs = []
training_times = []

for result in results["学习率对比"]["results"]:
    learning_rates.append(result["learning_rate"])
    final_test_accs.append(result["final_test_accuracy"])
    final_train_accs.append(result["final_train_accuracy"])
    training_times.append(result["training_time"])

# 测试准确率
axes[0, 0].plot(
    learning_rates,
    final_test_accs,
    "o-",
    linewidth=2,
    markersize=8,
    label="Test Accuracy",
)
axes[0, 0].set_xlabel("Learning Rate", fontsize=11)
axes[0, 0].set_ylabel("Accuracy", fontsize=11)
axes[0, 0].set_title("Final Test Accuracy vs Learning Rate")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xscale("log")
set_zoomed_ylim(axes[0, 0], final_test_accs, pad_ratio=0.25, min_span=0.006)
annotate_points(axes[0, 0], learning_rates, final_test_accs)

# 训练准确率
axes[0, 1].plot(
    learning_rates,
    final_train_accs,
    "s-",
    linewidth=2,
    markersize=8,
    label="Train Accuracy",
)
axes[0, 1].set_xlabel("Learning Rate", fontsize=11)
axes[0, 1].set_ylabel("Accuracy", fontsize=11)
axes[0, 1].set_title("Final Train Accuracy vs Learning Rate")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xscale("log")
set_zoomed_ylim(axes[0, 1], final_train_accs, pad_ratio=0.25, min_span=0.006)
annotate_points(axes[0, 1], learning_rates, final_train_accs)

# 训练时间
bar_x = np.arange(len(learning_rates))
axes[1, 0].bar(bar_x, training_times, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
axes[1, 0].set_xlabel("Learning Rate", fontsize=11)
axes[1, 0].set_ylabel("Training Time (s)", fontsize=11)
axes[1, 0].set_title("Training Time vs Learning Rate")
axes[1, 0].set_xticks(bar_x)
axes[1, 0].set_xticklabels([f"{lr}" for lr in learning_rates])
axes[1, 0].set_ylim(0, max(training_times) * 1.08)
annotate_bars(axes[1, 0], bar_x, training_times)

# 训练曲线（最好的学习率）
best_idx = int(np.argmax(final_test_accs))
best_result = results["学习率对比"]["results"][best_idx]
epochs = list(range(1, len(best_result["history"]["test_acc"]) + 1))
axes[1, 1].plot(
    epochs, best_result["history"]["train_acc"], "o-", label="Train Acc", linewidth=2
)
axes[1, 1].plot(
    epochs, best_result["history"]["test_acc"], "s-", label="Test Acc", linewidth=2
)
axes[1, 1].set_xlabel("Epoch", fontsize=11)
axes[1, 1].set_ylabel("Accuracy", fontsize=11)
axes[1, 1].set_title(f"Best LR={learning_rates[best_idx]} Training Curve")
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)
curve_vals = best_result["history"]["train_acc"] + best_result["history"]["test_acc"]
set_zoomed_ylim(axes[1, 1], curve_vals, pad_ratio=0.15, min_span=0.01)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(output_dir / "learning_rate_comparison.png", dpi=300, bbox_inches="tight")
print("已保存: learning_rate_comparison.png")
plt.close()


# ==================== 2. 批大小对比 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Batch Size Comparison (SGD, Learning Rate=0.01)", fontsize=16, fontweight="bold"
)

batch_sizes = []
final_test_accs = []
final_train_accs = []
training_times = []

for result in results["批大小对比"]["results"]:
    batch_sizes.append(result["batch_size"])
    final_test_accs.append(result["final_test_accuracy"])
    final_train_accs.append(result["final_train_accuracy"])
    training_times.append(result["training_time"])

# 测试准确率：局部缩放，不再从 0 到 1
axes[0, 0].plot(batch_sizes, final_test_accs, "o-", linewidth=2, markersize=8)
axes[0, 0].set_xlabel("Batch Size", fontsize=11)
axes[0, 0].set_ylabel("Accuracy", fontsize=11)
axes[0, 0].set_title("Final Test Accuracy vs Batch Size")
axes[0, 0].grid(True, alpha=0.3)
set_zoomed_ylim(axes[0, 0], final_test_accs, pad_ratio=0.30, min_span=0.006)
annotate_points(axes[0, 0], batch_sizes, final_test_accs)

# 训练准确率：局部缩放，不再从 0 到 1
axes[0, 1].plot(batch_sizes, final_train_accs, "s-", linewidth=2, markersize=8)
axes[0, 1].set_xlabel("Batch Size", fontsize=11)
axes[0, 1].set_ylabel("Accuracy", fontsize=11)
axes[0, 1].set_title("Final Train Accuracy vs Batch Size")
axes[0, 1].grid(True, alpha=0.3)
set_zoomed_ylim(axes[0, 1], final_train_accs, pad_ratio=0.30, min_span=0.006)
annotate_points(axes[0, 1], batch_sizes, final_train_accs)

# 训练时间
bar_x = np.arange(len(batch_sizes))
axes[1, 0].bar(bar_x, training_times, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
axes[1, 0].set_xlabel("Batch Size", fontsize=11)
axes[1, 0].set_ylabel("Training Time (s)", fontsize=11)
axes[1, 0].set_title("Training Time vs Batch Size")
axes[1, 0].set_xticks(bar_x)
axes[1, 0].set_xticklabels([f"{bs}" for bs in batch_sizes])
axes[1, 0].set_ylim(0, max(training_times) * 1.08)
annotate_bars(axes[1, 0], bar_x, training_times)

# 训练曲线（最好的批大小）
best_idx = int(np.argmax(final_test_accs))
best_result = results["批大小对比"]["results"][best_idx]
epochs = list(range(1, len(best_result["history"]["test_acc"]) + 1))
axes[1, 1].plot(
    epochs, best_result["history"]["train_acc"], "o-", label="Train Acc", linewidth=2
)
axes[1, 1].plot(
    epochs, best_result["history"]["test_acc"], "s-", label="Test Acc", linewidth=2
)
axes[1, 1].set_xlabel("Epoch", fontsize=11)
axes[1, 1].set_ylabel("Accuracy", fontsize=11)
axes[1, 1].set_title(f"Best Batch Size={batch_sizes[best_idx]} Training Curve")
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)
curve_vals = best_result["history"]["train_acc"] + best_result["history"]["test_acc"]
set_zoomed_ylim(axes[1, 1], curve_vals, pad_ratio=0.15, min_span=0.01)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(output_dir / "batch_size_comparison.png", dpi=300, bbox_inches="tight")
print("已保存: batch_size_comparison.png")
plt.close()


# ==================== 3. 优化器对比 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Optimizer Comparison (Learning Rate=0.01, Batch Size=32)",
    fontsize=16,
    fontweight="bold",
)

optimizers = []
final_test_accs = []
final_train_accs = []
training_times = []

for result in results["优化器对比"]["results"]:
    optimizers.append(result["optimizer"])
    final_test_accs.append(result["final_test_accuracy"])
    final_train_accs.append(result["final_train_accuracy"])
    training_times.append(result["training_time"])

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
bar_x = np.arange(len(optimizers))

# 测试准确率
axes[0, 0].bar(bar_x, final_test_accs, color=colors)
axes[0, 0].set_ylabel("Accuracy", fontsize=11)
axes[0, 0].set_title("Final Test Accuracy by Optimizer")
axes[0, 0].set_xticks(bar_x)
axes[0, 0].set_xticklabels(optimizers)
axes[0, 0].set_ylim([0, 1.0])
for x, acc in zip(bar_x, final_test_accs):
    axes[0, 0].annotate(
        f"{acc:.3f}",
        xy=(x, acc),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
    )

# 训练准确率
axes[0, 1].bar(bar_x, final_train_accs, color=colors)
axes[0, 1].set_ylabel("Accuracy", fontsize=11)
axes[0, 1].set_title("Final Train Accuracy by Optimizer")
axes[0, 1].set_xticks(bar_x)
axes[0, 1].set_xticklabels(optimizers)
axes[0, 1].set_ylim([0, 1.0])
for x, acc in zip(bar_x, final_train_accs):
    axes[0, 1].annotate(
        f"{acc:.3f}",
        xy=(x, acc),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
    )

# 训练时间
axes[1, 0].bar(bar_x, training_times, color=colors)
axes[1, 0].set_ylabel("Training Time (s)", fontsize=11)
axes[1, 0].set_title("Training Time by Optimizer")
axes[1, 0].set_xticks(bar_x)
axes[1, 0].set_xticklabels(optimizers)
axes[1, 0].set_ylim(0, max(training_times) * 1.08)
annotate_bars(axes[1, 0], bar_x, training_times)

# 训练曲线对比
for i, (opt, result) in enumerate(zip(optimizers, results["优化器对比"]["results"])):
    epochs = list(range(1, len(result["history"]["test_acc"]) + 1))
    axes[1, 1].plot(
        epochs,
        result["history"]["test_acc"],
        "o-",
        label=opt,
        linewidth=2,
        color=colors[i],
    )

axes[1, 1].set_xlabel("Epoch", fontsize=11)
axes[1, 1].set_ylabel("Test Accuracy", fontsize=11)
axes[1, 1].set_title("Test Accuracy Curves by Optimizer")
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)
all_opt_curve_vals = []
for result in results["优化器对比"]["results"]:
    all_opt_curve_vals.extend(result["history"]["test_acc"])
set_zoomed_ylim(axes[1, 1], all_opt_curve_vals, pad_ratio=0.15, min_span=0.01)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(output_dir / "optimizer_comparison.png", dpi=300, bbox_inches="tight")
print("已保存: optimizer_comparison.png")
plt.close()


# ==================== 4. 综合对比 ====================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Comprehensive Comparison of All Experiments", fontsize=16, fontweight="bold"
)

# 学习率实验
lr_accs = [r["final_test_accuracy"] for r in results["学习率对比"]["results"]]
lr_labels = [f"LR={r['learning_rate']}" for r in results["学习率对比"]["results"]]
bar_x = np.arange(len(lr_accs))
axes[0].bar(bar_x, lr_accs, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
axes[0].set_ylabel("Test Accuracy", fontsize=11)
axes[0].set_title("Learning Rate Comparison")
axes[0].set_xticks(bar_x)
axes[0].set_xticklabels(lr_labels, rotation=15)
axes[0].set_ylim([0, 1.0])
for x, acc in zip(bar_x, lr_accs):
    axes[0].annotate(
        f"{acc:.3f}",
        xy=(x, acc),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
    )

# 批大小实验
bs_accs = [r["final_test_accuracy"] for r in results["批大小对比"]["results"]]
bs_labels = [f"BS={r['batch_size']}" for r in results["批大小对比"]["results"]]
bar_x = np.arange(len(bs_accs))
axes[1].bar(bar_x, bs_accs, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
axes[1].set_ylabel("Test Accuracy", fontsize=11)
axes[1].set_title("Batch Size Comparison")
axes[1].set_xticks(bar_x)
axes[1].set_xticklabels(bs_labels)
axes[1].set_ylim([0, 1.0])
for x, acc in zip(bar_x, bs_accs):
    axes[1].annotate(
        f"{acc:.3f}",
        xy=(x, acc),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
    )

# 优化器实验
opt_accs = [r["final_test_accuracy"] for r in results["优化器对比"]["results"]]
opt_labels = [r["optimizer"] for r in results["优化器对比"]["results"]]
bar_x = np.arange(len(opt_accs))
axes[2].bar(bar_x, opt_accs, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
axes[2].set_ylabel("Test Accuracy", fontsize=11)
axes[2].set_title("Optimizer Comparison")
axes[2].set_xticks(bar_x)
axes[2].set_xticklabels(opt_labels)
axes[2].set_ylim([0, 1.0])
for x, acc in zip(bar_x, opt_accs):
    axes[2].annotate(
        f"{acc:.3f}",
        xy=(x, acc),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
    )

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(output_dir / "comprehensive_comparison.png", dpi=300, bbox_inches="tight")
print("已保存: comprehensive_comparison.png")
plt.close()

print("\n所有图表已生成完成！")
