"""
实验结果可视化
"""

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载实验结果
with open('./experiment_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# 创建输出目录
output_dir = Path('./figures')
output_dir.mkdir(exist_ok=True)

# ==================== 1. 学习率对比 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Learning Rate Comparison (SGD, Batch Size=32)', fontsize=16, fontweight='bold')

learning_rates = []
final_test_accs = []
final_train_accs = []
training_times = []

for result in results['学习率对比']['results']:
    learning_rates.append(result['learning_rate'])
    final_test_accs.append(result['final_test_accuracy'])
    final_train_accs.append(result['final_train_accuracy'])
    training_times.append(result['training_time'])

# 测试准确率
axes[0, 0].plot(learning_rates, final_test_accs, 'o-', linewidth=2, markersize=8, label='Test Accuracy')
axes[0, 0].set_xlabel('Learning Rate', fontsize=11)
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].set_title('Final Test Accuracy vs Learning Rate')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xscale('log')
for i, (lr, acc) in enumerate(zip(learning_rates, final_test_accs)):
    axes[0, 0].text(lr, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)

# 训练准确率
axes[0, 1].plot(learning_rates, final_train_accs, 's-', linewidth=2, markersize=8, label='Train Accuracy')
axes[0, 1].set_xlabel('Learning Rate', fontsize=11)
axes[0, 1].set_ylabel('Accuracy', fontsize=11)
axes[0, 1].set_title('Final Train Accuracy vs Learning Rate')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xscale('log')
for i, (lr, acc) in enumerate(zip(learning_rates, final_train_accs)):
    axes[0, 1].text(lr, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)

# 训练时间
axes[1, 0].bar(range(len(learning_rates)), training_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1, 0].set_xlabel('Learning Rate', fontsize=11)
axes[1, 0].set_ylabel('Training Time (s)', fontsize=11)
axes[1, 0].set_title('Training Time vs Learning Rate')
axes[1, 0].set_xticks(range(len(learning_rates)))
axes[1, 0].set_xticklabels([f'{lr}' for lr in learning_rates])
for i, t in enumerate(training_times):
    axes[1, 0].text(i, t + 0.2, f'{t:.2f}s', ha='center', fontsize=10)

# 训练曲线（最好的学习率）
best_idx = np.argmax(final_test_accs)
best_result = results['学习率对比']['results'][best_idx]
epochs = list(range(1, len(best_result['history']['test_acc']) + 1))
axes[1, 1].plot(epochs, best_result['history']['train_acc'], 'o-', label='Train Acc', linewidth=2)
axes[1, 1].plot(epochs, best_result['history']['test_acc'], 's-', label='Test Acc', linewidth=2)
axes[1, 1].set_xlabel('Epoch', fontsize=11)
axes[1, 1].set_ylabel('Accuracy', fontsize=11)
axes[1, 1].set_title(f'Best LR={learning_rates[best_idx]} Training Curve')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'learning_rate_comparison.png', dpi=300, bbox_inches='tight')
print("已保存: learning_rate_comparison.png")
plt.close()

# ==================== 2. 批大小对比 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Batch Size Comparison (SGD, Learning Rate=0.01)', fontsize=16, fontweight='bold')

batch_sizes = []
final_test_accs = []
final_train_accs = []
training_times = []

for result in results['批大小对比']['results']:
    batch_sizes.append(result['batch_size'])
    final_test_accs.append(result['final_test_accuracy'])
    final_train_accs.append(result['final_train_accuracy'])
    training_times.append(result['training_time'])

# 测试准确率
axes[0, 0].plot(batch_sizes, final_test_accs, 'o-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Batch Size', fontsize=11)
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].set_title('Final Test Accuracy vs Batch Size')
axes[0, 0].grid(True, alpha=0.3)
for i, (bs, acc) in enumerate(zip(batch_sizes, final_test_accs)):
    axes[0, 0].text(bs, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)

# 训练准确率
axes[0, 1].plot(batch_sizes, final_train_accs, 's-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Batch Size', fontsize=11)
axes[0, 1].set_ylabel('Accuracy', fontsize=11)
axes[0, 1].set_title('Final Train Accuracy vs Batch Size')
axes[0, 1].grid(True, alpha=0.3)
for i, (bs, acc) in enumerate(zip(batch_sizes, final_train_accs)):
    axes[0, 1].text(bs, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)

# 训练时间
axes[1, 0].bar(range(len(batch_sizes)), training_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1, 0].set_xlabel('Batch Size', fontsize=11)
axes[1, 0].set_ylabel('Training Time (s)', fontsize=11)
axes[1, 0].set_title('Training Time vs Batch Size')
axes[1, 0].set_xticks(range(len(batch_sizes)))
axes[1, 0].set_xticklabels([f'{bs}' for bs in batch_sizes])
for i, t in enumerate(training_times):
    axes[1, 0].text(i, t + 0.2, f'{t:.2f}s', ha='center', fontsize=10)

# 训练曲线（最好的批大小）
best_idx = np.argmax(final_test_accs)
best_result = results['批大小对比']['results'][best_idx]
epochs = list(range(1, len(best_result['history']['test_acc']) + 1))
axes[1, 1].plot(epochs, best_result['history']['train_acc'], 'o-', label='Train Acc', linewidth=2)
axes[1, 1].plot(epochs, best_result['history']['test_acc'], 's-', label='Test Acc', linewidth=2)
axes[1, 1].set_xlabel('Epoch', fontsize=11)
axes[1, 1].set_ylabel('Accuracy', fontsize=11)
axes[1, 1].set_title(f'Best Batch Size={batch_sizes[best_idx]} Training Curve')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'batch_size_comparison.png', dpi=300, bbox_inches='tight')
print("已保存: batch_size_comparison.png")
plt.close()

# ==================== 3. 优化器对比 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Optimizer Comparison (Learning Rate=0.01, Batch Size=32)', fontsize=16, fontweight='bold')

optimizers = []
final_test_accs = []
final_train_accs = []
training_times = []

for result in results['优化器对比']['results']:
    optimizers.append(result['optimizer'])
    final_test_accs.append(result['final_test_accuracy'])
    final_train_accs.append(result['final_train_accuracy'])
    training_times.append(result['training_time'])

# 测试准确率
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
axes[0, 0].bar(range(len(optimizers)), final_test_accs, color=colors)
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].set_title('Final Test Accuracy by Optimizer')
axes[0, 0].set_xticks(range(len(optimizers)))
axes[0, 0].set_xticklabels(optimizers)
axes[0, 0].set_ylim([0, 1.0])
for i, acc in enumerate(final_test_accs):
    axes[0, 0].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)

# 训练准确率
axes[0, 1].bar(range(len(optimizers)), final_train_accs, color=colors)
axes[0, 1].set_ylabel('Accuracy', fontsize=11)
axes[0, 1].set_title('Final Train Accuracy by Optimizer')
axes[0, 1].set_xticks(range(len(optimizers)))
axes[0, 1].set_xticklabels(optimizers)
axes[0, 1].set_ylim([0, 1.0])
for i, acc in enumerate(final_train_accs):
    axes[0, 1].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)

# 训练时间
axes[1, 0].bar(range(len(optimizers)), training_times, color=colors)
axes[1, 0].set_ylabel('Training Time (s)', fontsize=11)
axes[1, 0].set_title('Training Time by Optimizer')
axes[1, 0].set_xticks(range(len(optimizers)))
axes[1, 0].set_xticklabels(optimizers)
for i, t in enumerate(training_times):
    axes[1, 0].text(i, t + 0.2, f'{t:.2f}s', ha='center', fontsize=10)

# 训练曲线对比
for i, (opt, result) in enumerate(zip(optimizers, results['优化器对比']['results'])):
    epochs = list(range(1, len(result['history']['test_acc']) + 1))
    axes[1, 1].plot(epochs, result['history']['test_acc'], 'o-', label=opt, linewidth=2, color=colors[i])

axes[1, 1].set_xlabel('Epoch', fontsize=11)
axes[1, 1].set_ylabel('Test Accuracy', fontsize=11)
axes[1, 1].set_title('Test Accuracy Curves by Optimizer')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'optimizer_comparison.png', dpi=300, bbox_inches='tight')
print("已保存: optimizer_comparison.png")
plt.close()

# ==================== 4. 综合对比 ====================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Comprehensive Comparison of All Experiments', fontsize=16, fontweight='bold')

# 学习率实验
lr_accs = [r['final_test_accuracy'] for r in results['学习率对比']['results']]
lr_labels = [f"LR={r['learning_rate']}" for r in results['学习率对比']['results']]
axes[0].bar(range(len(lr_accs)), lr_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0].set_ylabel('Test Accuracy', fontsize=11)
axes[0].set_title('Learning Rate Comparison')
axes[0].set_xticks(range(len(lr_labels)))
axes[0].set_xticklabels(lr_labels, rotation=15)
axes[0].set_ylim([0, 1.0])
for i, acc in enumerate(lr_accs):
    axes[0].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)

# 批大小实验
bs_accs = [r['final_test_accuracy'] for r in results['批大小对比']['results']]
bs_labels = [f"BS={r['batch_size']}" for r in results['批大小对比']['results']]
axes[1].bar(range(len(bs_accs)), bs_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1].set_ylabel('Test Accuracy', fontsize=11)
axes[1].set_title('Batch Size Comparison')
axes[1].set_xticks(range(len(bs_labels)))
axes[1].set_xticklabels(bs_labels)
axes[1].set_ylim([0, 1.0])
for i, acc in enumerate(bs_accs):
    axes[1].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)

# 优化器实验
opt_accs = [r['final_test_accuracy'] for r in results['优化器对比']['results']]
opt_labels = [r['optimizer'] for r in results['优化器对比']['results']]
axes[2].bar(range(len(opt_accs)), opt_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[2].set_ylabel('Test Accuracy', fontsize=11)
axes[2].set_title('Optimizer Comparison')
axes[2].set_xticks(range(len(opt_labels)))
axes[2].set_xticklabels(opt_labels)
axes[2].set_ylim([0, 1.0])
for i, acc in enumerate(opt_accs):
    axes[2].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print("已保存: comprehensive_comparison.png")
plt.close()

print("\n所有图表已生成完成！")
