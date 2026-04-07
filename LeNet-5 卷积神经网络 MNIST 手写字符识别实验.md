# LeNet-5 卷积神经网络 MNIST 手写字符识别实验

## 项目概述

本项目实现了经典的 LeNet-5 卷积神经网络，用于识别 MNIST 手写数字数据集。通过采用控制变量法，对学习率、批大小和优化器三种关键超参数进行了系统的对比实验，探究不同参数设置对模型性能的影响。

## 文件结构

```
lenet5_experiment/
├── LeNet5_MNIST_Experiment_Report.pdf    # 完整的实验报告（PDF格式）
├── LeNet5_MindSpore_Code.zip             # 核心代码打包文件
├── lenet5_base.py                        # LeNet-5网络定义和基础训练代码
├── experiment_runner_v2.py               # MindSpore版本的参数对比实验脚本
├── experiment_pytorch.py                 # PyTorch版本的参数对比实验脚本（可选）
├── visualize_results.py                  # 实验结果可视化脚本
├── experiment_results.json               # 实验结果数据（JSON格式）
├── figures/                              # 生成的图表目录
│   ├── learning_rate_comparison.png
│   ├── batch_size_comparison.png
│   ├── optimizer_comparison.png
│   └── comprehensive_comparison.png
├── data/                                 # MNIST数据集目录
└── README.md                             # 本文件
```

## 核心代码说明

### 1. lenet5_base.py
**功能：** 定义 LeNet-5 网络结构，实现基础的数据加载、模型训练和评估功能。

**主要类和函数：**
- `LeNet5(nn.Cell)`: LeNet-5 网络类，包含卷积层、池化层和全连接层
- `create_mnist_dataset()`: 创建 MNIST 数据集加载器
- `train_model()`: 模型训练函数
- `evaluate_model()`: 模型评估函数

**使用方法：**
```bash
python3 lenet5_base.py
```

### 2. experiment_runner_v2.py
**功能：** 使用 MindSpore 框架运行参数对比实验，包括学习率、批大小和优化器的三组实验。

**实验设计：**
- **学习率对比**：固定 SGD 优化器和批大小 32，测试学习率 0.001、0.01、0.1
- **批大小对比**：固定 SGD 优化器和学习率 0.01，测试批大小 16、32、64
- **优化器对比**：固定学习率 0.01 和批大小 32，测试 SGD、Adam、RMSprop

**使用方法：**
```bash
python3 experiment_runner_v2.py
```

**输出：** `experiment_results.json` 文件包含所有实验的详细结果

### 3. experiment_pytorch.py
**功能：** 使用 PyTorch 框架实现相同的参数对比实验（作为备选方案）。

**使用方法：**
```bash
python3 experiment_pytorch.py
```

### 4. visualize_results.py
**功能：** 从实验结果 JSON 文件生成可视化图表。

**生成的图表：**
1. `learning_rate_comparison.png` - 学习率对比的四个子图（测试准确率、训练准确率、训练时间、最优学习率的训练曲线）
2. `batch_size_comparison.png` - 批大小对比的四个子图
3. `optimizer_comparison.png` - 优化器对比的四个子图
4. `comprehensive_comparison.png` - 综合对比的三个子图

**使用方法：**
```bash
python3 visualize_results.py
```

## 网络结构详解

LeNet-5 是一个 7 层的卷积神经网络，具体结构如下：

| 层级 | 类型 | 参数 | 输出尺寸 |
|------|------|------|---------|
| 输入 | - | 28×28 灰度图 | 1×28×28 |
| C1 | 卷积 | 6 个 5×5 卷积核 | 6×24×24 |
| S2 | 池化 | 2×2 平均池化 | 6×12×12 |
| C3 | 卷积 | 16 个 5×5 卷积核 | 16×8×8 |
| S4 | 池化 | 2×2 平均池化 | 16×4×4 |
| F5 | 全连接 | 120 个神经元 | 120 |
| F6 | 全连接 | 84 个神经元 | 84 |
| 输出 | 全连接 | 10 个神经元 | 10 |

## 实验结果摘要

### 学习率对比
- **LR=0.001**: 测试准确率 13.6%（收敛缓慢）
- **LR=0.01**: 测试准确率 65.4%（正常收敛）
- **LR=0.1**: 测试准确率 96.0%（快速收敛）✓ 最优

### 批大小对比
- **BS=16**: 测试准确率 86.3%（最高）✓ 最优
- **BS=32**: 测试准确率 83.1%
- **BS=64**: 测试准确率 64.5%（收敛缓慢）

### 优化器对比
- **SGD**: 测试准确率 49.2%（表现不佳）
- **Adam**: 测试准确率 97.1%（最高）✓ 最优
- **RMSprop**: 测试准确率 94.5%（次优）

## 环境要求

### 依赖包
- Python 3.11+
- MindSpore 2.8.0
- PyTorch 2.11.0+（可选）
- NumPy
- Matplotlib
- Torchvision（用于 PyTorch 版本）

### 安装依赖
```bash
# 安装 MindSpore
sudo uv pip install --system mindspore

# 安装 PyTorch（可选）
sudo uv pip install --system torch torchvision

# 安装其他依赖
sudo uv pip install --system matplotlib numpy
```

## 使用说明

### 步骤 1：准备环境
```bash
cd /home/ubuntu/lenet5_experiment
```

### 步骤 2：下载数据集
MNIST 数据集会在首次运行时自动下载到 `./data` 目录。

### 步骤 3：运行实验
```bash
# 运行 MindSpore 版本的参数对比实验
python3 experiment_runner_v2.py

# 或运行 PyTorch 版本
python3 experiment_pytorch.py
```

### 步骤 4：生成可视化图表
```bash
python3 visualize_results.py
```

### 步骤 5：查看结果
- 实验结果保存在 `experiment_results.json`
- 图表保存在 `figures/` 目录
- 完整报告在 `LeNet5_MNIST_Experiment_Report.pdf`

## 关键发现

1. **学习率的重要性**：在 SGD 优化器下，学习率从 0.001 提升到 0.1 时，测试准确率从 13.6% 提升到 96.0%，说明学习率的选择对模型性能有决定性影响。

2. **批大小的权衡**：较小的批大小（16）相比较大的批大小（64）能取得更好的准确率（86.3% vs 64.5%），但会增加训练时间。

3. **优化器的优越性**：自适应优化器（Adam、RMSprop）相比基础 SGD 表现出显著优势，Adam 达到 97.1% 的测试准确率，而 SGD 仅有 49.2%。

## 实验报告

详细的实验报告请参阅 `LeNet5_MNIST_Experiment_Report.pdf`，包含：
- 详细的网络结构介绍
- 实验设计与方法
- 完整的实验结果与分析
- 结论与建议

## 代码特点

- **模块化设计**：代码结构清晰，易于理解和扩展
- **完整的注释**：每个关键函数都有详细的中文注释
- **灵活的参数**：支持自定义学习率、批大小、优化器等参数
- **详细的日志**：训练过程中输出详细的进度信息
- **结果保存**：自动保存实验结果为 JSON 格式，便于后续分析

## 扩展建议

1. **增加更多优化器**：可以尝试 Adagrad、Adadelta 等其他优化器
2. **数据增强**：使用旋转、缩放等数据增强技术提升模型泛化能力
3. **网络改进**：尝试添加 Dropout、Batch Normalization 等正则化技术
4. **超参数搜索**：使用网格搜索或贝叶斯优化进行更系统的超参数调优
5. **模型保存与加载**：实现模型的保存和加载功能，便于后续使用

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有任何问题或建议，欢迎反馈。

---

**最后更新：** 2026年3月30日
