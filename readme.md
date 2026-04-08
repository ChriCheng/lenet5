# LeNet-5 卷积神经网络 MNIST 手写字符识别实验

## 项目概述

本项目实现了经典的 LeNet-5 卷积神经网络，用于识别 MNIST 手写数字数据集。通过采用控制变量法，对学习率、批大小和优化器三种关键超参数进行了系统的对比实验，探究不同参数设置对模型性能的影响。

## 文件结构

```
lenet5_experiment/
├── readme.md   
├── LeNet5_MNIST_Experiment_Report.md
├── LeNet5_MNIST_Experiment_Report.pdf
├── LeNet5_MindSpore_Code.zip
├── data
│   ├── MNIST
├── download_mnist.py
├── environment.yml
├── experiment_pytorch.py
├── experiment_results.json
├── experiment_runner_v2.py
├── figures
│   ├── batch_size_comparison.png
│   ├── comprehensive_comparison.png
│   ├── learning_rate_comparison.png
│   └── optimizer_comparison.png
├── lenet5_base.py
├── log
│   ├── torch.log
│   ├── trainv2.log
│   └── visualize.log
├── readme.md                               # 实验报告
└── visualize_results.py                    # 实验结果可视化脚本


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
- **LR=0.001**: 测试准确率 95.2%（收敛缓慢）
- **LR=0.01**: 测试准确率 95.5%（正常收敛）✓ 最优
- **LR=0.1**: 测试准确率 12.6%（快速收敛）

没有 momentum 沿稳定方向推进，过高学习率容易容易在梯度下降中难以有效降低loss

### 批大小对比
- **BS=16**: 测试准确率 94.0%（耗时210.89s）
- **BS=32**: 测试准确率 95.5%（耗时81.39s）✓ 最优
- **BS=64**: 测试准确率 96.4%（耗时61.74s） 

合理的batchsize兼具准确度与耗时
### 优化器对比
- **SGD**: 测试准确率 95.5% ✓ 最优
- **Adam**: 测试准确率 89.4%  
- **RMSprop**: 测试准确率 73.6%

这里是受到我自己科研的启发，在我的那个项目中adam优化器能提高约10%的准确度，此处是传统SGD效果最好
