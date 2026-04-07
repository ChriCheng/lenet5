"""
LeNet-5 CNN实现 - 基础版本
用于MNIST手写数字识别任务
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import numpy as np
import os
from datetime import datetime


class LeNet5(nn.Cell):
    """
    LeNet-5卷积神经网络

    网络结构：
    - 输入层: 32x32 灰度图像
    - C1: 6个5x5卷积核，输出28x28x6
    - S2: 2x2平均池化，输出14x14x6
    - C3: 16个5x5卷积核，输出10x10x16
    - S4: 2x2平均池化，输出5x5x16
    - F5: 120个神经元的全连接层
    - F6: 84个神经元的全连接层
    - 输出层: 10个神经元（0-9数字分类）
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, pad_mode="valid")

        # 池化层
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, pad_mode="valid")

        # 全连接层
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

        # 激活函数
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def construct(self, x):
        """前向传播"""
        # C1: 卷积层
        x = self.conv1(x)
        x = self.relu(x)

        # S2: 池化层
        x = self.pool(x)

        # C3: 卷积层
        x = self.conv2(x)
        x = self.relu(x)

        # S4: 池化层
        x = self.pool(x)

        # 展平
        x = self.flatten(x)

        # F5: 全连接层
        x = self.fc1(x)
        x = self.relu(x)

        # F6: 全连接层
        x = self.fc2(x)
        x = self.relu(x)

        # 输出层
        x = self.fc3(x)

        return x


def create_mnist_dataset(data_path, batch_size=32, is_training=True):
    """
    创建MNIST数据集

    Args:
        data_path: 数据集路径
        batch_size: 批大小
        is_training: 是否为训练集

    Returns:
        数据集对象
    """
    # 确定数据集类型
    dataset_type = "train" if is_training else "test"

    # 创建数据集
    dataset = ds.MnistDataset(
        dataset_dir=data_path, usage=dataset_type, shuffle=is_training
    )

    # 数据预处理
    # 1. 将MNIST图像调整到LeNet-5期望的32x32输入
    dataset = dataset.map(
        operations=vision.Resize((32, 32)), input_columns="image"
    )

    # 2. 将图像转换为浮点数并标准化
    dataset = dataset.map(
        operations=transforms.TypeCast(mindspore.float32), input_columns="image"
    )

    dataset = dataset.map(
        operations=vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        input_columns="image",
    )

    # 3. 将图像从HWC转换为CHW，以匹配Conv2d默认的NCHW布局
    dataset = dataset.map(operations=vision.HWC2CHW(), input_columns="image")

    # 4. 将标签转换为int32
    dataset = dataset.map(
        operations=transforms.TypeCast(mindspore.int32), input_columns="label"
    )

    # 5. 批处理
    dataset = dataset.batch(batch_size, drop_remainder=is_training)

    return dataset


def train_model(
    model,
    train_dataset,
    test_dataset,
    epochs,
    learning_rate,
    optimizer_type="SGD",
    save_path="./model_checkpoint.ckpt",
):
    """
    训练模型

    Args:
        model: 模型对象
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        epochs: 训练轮数
        learning_rate: 学习率
        optimizer_type: 优化器类型 ('SGD', 'Adam', 'RMSprop')
        save_path: 模型保存路径

    Returns:
        训练历史记录
    """

    # 定义优化器
    if optimizer_type == "SGD":
        optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)
    elif optimizer_type == "Adam":
        optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    elif optimizer_type == "RMSprop":
        optimizer = nn.RMSprop(model.trainable_params(), learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # 定义损失函数
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # 定义前向函数
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    # 定义梯度函数
    grad_fn = mindspore.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True
    )

    # 定义训练步骤
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        return loss

    # 定义评估函数
    def eval_model(dataset):
        total_correct = 0
        total_samples = 0
        total_loss = 0

        for data, label in dataset.create_tuple_iterator():
            logits = model(data)
            loss = loss_fn(logits, label)

            # 计算准确率
            predictions = ops.argmax(logits, 1)
            correct = ops.sum(predictions == label)

            total_correct += correct.asnumpy()
            total_samples += label.shape[0]
            total_loss += loss.asnumpy()

        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(dataset)

        return accuracy, avg_loss

    # 训练历史
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "test_loss": []}

    print(f"\n{'='*70}")
    print(f"开始训练 - 优化器: {optimizer_type}, 学习率: {learning_rate}")
    print(f"{'='*70}")

    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        train_loss = 0
        train_correct = 0
        train_total = 0

        for data, label in train_dataset.create_tuple_iterator():
            loss = train_step(data, label)

            # 计算训练准确率
            logits = model(data)
            predictions = ops.argmax(logits, 1)
            correct = ops.sum(predictions == label)

            train_loss += loss.asnumpy()
            train_correct += correct.asnumpy()
            train_total += label.shape[0]

        train_loss /= len(train_dataset)
        train_acc = train_correct / train_total

        # 测试阶段
        test_acc, test_loss = eval_model(test_dataset)

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)

        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
            )

    print(f"{'='*70}")
    print(f"训练完成！最终测试准确率: {history['test_acc'][-1]:.4f}")
    print(f"{'='*70}\n")

    return history


def evaluate_model(model, test_dataset):
    """
    评估模型

    Args:
        model: 模型对象
        test_dataset: 测试数据集

    Returns:
        准确率和损失
    """
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    total_correct = 0
    total_samples = 0
    total_loss = 0

    for data, label in test_dataset.create_tuple_iterator():
        logits = model(data)
        loss = loss_fn(logits, label)

        predictions = ops.argmax(logits, 1)
        correct = ops.sum(predictions == label)

        total_correct += correct.asnumpy()
        total_samples += label.shape[0]
        total_loss += loss.asnumpy()

    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(test_dataset)

    return accuracy, avg_loss


if __name__ == "__main__":
    # 设置MindSpore上下文
    mindspore.set_context(mode=mindspore.GRAPH_MODE)
    if hasattr(mindspore, "set_device"):
        mindspore.set_device("CPU")
    else:
        mindspore.set_context(device_target="CPU")

    # 数据集路径
    data_path = "./data"

    # 创建数据集
    print("加载MNIST数据集...")
    train_dataset = create_mnist_dataset(data_path, batch_size=32, is_training=True)
    test_dataset = create_mnist_dataset(data_path, batch_size=32, is_training=False)

    # 创建模型
    print("创建LeNet-5模型...")
    model = LeNet5()

    # 训练模型
    history = train_model(
        model,
        train_dataset,
        test_dataset,
        epochs=10,
        learning_rate=0.01,
        optimizer_type="SGD",
    )

    # 评估模型
    test_acc, test_loss = evaluate_model(model, test_dataset)
    print(f"\n最终测试准确率: {test_acc:.4f}")
    print(f"最终测试损失: {test_loss:.4f}")
