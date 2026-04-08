"""
LeNet-5参数对比实验 - 优化版本
采用控制变量法，对学习率、批大小、优化器三种参数进行实验
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import numpy as np
import json
import os
from datetime import datetime
import time
import random


RANDOM_SEED = 42


def set_random_seed(seed=RANDOM_SEED):
    """固定随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)
    ds.config.set_seed(seed)


class LeNet5(nn.Cell):
    """LeNet-5卷积神经网络"""

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, pad_mode="valid")

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, pad_mode="valid")

        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x


def create_mnist_dataset(data_path, batch_size=32, is_training=True, num_samples=None):
    """创建MNIST数据集"""

    dataset_type = "train" if is_training else "test"

    dataset = ds.MnistDataset(
        dataset_dir=data_path, usage=dataset_type, shuffle=is_training
    )

    # 限制样本数量以加快实验
    if num_samples is not None:
        dataset = dataset.take(num_samples)

    dataset = dataset.map(operations=vision.Resize((32, 32)), input_columns="image")

    dataset = dataset.map(
        operations=transforms.TypeCast(mindspore.float32), input_columns="image"
    )

    dataset = dataset.map(
        operations=vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        input_columns="image",
    )

    dataset = dataset.map(operations=vision.HWC2CHW(), input_columns="image")

    dataset = dataset.map(
        operations=transforms.TypeCast(mindspore.int32), input_columns="label"
    )

    dataset = dataset.batch(batch_size, drop_remainder=is_training)

    return dataset


def train_and_evaluate(
    model,
    train_dataset,
    test_dataset,
    epochs,
    learning_rate,
    optimizer_type="SGD",
    verbose=False,
):
    """
    训练和评估模型
    """

    # 定义优化器
    if optimizer_type == "SGD":
        optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)
    elif optimizer_type == "Adam":
        optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    elif optimizer_type == "RMSprop":
        optimizer = nn.RMSProp(model.trainable_params(), learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = mindspore.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True
    )

    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        optimizer(grads)
        return loss, logits

    def eval_model(dataset):
        total_correct = 0
        total_samples = 0
        total_loss = 0

        for data, label in dataset.create_tuple_iterator():
            logits = model(data)
            loss = loss_fn(logits, label)

            predictions = ops.argmax(logits, 1)
            correct = ops.sum(predictions == label)

            total_correct += correct.asnumpy()
            total_samples += label.shape[0]
            total_loss += loss.asnumpy()

        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(dataset)

        return accuracy, avg_loss

    history = {"train_loss": [], "train_acc": [], "test_acc": [], "test_loss": []}

    if verbose:
        print(f"\n开始训练 - 优化器: {optimizer_type}, 学习率: {learning_rate}")

    start_time = time.time()

    for epoch in range(epochs):
        train_loss = 0
        train_correct = 0
        train_total = 0

        for data, label in train_dataset.create_tuple_iterator():
            loss, logits = train_step(data, label)
            predictions = ops.argmax(logits, 1)
            correct = ops.sum(predictions == label)

            train_loss += loss.asnumpy()
            train_correct += correct.asnumpy()
            train_total += label.shape[0]

        train_loss /= len(train_dataset)
        train_acc = train_correct / train_total

        test_acc, test_loss = eval_model(test_dataset)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["test_acc"].append(float(test_acc))
        history["test_loss"].append(float(test_loss))

        print(
            f"  Epoch {epoch+1:2d}/{epochs} | "
            f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
        )

    elapsed_time = time.time() - start_time

    final_test_acc = history["test_acc"][-1]
    final_train_acc = history["train_acc"][-1]

    if verbose:
        print(
            f"训练完成！最终测试准确率: {final_test_acc:.4f}, 耗时: {elapsed_time:.2f}s"
        )

    return final_test_acc, final_train_acc, history, elapsed_time


def run_experiments():
    """运行所有参数对比实验"""

    set_random_seed()

    # 设置MindSpore上下文 - 使用PYNATIVE模式
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

    # 加载数据集（仅一次）
    print("=" * 70)
    print("加载MNIST数据集...")
    print("=" * 70)

    data_path = "./data"

    # 使用部分数据集以加快实验（5000训练样本，1000测试样本）
    train_dataset_base = create_mnist_dataset(
        data_path, batch_size=32, is_training=True, num_samples=5000
    )
    test_dataset_base = create_mnist_dataset(
        data_path, batch_size=32, is_training=False, num_samples=1000
    )

    print("数据集加载完成！\n")

    # 实验配置
    experiments = {
        "学习率对比": {
            "description": "固定优化器(SGD)和批大小(32)，改变学习率",
            "params": [
                {"learning_rate": 0.001, "batch_size": 32, "optimizer": "SGD"},
                {"learning_rate": 0.01, "batch_size": 32, "optimizer": "SGD"},
                {"learning_rate": 0.1, "batch_size": 32, "optimizer": "SGD"},
            ],
        },
        "批大小对比": {
            "description": "固定优化器(SGD)和学习率(0.01)，改变批大小",
            "params": [
                {"learning_rate": 0.01, "batch_size": 16, "optimizer": "SGD"},
                {"learning_rate": 0.01, "batch_size": 32, "optimizer": "SGD"},
                {"learning_rate": 0.01, "batch_size": 64, "optimizer": "SGD"},
            ],
        },
        "优化器对比": {
            "description": "固定学习率(0.01)和批大小(32)，改变优化器",
            "params": [
                {"learning_rate": 0.01, "batch_size": 32, "optimizer": "SGD"},
                {"learning_rate": 0.01, "batch_size": 32, "optimizer": "Adam"},
                {"learning_rate": 0.01, "batch_size": 32, "optimizer": "RMSprop"},
            ],
        },
    }

    # 存储所有实验结果
    all_results = {}

    # 运行实验
    for exp_name, exp_config in experiments.items():
        print("=" * 70)
        print(f"实验: {exp_name}")
        print(f"说明: {exp_config['description']}")
        print("=" * 70)

        exp_results = []

        for i, params in enumerate(exp_config["params"]):
            set_random_seed()
            print(f"\n第 {i+1}/3 组实验:")
            print(f"  学习率: {params['learning_rate']}")
            print(f"  批大小: {params['batch_size']}")
            print(f"  优化器: {params['optimizer']}")

            # 创建新的数据集（使用指定的批大小）
            train_dataset = create_mnist_dataset(
                data_path,
                batch_size=params["batch_size"],
                is_training=True,
                num_samples=5000,
            )
            test_dataset = create_mnist_dataset(
                data_path,
                batch_size=params["batch_size"],
                is_training=False,
                num_samples=1000,
            )

            # 创建新的模型
            model = LeNet5()

            # 训练和评估
            test_acc, train_acc, history, elapsed_time = train_and_evaluate(
                model,
                train_dataset,
                test_dataset,
                epochs=20,
                learning_rate=params["learning_rate"],
                optimizer_type=params["optimizer"],
                verbose=True,
            )

            # 记录结果
            result = {
                "group": i + 1,
                "learning_rate": params["learning_rate"],
                "batch_size": params["batch_size"],
                "optimizer": params["optimizer"],
                "final_test_accuracy": test_acc,
                "final_train_accuracy": train_acc,
                "training_time": elapsed_time,
                "history": history,
            }

            exp_results.append(result)

        all_results[exp_name] = {
            "description": exp_config["description"],
            "results": exp_results,
        }

        print("\n" + "=" * 70)
        print(f"实验 {exp_name} 完成！")
        print("=" * 70)

        # 打印汇总
        print(f"\n{exp_name} - 结果汇总:")
        print("-" * 70)
        for result in exp_results:
            print(
                f"第 {result['group']} 组: 测试准确率 = {result['final_test_accuracy']:.4f}, "
                f"训练时间 = {result['training_time']:.2f}s"
            )
        print()

    # 保存结果
    output_file = "./experiment_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"实验结果已保存到: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiments()
