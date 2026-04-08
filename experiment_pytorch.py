"""
LeNet-5参数对比实验 - PyTorch版本
采用控制变量法，对学习率、批大小、优化器三种参数进行实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
import time
import os
import random


RANDOM_SEED = 43


def set_random_seed(seed=RANDOM_SEED):
    """固定随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class LeNet5(nn.Module):
    """LeNet-5卷积神经网络"""

    def __init__(self):
        super(LeNet5, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)

        # 池化层
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # C1: 卷积层
        x = self.conv1(x)  # (1, 32, 32) -> (6, 28, 28)
        x = self.relu(x)

        # S2: 池化层
        x = self.pool(x)  # (6, 28, 28) -> (6, 14, 14)

        # C3: 卷积层
        x = self.conv2(x)  # (6, 14, 14) -> (16, 10, 10)
        x = self.relu(x)

        # S4: 池化层
        x = self.pool(x)  # (16, 10, 10) -> (16, 5, 5)

        # 展平
        x = x.view(x.size(0), -1)  # (16, 5, 5) -> (batch_size, 400)

        # F5: 全连接层
        x = self.fc1(x)
        x = self.relu(x)

        # F6: 全连接层
        x = self.fc2(x)
        x = self.relu(x)

        # 输出层
        x = self.fc3(x)

        return x


def create_mnist_dataloaders(
    batch_size=32, num_train_samples=None, num_test_samples=None, seed=RANDOM_SEED
):
    """创建MNIST数据加载器"""

    # 数据预处理
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # 加载训练集
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # 限制样本数量
    if num_train_samples is not None:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(min(num_train_samples, len(train_dataset)))
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    # 加载测试集
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # 限制样本数量
    if num_test_samples is not None:
        test_dataset = torch.utils.data.Subset(
            test_dataset, range(min(num_test_samples, len(test_dataset)))
        )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader


def train_and_evaluate(
    model,
    train_loader,
    test_loader,
    epochs,
    learning_rate,
    optimizer_type="SGD",
    device="cpu",
    verbose=False,
):
    """
    训练和评估模型
    """

    model.to(device)

    # 定义优化器
    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "test_acc": [], "test_loss": []}

    if verbose:
        print(f"\n开始训练 - 优化器: {optimizer_type}, 学习率: {learning_rate}")

    start_time = time.time()

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # 测试阶段
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_acc = test_correct / test_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)

        if verbose:
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

    # 检查GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    print("=" * 70)
    print("加载MNIST数据集...")
    print("=" * 70)

    # 创建基础数据加载器
    train_loader_base, test_loader_base = create_mnist_dataloaders(batch_size=32)

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

            # 创建新的数据加载器（使用指定的批大小）
            train_loader, test_loader = create_mnist_dataloaders(
                batch_size=params["batch_size"],
                seed=RANDOM_SEED,
            )

            # 创建新的模型
            model = LeNet5()

            # 训练和评估
            test_acc, train_acc, history, elapsed_time = train_and_evaluate(
                model,
                train_loader,
                test_loader,
                epochs=20,
                learning_rate=params["learning_rate"],
                optimizer_type=params["optimizer"],
                device=device,
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
