"""
LeNet-5 RMSprop单项实验 - MindSpore PYNATIVE版本
输出格式与experiment_runner_v2保持一致，便于在Jupyter中单独复现。
"""

import json
import random
import time

import mindspore
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


RANDOM_SEED = 43


def set_random_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)
    ds.config.set_seed(seed)


class LeNet5(nn.Cell):
    def __init__(self):
        super().__init__()
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
        return self.fc3(x)


def create_mnist_dataset(data_path, batch_size=32, is_training=True):
    dataset_type = "train" if is_training else "test"
    dataset = ds.MnistDataset(
        dataset_dir=data_path, usage=dataset_type, shuffle=is_training
    )
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
    model, train_dataset, test_dataset, epochs, learning_rate, verbose=True
):
    optimizer = nn.RMSProp(model.trainable_params(), learning_rate=learning_rate)
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
        return total_correct / total_samples, total_loss / len(dataset)

    history = {"train_loss": [], "train_acc": [], "test_acc": [], "test_loss": []}

    if verbose:
        print(f"\n开始训练 - 优化器: RMSprop, 学习率: {learning_rate}")

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


def run_rmsprop_experiment(
    data_path="./data", batch_size=32, learning_rate=0.001, epochs=20, output_file=None
):
    set_random_seed()
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

    print("=" * 70)
    print("加载MNIST数据集...")
    print("=" * 70)
    train_dataset = create_mnist_dataset(
        data_path, batch_size=batch_size, is_training=True
    )
    test_dataset = create_mnist_dataset(
        data_path, batch_size=batch_size, is_training=False
    )
    print("数据集加载完成！\n")

    print("=" * 70)
    print("实验: RMSprop 单项复现实验")
    print("说明: 固定优化器(RMSprop)，单独复现其训练过程")
    print("=" * 70)
    print("\n第 1/1 组实验:")
    print(f"  学习率: {learning_rate}")
    print(f"  批大小: {batch_size}")
    print("  优化器: RMSprop")

    model = LeNet5()
    test_acc, train_acc, history, elapsed_time = train_and_evaluate(
        model,
        train_dataset,
        test_dataset,
        epochs=epochs,
        learning_rate=learning_rate,
        verbose=True,
    )

    result = {
        "group": 1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": "RMSprop",
        "final_test_accuracy": test_acc,
        "final_train_accuracy": train_acc,
        "training_time": elapsed_time,
        "history": history,
    }
    all_results = {
        "RMSprop单项复现实验": {
            "description": "固定优化器(RMSprop)，单独复现其训练过程",
            "results": [result],
        }
    }

    print("\n" + "=" * 70)
    print("实验 RMSprop 单项复现实验 完成！")
    print("=" * 70)
    print("\nRMSprop 单项复现实验 - 结果汇总:")
    print("-" * 70)
    print(f"第 1 组: 测试准确率 = {test_acc:.4f}, 训练时间 = {elapsed_time:.2f}s")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n实验结果已保存到: {output_file}")

    return all_results


if __name__ == "__main__":
    run_rmsprop_experiment(output_file="./experiment_rmsprop_runner_v2.json")
