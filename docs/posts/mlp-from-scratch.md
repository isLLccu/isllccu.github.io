---
title: 从零实现 MLP：当 nn.Linear 不再替你隐藏细节
description: 手动实现参数、ReLU、稳定交叉熵与 SGD，并用验证集完成公平比较。
---

# 从零实现 MLP：当 `nn.Linear` 不再替你隐藏细节

> 项目代码：[deep-learning-foundations](https://github.com/isLLccu/deep-learning-foundations/tree/main/projects/01-mlp-from-scratch)

调用 `nn.Linear` 写出一个 MLP 很容易，但如果把它拿掉，我们是否真的知道模型内部发生了什么？这个实验从张量开始，实现一个 `784 → 128 → 10` 的 Fashion-MNIST 分类器。

## 1. 我手动实现了什么

模型只有一个隐藏层：

$$
H = \operatorname{ReLU}(XW_1+b_1), \qquad
O = HW_2+b_2.
$$

手写版本没有使用 `nn.Linear`，而是显式创建四组可训练张量：`W1`、`b1`、`W2`、`b2`。前向传播使用矩阵乘法，ReLU 使用逐元素最大值。

交叉熵也不先计算 softmax，而是使用稳定形式：

$$
\log p_y = o_y - \operatorname{logsumexp}(O).
$$

这样避免指数值过大造成数值溢出。反向传播仍使用 PyTorch autograd，但 SGD 更新由我在 `torch.no_grad()` 中手动完成。

## 2. 怎样保证比较公平

我又实现了等价的 `nn.Module + nn.Linear` 版本。两者比较时固定：

- 完全相同的初始权重和偏置；
- 完全相同的 mini-batch 顺序；
- 相同的网络结构、学习率、批量大小和 epoch 数；
- 相同的 55,000 / 5,000 训练—验证划分。

两种实现的参数量都为：

$$
784\times128 + 128 + 128\times10 + 10 = 101{,}770.
$$

## 3. 我修正的实验问题

原始版本每个 epoch 都查看测试准确率，并据此选择学习率。这会让测试集参与决策，最终数字不再是独立估计。

精修版本把 Fashion-MNIST 的 60,000 张训练图片拆为 55,000 张训练数据和 5,000 张验证数据。学习率与 batch size 只根据验证曲线选择；官方测试集只在两个模型训练完成后评估一次。

这次修正比提高零点几个百分点更重要，因为它决定了实验结论是否可信。

## 4. 结论

两个模型的曲线非常接近，说明手写的矩阵运算、激活、损失和 SGD 与 PyTorch 封装具有相同数学含义。`nn.Module` 的优势是代码更简洁、工程接口更完整；手写版本的价值是把抽象层下面的机制变成可以检查的对象。

这个 MLP 没有利用图像的空间结构，因此它不是追求最高精度的方案。它是一块干净的“机制验证板”。

