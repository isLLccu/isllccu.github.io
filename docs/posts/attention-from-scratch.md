---
title: 从 Q/K/V 开始手写 Multi-Head Attention
description: 从张量形状、mask 单元测试到 IMDb 分类和注意力诊断。
---

# 从 Q/K/V 开始手写 Multi-Head Attention

> 项目代码：[deep-learning-foundations](https://github.com/isLLccu/deep-learning-foundations/tree/main/projects/04-transformer-from-scratch)

为了理解 Transformer，我没有从 `nn.MultiheadAttention` 开始，而是自己完成 Q/K/V 投影、拆头、缩放点积、mask、softmax、加权求和与合并多头。

## 1. 数据怎样流过注意力层

输入为：

$$X\in\mathbb{R}^{B\times L\times D}.$$

经过三组线性投影：

$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V.$$

假设有 $H$ 个头，每个头的维度为 $D_h=D/H$，拆头后的形状是：

$$Q,K,V\in\mathbb{R}^{B\times H\times L\times D_h}.$$

注意力权重为：

$$A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D_h}}+M\right).$$

这里最容易出错的不是公式，而是维度顺序和 mask 语义。

## 2. 在训练之前先写测试

“代码可以运行”不代表注意力正确。我先检查：

1. 输出是否为 `(B, L, D)`；
2. 权重是否为 `(B, H, L, L)`；
3. 每个 query 对 key 的概率之和是否接近 1；
4. padding key 的权重是否接近 0；
5. 手写路径和 SDPA 路径是否复用同一组 Q/K/V/O 投影。

保存的运行结果中，概率归一化误差约为 `1.19e-7`，padding key 最大权重为 `0`。

## 3. IMDb 分类实验

实验使用 4,000 / 500 / 500 条平衡的训练、验证和测试评论。简化 Transformer 的最佳验证准确率为 75.0%，独立测试准确率为 68.8%。

这个结果不是为了追赶预训练语言模型，而是验证从零实现的注意力能够端到端学习。小数据、词级分词和最大 96 token 截断都会限制性能。

## 4. 注意力图应该怎样解释

我可视化了 `[CLS]` 对各 token 的平均注意力。一些情感词确实得到较高权重，但连接词或高频词也可能被关注。

因此，注意力图适合做模型行为诊断，不足以单独证明模型“理解”了情感，更不能直接当作因果解释。

## 5. 关于 FlashAttention

CPU 环境下，PyTorch SDPA 不一定比小规模手写实现更快；FlashAttention 的优势主要出现在匹配的 CUDA、精度与序列规模上。项目在没有 CUDA 时明确跳过 Flash 后端测试，不伪造加速结论。

我从这个项目学到的核心不是背下公式，而是把每一步都变成可以用形状、断言和对照路径验证的代码。

