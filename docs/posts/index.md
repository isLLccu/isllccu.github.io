# 技术文章

这里记录我怎样实现模型、设计实验、定位问题，并把一次运行整理成可复现的技术证据。

## 深度学习实践

### [从零实现 MLP：当 `nn.Linear` 不再替你隐藏细节](./mlp-from-scratch)

手动实现参数、ReLU、稳定交叉熵与 SGD，并修正原实验中训练期间查看测试集的问题。

### [从 Q/K/V 开始手写 Multi-Head Attention](./attention-from-scratch)

从张量形状推导到 mask 单元测试，再到 IMDb 分类、SDPA 对照和注意力诊断。

### [如何避免测试集泄漏与无效高分](./evaluation-without-leakage)

结合三个真实例子，说明验证集、随机标签和小样本指标应该怎样解释。

## AI for Science

### [从拟合数据到学习物理：算子学习深度解析](./first-post)

神经算子、跨分辨率泛化、FNO/CNO、时间依赖 PDE 与生成式科学建模。
