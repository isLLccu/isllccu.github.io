# 精选项目

项目按“问题—实现—证据—局限”组织。完整代码可在 [GitHub](https://github.com/isLLccu) 查看。

<div class="project-grid page-grid">
  <a class="project-card featured" href="https://github.com/isLLccu/deep-learning-foundations">
    <span class="eyebrow">PYTORCH · JUPYTER</span>
    <h3>Deep Learning Foundations</h3>
    <p>五个从基础张量运算到 Grad-CAM 的项目，强调 from-scratch 实现、实验公平性和诚实报告。</p>
    <span class="metric">MLP · ResNet · Attention · Grad-CAM →</span>
  </a>
  <a class="project-card" href="https://github.com/isLLccu/seismic-wavefield-prediction">
    <span class="eyebrow">AI FOR SCIENCE</span>
    <h3>Seismic Wavefield Prediction</h3>
    <p>面向长时序地震波场预测的 Transformer、物理约束与 PyTorch DDP 实践。</p>
    <span class="metric">Transformer · Physics constraints · DDP →</span>
  </a>
  <a class="project-card" href="https://github.com/isLLccu/cross-cultural-sentiment-analysis">
    <span class="eyebrow">NLP · RESEARCH</span>
    <h3>Cross-cultural Sentiment Analysis</h3>
    <p>围绕非遗内容的跨平台情感与主题分析，结合 BERT、BERTopic 和语义表示。</p>
    <span class="metric">BERT · BERTopic · RoBERTa →</span>
  </a>
  <a class="project-card" href="https://github.com/isLLccu/adcraft-agent">
    <span class="eyebrow">BACKEND · LLM</span>
    <h3>AdCraft Agent</h3>
    <p>异步广告素材生成平台，关注任务队列、多用户隔离和可维护的 LLM 调用链路。</p>
    <span class="metric">FastAPI · Celery · Redis →</span>
  </a>
</div>

## 深度学习仓库中的五个实验

| 项目 | 证据 | 状态 |
|---|---|---|
| MLP from scratch | 手写实现和 `nn.Module` 公平对照 | 可复现重点项目 |
| Residual + LayerNorm | 参数共享、保存恢复、Gradient Checkpoint | 结构实验 |
| ResNet-18 / CIFAR-10 | 四组优化配置和梯度范数 | 代码完成，正式重跑待完成 |
| Multi-Head Attention | 单元测试、IMDb 分类、注意力可视化 | 可复现重点项目 |
| Transfer Learning + Grad-CAM | 完整 Pipeline 和可解释性视图 | 小样本演示 |

