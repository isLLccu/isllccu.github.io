---
layout: home

hero:
  name: "林乐珊 · Leshan Lin"
  text: "把模型写明白，把实验做扎实"
  tagline: "计算机科学本科生｜深度学习 · AI for Science · 可靠工程"
  image:
    src: https://github.com/isLLccu.png
    alt: Leshan Lin
  actions:
    - theme: brand
      text: 查看精选项目
      link: /projects
    - theme: alt
      text: 阅读技术文章
      link: /posts/
    - theme: alt
      text: GitHub
      link: https://github.com/isLLccu

features:
  - icon: 🧠
    title: From Scratch
    details: 手动实现 MLP 与 Multi-Head Attention，用可验证的代码理解模型内部机制。
  - icon: 📐
    title: Experimental Rigor
    details: 区分训练、验证与测试，控制初始化和数据顺序，真实报告实验局限。
  - icon: 🌊
    title: AI for Science
    details: 关注物理约束学习、神经算子与长时序地震波场预测。
---

<div class="home-intro">

## 不只展示结果，也展示判断过程

我关注的不只是“模型跑出了多少分”，还包括：比较是否公平、指标是否可信、失败意味着什么，以及代码能否被别人复现。

<div class="project-grid">
  <a class="project-card featured" href="/posts/attention-from-scratch">
    <span class="eyebrow">FEATURED · NLP</span>
    <h3>从 Q/K/V 手写多头注意力</h3>
    <p>从形状推导、mask 单元测试到 IMDb 分类与注意力可视化。</p>
    <span class="metric">68.8% held-out test accuracy →</span>
  </a>
  <a class="project-card" href="/posts/mlp-from-scratch">
    <span class="eyebrow">FOUNDATIONS · CV</span>
    <h3>不用 nn.Linear 实现 MLP</h3>
    <p>手动完成参数、前向传播、交叉熵和 SGD，并与 PyTorch 实现公平对照。</p>
    <span class="metric">101,770 parameters · two implementations →</span>
  </a>
  <a class="project-card" href="/posts/evaluation-without-leakage">
    <span class="eyebrow">METHODOLOGY</span>
    <h3>如何避免“看起来很高”的无效指标</h3>
    <p>从测试集泄漏、随机标签记忆和四张图的 75% 谈可靠实验。</p>
    <span class="metric">Honest evaluation checklist →</span>
  </a>
</div>

## 当前关注

`PyTorch` · `Transformer` · `Physics-informed Learning` · `Distributed Training` · `FastAPI`

</div>
