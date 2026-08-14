import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'Leshan Lin',
  description: '林乐珊的机器学习项目、实验复盘与技术笔记',
  base: '/',
  cleanUrls: true,

  head: [
    ['meta', { name: 'theme-color', content: '#5b5bd6' }],
    ['meta', { name: 'author', content: 'Leshan Lin' }]
  ],

  themeConfig: {
    siteTitle: 'Leshan Lin',
    nav: [
      { text: '首页', link: '/' },
      { text: '项目', link: '/projects' },
      { text: '文章', link: '/posts/' },
      { text: '关于', link: '/about' }
    ],

    sidebar: {
      '/posts/': [
        {
          text: '深度学习实践',
          items: [
            { text: '从零实现 MLP', link: '/posts/mlp-from-scratch' },
            { text: '从 Q/K/V 手写多头注意力', link: '/posts/attention-from-scratch' },
            { text: '如何避免测试集泄漏', link: '/posts/evaluation-without-leakage' }
          ]
        },
        {
          text: 'AI for Science',
          items: [
            { text: '从拟合数据到学习物理', link: '/posts/first-post' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/isLLccu' }
    ],

    search: { provider: 'local' },

    footer: {
      message: 'Build, measure, explain.',
      copyright: 'Copyright © 2026 Leshan Lin'
    },

    outline: {
      level: [2, 3],
      label: '本页目录'
    },
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },
    lastUpdated: {
      text: '最后更新'
    }
  }
})
