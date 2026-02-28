export default {
    title: "isLLblog",
    description: "我的技术成长之路",
    themeConfig: {
        nav: [
            { text: '首页', link: '/' },
            { text: '文章', link: '/posts/' },
            { text: '关于我', link: '/about' }
        ],
        sidebar: {
            '/posts/': [
                { text: '第一篇文章', link: '/posts/first-post' }
            ]
        }
    }
}