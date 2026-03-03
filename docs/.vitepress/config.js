export default {
    title: "isLLblog",
    description: "我的技术成长之路",
    base: '/',

    themeConfig: {
        nav: [
            { text: '首页', link: '/' },
            // ✅ 确保这里指向 /posts/ (末尾有斜杠)
            { text: '文章', link: '/posts/' },
            { text: '关于我', link: '/about' }
        ],

        sidebar: {
            '/posts/': [
                {
                    text: '技术文章',
                    items: [
                        // ✅ 确保这里指向具体的文件名 (相对路径)
                        { text: '从拟合数据到学习物理：算子学习深度解析', link: '/posts/first-post' }
                    ]
                }
            ]
        }
    }
}