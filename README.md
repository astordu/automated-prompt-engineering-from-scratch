# 贝叶斯优化提示语工程

假设我们有一个初版的提示语工程，然后对这个的提示语做一个评价，我们基于评价信息来产生新的提示语，这样一轮一轮的迭代过程可以把提示语工程进行自动化提升。这也就是一个贝叶斯优化的过程：
![[Pasted image 20241023090101.png]]

流程是这样：
1.提供一个初始化提示词
2.使用提示词批量跑标准测试集（比如历史选择题试卷）
3.把提示词与试卷的得分（比如80分，100道题答对了80道题）给到一个分析改善的大模型，让他帮我们分析改良提示词
4.让模型对分析改良后的提示词重新进行批量标准测试机的评估
。。。。
以此类推。让程序自己改良提示词，这样，自动提示语工程的程序就做好了。
（注意：让模型进行提示语分析改善提示词的时候，应该将之前所有跑过测评和得分的结果都传过去。让他综合进行分析：哪些提示词好了，为什么好，可以借鉴吗。哪些提示词部分不好，为什么不好，可以避免吗）

为了实现思考的流程，图中应该重点关注这两个模块： 
1. 对提示语的测评
2. 改善提示语

# 具体的可以看我的博客文章： 



# 项目的一些说明：

此项目是在Heiko Hotz原项目的基础上，我进行了修改，修改内容如下：

1. 将vertai的工具去除掉（因为不支持国内模型）
2. 将国内模型DeepSeek整合进了项目
3. 将英文几何徒刑SVG测试集换成了中国历史高考题[2]
4. 将所有提示词更改调成了中文提示词

原来项目地址：https://github.com/marshmellow77/automated-prompt-engineering-from-scratch