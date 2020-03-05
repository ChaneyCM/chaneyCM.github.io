---
layout:     post
title:      CS224n:Project2Word2Vec原理及实现
subtitle:   CS224n系列
date:       2020-03-02
author:     chen cheng
header-img: img/post-bg-unix-linux.jpg
catalog: true
tags:
    - 项目整理
---

### 一、项目介绍
word2vec是一种获取词嵌入的方式，此次project2利用python、numpy复现一个Word2vec的训练模型，以掌握word2vec的原理。


### 二、源代码重点介绍

![a2_softmax.jpg](/img/2020-03-02/a2_softmax.jpg)

从图中可以看到完全可以把它理解成两层感知机，最后一层的激活函数是softmax，然后loss需要再计算一下交叉熵。

因为激活函数是softmax，所以每次挑选两个词（一个中心词和一个上下文窗口词），对模型参数进行求导的时候，需要将第一层表示v的那部分参数以及第二层所有参数都要进行求导，并更新。

本次作业还实现了负采样，主要在于在vocab很大的情况下，求softmax以及求导等等消耗很大。所以我们需要进行优化，优化方式就是激活函数改成sigmoid，然后这样我们在更新参数的时候就只是需要改进v和u对应的两层部分参数，那与之前更新第二层全部参数差别很大（增大u和v同窗口的概率，减小v和其它所有词的同在窗口的概率）。如果不负采样，明显不易训练，只有正例，也许也能出效果但是不够好。因此需要选出k个不在窗口中的，然后通过训练，降低它们同在窗口的概率。

### 三、 word2vec的一些个人理解

https://zhuanlan.zhihu.com/p/26306795

word2vec有跳字模型和连续词袋模型。

对每个词的embedding的值就成为了模型的参数。跳字模型就是一个词生成另外窗口个词的概率，也就是以那个词为中心，生成N个窗口背景词的概率，vc和N个uo，最后使用的是v中心词向量来作为表征向量。而连续词袋模型使用的是u背景词向量来作为表征向量。N个中心词生成某个背景词的概率。使用极大似然概率去估计词嵌入。

注意两种训练时的优化方式是负采样和层序。

跳字模型原本的训练方式的核心是使用softmax来获取在给定中心词Wc来生成背景词Wo的条件概率。

跳字模型的意思。其实就是通过不断的训练，比如训练中心词Vc和某窗口词Uo，它的训练结果一定是希望Vc去接近Uo，这样才能使得概率更大，也就是想让Vc跟Uo去相似，然后训练Vc和某同窗口的词Uo2的时候，又想让Vc跟Uo2去接近，最后的训练结果就是让Vc去跟所有跟它同窗口过的词的背景词向量U们去接近，这也就导致可以说Vc最终的值是由它的上下文词语的U们去决定的了。

因此Vc如果等于Vc2，那么就可以说单词c和单词c2是含义非常接近的，因为它们有着相同的上下文。我们说两个词含义相同的时候，其实不是说单词A更容易跟单词B同时出现（出现在一个窗口）（这样反而其实含义并不相似），而是大多数情况下两个单词可以互相替换，也就是拥有一个上下文的时候，它俩的含义才相似。

这也是为什么要拥有U和V两个向量的意思。如果只有一套向量V来表示，那么训练的时候的意义不明，出现在一个窗口，并不意味着含义有多么接近，为什么要把它们的表示训练得更接近？

负采样大概是这样的。我们为了知道两个词在一个窗口的概率，最好的方式自然是softmax，算得很准确，可以softmax需要过一遍单词库才能求出导数，计算太慢，所以不能使用softmax，所以这里就打算使用sigmoid将两个向量的点乘结果映射到0-1。为什么后面要再加几个负面样本呢？因为softmax每次更新，不光会使得UV两个向量更接近，最后得分更高，同时还会使得其它的得分更低。而如果放弃了softmax，就没有这样的现象了。没有负面样本容易使得最后所有词向量相等且为无穷大，因为这样是最优的。因此为了充分模仿softmax（毕竟本来就是希望以更小的算法消耗去模仿），那么我们就选择随机挑选K个负样本，并最小化UV负在窗口中的概率，即最大化UV负不在窗口中的概率。

层序softmax也是想办法优化softmax的计算效率。
[https://www.jianshu.com/p/de494438e585](https://www.jianshu.com/p/de494438e585)

zip(*a) 这个*号有一种将一个list或者一个tuple消解掉的意思，即去掉最外层的包裹的东西，变成多个item，然后这些item的长度如果相同，那么就可以进行zip操作。

训练目标就是让V和窗口中的背景词中的U点乘向1倾斜，这就是softmax反向传播的目标。当然1还是不可能的，总有损失，这是这就是label（背景词是1，非背景词是0，求交叉熵为loss）

小结：为什么word2vec有两套词向量？
1. 两套词向量求导好求导。如果以两层神经网络去看待word2vec的训练过程，其实完全可以这么看待，那当然容易求导。可如果我们只用一套词向量，那也就是让两层神经网络的参数时刻保持一致，想想就十分难以求导。

网上说的是：训练两组词向量是为了计算梯度的时候求导更方便。如果只用一组词向量 ，那么Softmax计算的概率公式里分母会出现一项平方项 ，那么再对 求导就会比较麻烦。相反如果用两套词向量，求导结果就会很干净。但其实，因为在窗口移动的时候，先前窗口的中心词会变成当前窗口的上下文词，先前窗口的某一个上下文词会变成当前窗口的中心词。所以这两组词向量用来训练的词对其实很相近，训练结果也会很相近。一般做法是取两组向量的平均值作为最后的词向量。

2. 使用一套词向量，与word2vec模型最初的假设不同。
word2vec的模型的假设是，含义相似的word，那么他们的context是相似的。也就是说，可能这些词不会同时出现，但是因为它们的context相似，所以它们的表示就是相似的。这跟我们理解的同义词是一个意思。

而如果我们使用一套词向量，我们就是把常常同时出现的词的表示向一致去训练，就成了经常同时出现的2个词的表示越相似。这与模型假设明显有违背。

### 课后习题
### Solution for Assignment #2

### 1 Understanding word2vec

 **(a)**  As described in the doc, $\boldsymbol{y}$ is a one-hot vector with a 1 for the true outside word $o$, that means $y_i$ is 1 if and only if $i == o$. so the proof could be below:
<!-- $ - \sum_{w\in Vocab}y_w\log(\hat{y}_o) = $ -->

$\begin{aligned}
    - \sum_{w\in Vocab}y_w\log(\hat{y}_w) &= - [y_1\log(\hat{y}_1) + \cdots + y_o\log(\hat{y}_o) + \cdots + y_w\log(\hat{y}_w)] \\
    & = - y_o\log(\hat{y}_o) \\
    & = -\log(\hat{y}_o) \\
    & = -\log \mathrm{P}(O = o | C = c)
\end{aligned}$

**(b)** we know this deravatives:
$$
\because J = CE(y, \hat{y}) \\
\hat{y} = softmax(\theta)\\
\therefore \frac{\partial J}{\partial \theta} = (\hat{y} - y)^T
$$

$y$ is a column vector in the above equation. So, we can use chain rules to solve the deravitive:

$$\begin{aligned}
\frac{\partial J}{\partial v_c} &= \frac{\partial J}{\partial \theta} \frac{\partial \theta}{\partial v_c} \\
&= (\hat{y} - y) \frac{\partial U^Tv_c}{\partial v_c} \\
&= U^T(\hat{y} - y)^T
\end{aligned}$$

**(c)**
similar to the equation above.
$$\begin{aligned}
\frac{\partial J}{\partial U} &= \frac{\partial J}{\partial \theta} \frac{\partial \theta}{\partial U} \\
&= (\hat{y} - y) \frac{\partial U^Tv_c}{\partial U} \\
&= v_c(\hat{y} - y)^T
\end{aligned}$$

**(d)**