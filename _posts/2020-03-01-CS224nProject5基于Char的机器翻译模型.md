---
layout:     post
title:      CS224n:Project5基于Char的机器翻译模型
subtitle:   CS224n系列
date:       2020-03-01
author:     chen cheng
header-img: img/post-bg-unix-linux.jpg
catalog: true
tags:
    - 项目整理
---

### 一、项目介绍

![a5_char_based_conv_encoder.jpg](/img/2020-03-01/a5_char_based_conv_encoder.jpg)

![a5_model.jpg](/img/2020-03-01/a5_model.jpg)

模型大体与a4相像，最主要的区别来自encoder是基于字符的，而decoder有两个：一个是基于word的decoder模型的，另一个是在基于word的decoder模型输出\<unk\>的时候启用的基于char的decoder。

OOV问题（Out Of Memory问题）


### 二、源代码重点介绍
#### 1.如何将文本资源（西班牙语和英语的翻译训练集）处理成Vocabulary类对象？


#### 2. 如何将句子按批处理成模型输入的？即Model的forward方法的参数。
![a5_to_input_tensor_char.jpg](/img/2020-03-01/a5_to_input_tensor_char.jpg)
idx_sents = words2charindices(sents)仅仅是将句子中每个字符都变成对应的charid，同时记得每个单词的前后都添加上'{'和'}'这两个char对应的id。

pad_sents_char(idx_sents, char_pad_token=self.char2id['<pad>']). 
这个方法将句子补齐到max_sentence_len长度，将每个单词都补齐到固定的字符数量，只利用pad这一个字符来填充。

使用torch.tensor把三维list变成三维tensor，然后torch.transpose(sents_var, 0, 1)变成(max_sentence_len, batch, max_word_len)。


#### 3. 模型的forward函数具体做了什么？（forward函数非常重要，可从forward的返回值中计算出loss损失以进行反向传播和训练）

相比较A4，代码几乎全赋值，少量变化有：

- Embedding层是精心设计的char-level的模型，利用了CNN和high-way结构。而不再是简单的基于word的nn.Embedding层；
- 引入一个新的charDecoder模型，处理decoder模型输出\<unk\>单词的情形。

encoder的时候，已经不需要源word2id的这种映射关系了。encode时，仅仅在embedding层上做改变即可，同时放弃了word2id。decode方法中的Decoder也仅仅是embedding层变化。但是target上的word2id还是有用的，在Ot映射出具体单词的时候，还是需要这个word2id来标明正确答案的。

关键需要理解charDecoder的实现机制：

在训练的时候并不是仅仅依赖unk来训练，而是全体生成词，都需要训练。真正的译文是没有unk的，所以全部单词都可以参与训练。

将charDecoder的输入数据变成两维，第一维是单个word中最大char数量，第二维是所有单词（不论是相同batch句子还是不同batch句子）。

训练过程中，假设本次batch有两句话，那么charDecoder的训练就是先通过基于word的Decoder得到combined_output, 然后利用view(-1, 256)处理成(38, 256)的形式，38意味着这两句话一共有38个word，256则是对应的每一个word放进去之后的output向量。

然后将charDecoder的输入数据处理好，即这两句话共有38个词，max_word_len是21。因此处理成shape=(38, 21)的形式，在放入charDecoder前，将隐藏状态调用.t()将shape变成(21, 38)。




### 三、QA
BLEU的计算方式：
https://blog.csdn.net/guolindonggld/article/details/56966200

PPL的计算方式：
https://blog.csdn.net/u012852385/article/details/81224558

## 课后习题
# CS 224n: Assignment #5

## 1

### (a)

&emsp;&emsp;Embedding size used for character-level embeddings is typically lower than that used for word embeddings. I guess the possible reason is that the size of chars-vocabulary is much smaller that the size of word-vocabulary. That means we don't need more dims to represent a concrete chars.

### (b)


&emsp;&emsp;For char-based embedding, total params:
$$
V_{char} \times e_{char} + e_{word} \times k \times e_{char} + 2 \times(e_{word} \times e_{word} + e_{word})
$$
&emsp;&emsp;First product is embedding lookup, second is convolution operation, third is Highway network
&emsp;&emsp;For word-based embedding, total params:
$$
V_{word} \times e_{word}
$$

&emsp;&emsp;If $k = 5, V_{word} ≈ 50,000$ and $V_{char} = 96$. It's obvious that word embedding model has more params and by $V_{word}$ the word embedding model is 100 times as many params as char-based model.

### (c)

&emsp;&emsp;As mentioned in lecture, one cell state or hidden state always be a mixture of the info from left and right, i.e. we get a fixed pattern or contextual info about one position. However, the filter of CNN just based on several words/chars in a window. That means we could not only make computation paralell but also have several filters to capture different feature/patterns we expect.

### (d)

&emsp;&emsp;This question may involve more knowledges about CNN. According to some theories, the error of features come from two aspects: $(1)$ increased variance of estimate caused by limited neighbourhood and $(2)$ the offset of mean caused by the error of params of conv layer. Max pooling could mitigate the first impact and preserve background info. Avg pooling could mitigate the second impact and retain texture.
&emsp;&emsp;In this question, we could have another eplaination for those two pooling methods.
&emsp;&emsp;**Max pooling**:

- Advantage: Get the strongest pattern in the data.
- Disadvantage: Discard most info in the data.

&emsp;&emsp;**Average pooling**:

- Advantage: Preseved all data info, because we make a average.
- Disadvantage: For other hand, if there is two many small values and only few big values, the strong signal dilutes and we just get a relatively small result(pattern or texture).

## 2

### (f)

$BLEU$ score is 24.36

## 3

### (a)

&emsp;&emsp;traducir and traduce are in word-vocabulary.
&emsp;&emsp;For word-based NMT, there may exist OOV problem while the decoder encounter word like 'traduzco' and decoder will predict 'unk' token.
&emsp;&emsp;The new char-aware NMT can overcome this problem because there is no 'unk' token in char-level vocab. While word-based NMT predict 'unk', char-level decoder takes it as input($h_0$ and $c_0$) to generate char sequence to replace 'unk' token.

### (b)

$\mathrm{i}.$

financial:
![finacial](../resource/financial.png)

neuron:
![neuron](../resource/neuron.png)

Francisco:
![Francisco](../resource/Francisco.png)

naturally:
![naturally](../resource/naturally.png)

expectation:
![expectation](../resource/expectation.png)

$\mathrm{ii}.$

financial:
![finacial](../resource/financial2.png)

neuron:
![neuron](../resource/neuron2.png)

Francisco:
![Francisco](../resource/Francisco2.png)

naturally:
![naturally](../resource/naturally2.png)

expectation:
![expectation](../resource/expectation2.png)

$\mathrm{iii}.$


|word|Word2Vec|CharCNN|
|:-:|:-:|:-:|
|financial|economic|informal|
|neuron|nerve|Newton|
|Francisco|san|France|
|naturally|occurring|practically|
|expectation|norms|exception|

&emsp;&emsp;Word2Vec models the sematic similarity and CharCNN models the structure similarity.
&emsp;&emsp;Recall Word2Vec, it think simIlar words have similar context, i.e. a word is represented by its context, while CharCNN extract info from a window-based feature(vector/matrix), so these words similar in word structure will be close in feature space.

### (c)

$\mathrm{i}.$
1. Hoy estoy aqu para hablarles sobre crculos y epifanas.
2. I'm here today to talk to you  about circles and epiphanies.
3. I'm here to talk to you about circles and <unk>
4. I'm here today to talk about circles and <u>episodes</u>.
5. Incorrect example. episode and epiphanies are similar in word structure.

$\mathrm{i}.$

1. Bien, al da siguiente estbamos en Cleveland.
2. Well, the next day we were in Cleveland.
3. Well, the next day we were in <unk>
4. Well, the next day we were in <u>Cleveland</u>, right?
5. Acceptable. Maybe any word in corpus isn't similar to Cleveland except itself.


#### 答案拷贝自https://github.com/ZacBi/CS224n-2019-solutions/blob/master/Assignments/written%20part/a5_solution.md并自己做了修改