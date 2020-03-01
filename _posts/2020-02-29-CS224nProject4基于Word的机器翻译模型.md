---
layout:     post
title:      CS224n:Project4基于Word的机器翻译模型
subtitle:   CS224n系列
date:       2020-02-29
author:     chen cheng
header-img: img/post-bg-unix-linux.jpg
catalog: true
tags:
    - 项目整理
---

## 一、项目介绍
![a4_model.jpg](/img/2020-02-29/a4_model.jpg)

本Project使用带有**全局注意力机制**的**seq2seq**模型实现了西班牙语（source，src）到英语（target，tgt）的机器翻译功能。因此对于**python**以及**pytorch**的使用会更加熟练，对相关模型的细节理解得更加深刻。

OOV问题（Out Of Memory问题）

## 二、源代码分析
### 1.如何将文本资源（西班牙语和英语的翻译训练集）处理成Vocabulary类对象？（提取文本资源中频率最高的一部分单词并分配对应id）

![a4_train_en.jpg](/img/2020-02-29/a4_train_en.jpg)

训练集有21万个句子。每行一句话，上图为英语文本，另一文件是对应的源语言西班牙语文本。

![a4_train_en.jpg](/img/2020-02-29/a4_test_es.jpg)

测试集有8064个句子，句子数是训练集的3.7%。下面是训练集文本变成Vocab类对象的流程（以英文target举例，西班牙语source相同）：

（1）读取文件，将句子变成List\<List\<str\>\>格式并在target训练集每句首尾加上字符'\<s\>'以及'\</s\>'：
![a4_read_corpus.jpg](/img/2020-02-29/a4_read_corpus.jpg)

重点API：
- line 53: data.append(sent)
  
  向一个list中append一个list的时候，原封不动地将list添加进去，而不是将list的每一个item添加进去。

（2）从List\<List\<str\>\>格式的corpus中提取频率最高的5w个单词word，并存到一个VocabEntry类对象中。
![a4_get_vocab_entry.jpg](/img/2020-02-29/a4_get_vocab_entry.jpg)

关于VocabEntry对象：它包含有一个**word2id词典**和**id2word词典**，记录单词word和其id的对应关系。当调用vocab_entry.add(word)方法的时候其实就是在这两个词典中添加一个item。整个项目中有一个Vocab对象，它包含两个VocabEntry对象（src和tgt），分别保存西班牙语和英语的单词id对应关系。

重点API：
- line 141: word_freq = Counter(chain(*corpus))
  
  Counter类来自python的collections包，chain类来自python的itertools包。chain用来串联多个可迭代对象以形成一个更大的迭代对象。因此corpus前加*，就形成了多个可迭代list，每个list代表一句话，再加上chain的意思就是将所有句子串联成一个可迭代对象。而构建Counter的构造函数的参数如果是一个可迭代对象，那么它将计算每个item及其出现次数。Counter对象很像一个词典。

- line 145: sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
  利用key参数的lambda表达式对valid_words从大到小排列，并取前size个词word。

（3）创建Vocab对象，包含src、tgt两个VocabEntry对象。
![a4_get_vocab.jpg](/img/2020-02-29/a4_get_vocab.jpg)

现在，一个保存了频率最高的单词及其id的相互映射关系的Vocab对象就从文本文件得到了。我们可以将生成的Vocab保存成文本文件，这样下次直接读文件即可，而不用重新统计词频并排序。

![a4_vocab_json.jpg](/img/2020-02-29/a4_vocab_json.jpg)


### 2. 如何将句子按批处理成模型输入的？即Model的forward方法的参数。
``` 
train_data_src = read_corpus(args['--train-src'], source='src')  
# list<list(str)>   len:21w
train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')
train_data = list(zip(train_data_src, train_data_tgt))
```

重点API：
- train_data = list(zip(train_data_src, train_data_tgt))
  
  python3中zip函数返回的是一个zip迭代对象，一般需要配合list生成list对象。zip使得参数中多个List\<str\>对应索引位置生成tuple。

现在的train_data形式是这样的：
[(["I","study","nlp"]，["我","学习","nlp"])，(...)]

``` 
while True:
    epoch += 1

    for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
        example_losses = -model(src_sents, tgt_sents)
```
可见batch_iter这个函数对于train_data的处理是很重要的，而这个函数是utils里自己写的方法。

![a4_batch_iter.jpg](/img/2020-02-29/a4_batch_iter.jpg)

可以看到生成的batch是按照src句子包括的单词个数来排序的，从长到短排序。这时，yield返回的两个值就可以作为Model的forward函数中的参数了。



### 3. 模型的forward函数具体做了什么？（forward函数非常重要，可从forward的返回值中计算出loss损失以进行反向传播和训练）
![a4_model_forward.jpg](/img/2020-02-29/a4_model_forward.jpg)
下面对重要的代码依次分析：
``` 
source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)  str->int
``` 
source是一个batch大小的源句子列表。通过VocabEntry类的to_input_tensor(source)将其转换为经过了padded的一个tensor列表。

![a4_to_input_tensor.jpg](/img/2020-02-29/a4_to_input_tensor.jpg)
words2indices(sents)是非常容易实现的，因为VocabEntry包含了单词到id的映射关系，所以遍历sents然后一一对应再生成一个id的列表就好了。
![a4_words2indices.jpg](/img/2020-02-29/a4_words2indices.jpg)
pad_sents是利用\<pad\>对应的id对所有句子进行长度补齐的操作，是重要的一个方法，如果不补齐，后续无法进行批处理。
![a4_pad_sents.jpg](/img/2020-02-29/a4_pad_sents.jpg)
pad_sents的实现代码可以再优化，因为最长的肯定是第一句，之前已经排过序了。（译文句子也需要补齐，但是译文最长就不一定是第一句了，所以必须这么写，不能修改）。

注意to_input_tensor在最后返回的时候，返回的是**torch.t(sents_var)**, 这就将sents_var的shape从(batch, max_sentence_len)变成了(max_sentence_len, batch)，这是为了方便将来传入LSTM中。 

以上，VocabEntry的to_inuput_tensor方法完成了，最后返回的是已经被补齐了的由index组成的batch大小的tensor，即source_padded。

``` 
enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
``` 

----------------------------
### 4. 模型的encode函数具体做了什么？（forward函数中调用了encode）
encode的两个参数：source_padded是被补齐的，每句话的真实长度由source_lengths记录。
![a4_encode_1.jpg](/img/2020-02-29/a4_encode_1.jpg)
![a4_encode_2.jpg](/img/2020-02-29/a4_encode_2.jpg)
model_embeddings是一个继承了nn.Module的类，拥有两个nn.Embeddings类，它比较重要的代码：
``` 
line 175:内部重要代码
self.source = nn.Embedding(len(vocab.src), self.embed_size, padding_idx=src_pad_token_idx)
self.target = nn.Embedding(len(vocab.tgt), self.embed_size, padding_idx=tgt_pad_token_idx)
```
Embedding层可以指定padding_idx，这样其对应的W参数都是固定的0，无需训练.
``` 
line 176:
X = pack_padded_sequence(X, lengths=source_lengths)             # if feed back to RNN, it will not calculate output for pad element
``` 
pack_padded_sequence, pad_packed_sequence都是torch.nn.utils.rnn包中的函数。十分有用。

![a4_pack_padded.jpg](/img/2020-02-29/a4_pack_padded.jpg)
会生成一个PackedSequence对象，有data和batch_sizes两个tensor属性。

下一步是将该PackedSequence对象放入encoder（LSTM）中，之后返回enc_hiddens, (last_hidden, last_cell)。将enc_hiddens进行pad_packed_sequence进行pad还原成正常tensor格式后，将第一维第二维转换，即可作为返回值返回了。

然后将2h维度的隐藏层h和c做一个2h到h维度的变换，即可作为decoder的隐藏状态的初始化值了。

所以重要的是，了解LSTM的forward的输入格式和返回值，输入值是(max_sentence_len, batch, embedding)，也可以进行pack_padded_sequence，返回值是enc_hiddens, (last_hidden, last_cell)。如果传入时pack了，那么enc_hiddens还需要pad_packed，用0去填充。元组中的last_hidden，如果是biLSTM，那么它有两个向量，是双向的最后一个隐藏层向量，我们可以手动使用torch.cat对这两个tensor向量进行连接。


### 5. 模型的decode函数具体做了什么？（forward函数中调用了decode）
![a4_decode_1.jpg](/img/2020-02-29/a4_decode_1.jpg)
![a4_decode_2.jpg](/img/2020-02-29/a4_decode_2.jpg)
参数target_padded就是原句子前后加上\<s\>\</s\>后，再使用\<pad\>进行补齐后的目标语言答案。




--------------------------------------
### 6. 模型的step函数具体做了什么？（decode函数中调用了多次step函数）

从这里也看出了LSTM和LSTMCell的区别到底是什么。主要是forward函数不同。前者可以把整个timestep输入，返回值则是所有的h等；而后者只是把单步数据输入，输出单步的隐藏层数据。

``` 
enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
``` 
![a4_generate_sent_mask.jpg](/img/2020-02-29/a4_generate_sent_mask.jpg)
enc_masks的shape是(max_sentence_len, batch)。1代表有单词，0代表是pad。

``` 
combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
``` 

``` 
P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)
``` 