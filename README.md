# Chinese NLP (albert/electra with Keras)

## Named Entity Recognization

### Dataset

三甲医院肺结节数据集，20000+字，BIO格式，形如：

```
中	B-ORG
共	I-ORG
中	I-ORG
央	I-ORG
致	O
中	B-ORG
国	I-ORG
致	I-ORG
公	I-ORG
党	I-ORG
十	I-ORG
一	I-ORG
大	I-ORG
的	O
贺	O
词	O
```
ATTENTION: 在处理自己数据集的时候需要注意：
- 字与标签之间用tab（"\t"）隔开
- 其中句子与句子之间使用空行隔开

### Steps

1. 替换数据集
2. 修改NER/train.py中的maxlen（超过截断，少于填充，最好设置训练集、测试集中最长句子作为MAX_SEQ_LEN）
3. 下载权重，放到项目中
4. 修改path.py中的地址
5. 根据需要修改NER/train.py模型结构
6. 训练前**debug看下train_generator数据**
7. 训练

### Model

[albert](https://github.com/bojone/albert_zh)

[electra](https://github.com/ymcui/Chinese-ELECTRA)

### Train

运行NER/train.py

### Evaluate

train时给出的F1即为实体级别的F1

albert最佳F1

```
Epoch 61/300
13/13 [==============================] - 16s 1s/step - loss: 0.1343 - sparse_accuracy: 0.9713
test:  f1: 0.82428, precision: 0.81775, recall: 0.83092
```

electra

```
Epoch 29/300
13/13 [==============================] - 16s 1s/step - loss: 0.3487 - sparse_accuracy: 0.9146
test:  f1: 0.83189, precision: 0.81579, recall: 0.84863
```

## Text Classification

### Dataset

某网站评论数据集，0中性，1褒，2贬

```
评论内容\t评分
性价比很高，金属外壳很有质感。喇叭的效果很好，外放音量很足。	1
太缺德了，蒙牛怎么不倒闭！！！	2
```
ATTENTION: 在处理自己数据集的时候需要注意：
- 句子与标签之间用tab（"\t"）隔开
- 此数据集已做最大长度截断


### Steps

1. 替换数据集
2. 修改NER/train.py中的maxlen（超过截断，少于填充，最好设置训练集、测试集中最长句子作为MAX_SEQ_LEN）
3. 下载权重，放到项目中
4. 修改path.py中的地址
5. 根据需要修改Classification/train.py模型结构
6. 训练前**debug看下train_generator数据**
7. 训练

### Model

[electra](https://github.com/ymcui/Chinese-ELECTRA)

### Train

运行Classification/train.py

### Evaluate

train时给出的acc

val最佳acc

```
134/134 [==============================] - 88s 660ms/step - loss: 0.1549 - acc: 0.9259 - val_loss: 0.0529 - val_acc: 0.9707
val_acc: 0.97070, best_val_acc: 0.97070, test_acc: 0.93549
```

测试集acc

```
final test acc: 0.943433
```

## Sementic Similarity
### Dataset

LCQMC数据集，0不相似，1相似

```
句子1\t橘子2\t是否相似
怎样知道自己驾照还有多少分？	怎么查驾驶证里有多少分，？	1
今天离过年还有多少天	大学放假还有多少天？	0
```
ATTENTION: 在处理自己数据集的时候需要注意：
- 句子与标签之间用tab（"\t"）隔开


### Steps

1. 替换数据集
2. 修改Similarity/train.py中的maxlen（超过截断，少于填充，最好设置训练集、测试集中最长句子作为MAX_SEQ_LEN）
3. 下载权重，放到项目中
4. 修改path.py中的地址
5. 根据需要修改Similarity/train.py模型结构
6. 训练前**debug看下train_generator数据**
7. 训练

### Model

[electra](https://github.com/ymcui/Chinese-ELECTRA)

### Train

运行Similarity/train.py

### Evaluate

train时给出的acc

val最佳acc

```
933/933 [==============================] - 138s 148ms/step - loss: 0.1199 - acc: 0.9555 - val_loss: 0.4887 - val_acc: 0.8383
val_acc: 0.83833, best_val_acc: 0.83833, test_acc: 0.84568
```

测试集acc

```
final min loss test acc: 0.850960
final max acc test acc: 0.845680
```

## Segmentation

### Dataset

[ICWB2数据集](http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip)
一句样本一行，分词的地方以两个空格隔开，程序中超过1个空格即视为分词

```
由于  西南  暖湿气流  较为  旺盛  ，  ６日  —  ８日  ，  我国  南方  大部  地区  仍  为  阴雨  天气  ，  一般  有  小到中雨  （  雪  ）  ，  部分  地区  有  大雨  ，  但  气温  较为  平稳  。
```

batch_labels标签形式：[0, 1, 3, 1, 3, 1, 2, 3, 1, 3, 1, 3, 0]

0表示单个字或填充字符，1表示多字的起始字符，2表示多字的中间字符，3表示多字的终止字符

### Steps

1. 替换数据集
2. 修改Segmentation/train.py中的maxlen、num_labels、batch_size、bert_layers、learning_rate、crf_lr_multiplier等参数
3. 下载权重，放到项目中
4. 修改path.py中的地址
5. 根据需要修改Segmentation/train.py模型结构
6. 训练前**debug看下train_generator数据**
7. 训练

### Model

[albert](https://github.com/bojone/albert_zh)

### Train

运行Segmentation/train.py

### Evaluate

train时给出的acc(仅计算了acc，应该是要用F1的)

val最佳acc

```
Epoch 19/100
67/67 [==============================] - 41s 605ms/step - loss: 2.9961 - sparse_accuracy: 0.9907 - val_loss: 12.7162 - val_sparse_accuracy: 0.9653
acc: 0.94867, best acc: 0.94867
```

### Predict

运行Segmentation/predict.py

```python
segmenter = WordSegmenter(tokenizer, model, trans = K.eval(CRF.trans), starts = [0], ends = [0])
txt = "张宝华当时是香港有线电视的时政记者，又就中央是否支持董建华连任特首一事向江泽民提问。江泽民面对记者的提问，先是风趣地以粤语“好啊”、“当然啦”回答，但随着记者提问为什么提前钦定董建华为特首，江泽民开始对“钦定”表露出不满情绪。他还称赞曾采访他的记者迈克·华莱士“比你们不知道高到哪里去了”。江泽民指责香港新闻界“你们有一个好，全世界跑到什么地方，你们比其他的西方记者跑得还快，但是问来问去的问题呀，都too simple, sometimes naive”。"
print("/ ".join(segmenter.tokenize(txt)))
```

结果

```
张/ 宝华/ 当时/ 是/ 香港/ 有线电视/ 的/ 时政/ 记者/ ，/ 又/ 就/ 中央/ 是否/ 支持/ 董/ 建华/ 连/ 任/ 特首一/ 事/ 向/ 江/ 泽民/ 提问/ 。/ 江/ 泽民/ 面对/ 记者/ 的/ 提问/ ，/ 先是/ 风趣/ 地/ 以/ 粤语/ “/ 好/ 啊/ ”/ 、/ “/ 当然/ 啦/ ”/ 回答/ ，/ 但/ 随着/ 记者/ 提问/ 为什么/ 提前/ 钦定/ 董/ 建华/ 为/ 特首/ ，/ 江/ 泽民/ 开始/ 对/ “/ 钦定/ ”/ 表露/ 出/ 不/ 满/ 情绪/ 。/ 他/ 还/ 称赞/ 曾/ 采访/ 他/ 的/ 记者/ 迈克·华莱士/ “/ 比/ 你们/ 不/ 知道/ 高/ 到/ 哪里/ 去/ 了/ ”/ 。/ 江/ 泽民/ 指责/ 香港/ 新闻界/ “/ 你们/ 有/ 一个/ 好/ ，/ 全世界/ 跑/ 到/ 什么/ 地方/ ，/ 你们/ 比/ 其他/ 的/ 西方/ 记者/ 跑/ 得/ 还/ 快/ ，/ 但是/ 问来/ 问/ 去/ 的/ 问题/ 呀/ ，/ 都/ too simple/ ,/ sometimes/ naive/ ”/ 。
```