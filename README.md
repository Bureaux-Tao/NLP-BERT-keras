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

