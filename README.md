# Chinese NLP (albert/electra with Keras)

## Named Entity Recognization

### Dataset

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

```python
segmenter = WordSegmenter(tokenizer, model, trans = K.eval(CRF.trans), starts = [0], ends = [0])
txt = "张宝华当时是香港有线电视的时政记者，又就中央是否支持董建华连任特首一事向江泽民提问。江泽民面对记者的提问，先是风趣地以粤语“好啊”、“当然啦”回答，但随着记者提问为什么提前钦定董建华为特首，江泽民开始对“钦定”表露出不满情绪。他还称赞曾采访他的记者迈克·华莱士“比你们不知道高到哪里去了”。江泽民指责香港新闻界“你们有一个好，全世界跑到什么地方，你们比其他的西方记者跑得还快，但是问来问去的问题呀，都too simple, sometimes naive”。"
print("/ ".join(segmenter.tokenize(txt)))
```

结果

```
张/ 宝华/ 当时/ 是/ 香港/ 有线电视/ 的/ 时政/ 记者/ ，/ 又/ 就/ 中央/ 是否/ 支持/ 董/ 建华/ 连/ 任/ 特首一/ 事/ 向/ 江/ 泽民/ 提问/ 。/ 江/ 泽民/ 面对/ 记者/ 的/ 提问/ ，/ 先是/ 风趣/ 地/ 以/ 粤语/ “/ 好/ 啊/ ”/ 、/ “/ 当然/ 啦/ ”/ 回答/ ，/ 但/ 随着/ 记者/ 提问/ 为什么/ 提前/ 钦定/ 董/ 建华/ 为/ 特首/ ，/ 江/ 泽民/ 开始/ 对/ “/ 钦定/ ”/ 表露/ 出/ 不/ 满/ 情绪/ 。/ 他/ 还/ 称赞/ 曾/ 采访/ 他/ 的/ 记者/ 迈克·华莱士/ “/ 比/ 你们/ 不/ 知道/ 高/ 到/ 哪里/ 去/ 了/ ”/ 。/ 江/ 泽民/ 指责/ 香港/ 新闻界/ “/ 你们/ 有/ 一个/ 好/ ，/ 全世界/ 跑/ 到/ 什么/ 地方/ ，/ 你们/ 比/ 其他/ 的/ 西方/ 记者/ 跑/ 得/ 还/ 快/ ，/ 但是/ 问来/ 问/ 去/ 的/ 问题/ 呀/ ，/ 都/ too simple/ ,/ sometimes/ naive/ ”/ 。
```

## Knowledge Extraction

### Dataset

[http://ai.baidu.com/broad/download?dataset=sked](http://ai.baidu.com/broad/download?dataset=sked)

```json
{
   "postag" : [
      {
         "pos" : "w",
         "word" : "《"
      },
      {
         "pos" : "nw",
         "word" : "课本上学不到的生物学2"
      },
      {
         "pos" : "w",
         "word" : "》"
      },
      {
         "pos" : "v",
         "word" : "是"
      },
      {
         "pos" : "t",
         "word" : "2013年"
      },
      {
         "pos" : "nt",
         "word" : "上海科技教育出版社"
      },
      {
         "pos" : "v",
         "word" : "出版"
      },
      {
         "pos" : "u",
         "word" : "的"
      },
      {
         "pos" : "n",
         "word" : "图书"
      }
   ],
   "spo_list" : [
      {
         "object" : "上海科技教育出版社",
         "object_type" : "出版社",
         "predicate" : "出版社",
         "subject" : "课本上学不到的生物学2",
         "subject_type" : "书籍"
      }
   ],
   "text" : "《课本上学不到的生物学2》是2013年上海科技教育出版社出版的图书"
}
```

主要用`spo_list`的部分

### Train

运行KnowledgeExtraction/train.py

### Evaluate

F1，当且仅当S、P、O全部一致时才算True

```
Epoch 11/50
1352/1352 [==============================] - 158s 117ms/step - loss: 0.0263
f1: 0.78116, precision: 0.82647, recall: 0.74055: : 21626it [06:11, 58.14it/s]
best f1: 0.78116
```

## QA

### Dataset

SogouQA.json、WebQA.json数据集，忘了在哪下的了

```
[
    ,...,
    {
        "passages": [
            {
                "answer": "", 
                "passage": "许攸建议袁绍派轻骑攻袭许都,迎接天子讨伐曹操,曹操首尾难顾,必败无疑(嗅到了围魏救赵的气息),袁绍又没有听从意见,许攸生气下投奔了曹操。...建安二十二年为丁酉年,发生汉魏时期最大的瘟疫,故称“丁酉大疫”,建安七子中王粲等五人罹难于此(仲宣怀远更凄凉,好可惜) ..."
            }, 
            {
                "answer": "", 
                "passage": "诸侯争霸是发生在春秋时期的一起重大的政治事件。古代曾有“ 春秋五霸”之说,五霸的一般说法,是...围魏救赵第二卷诸侯纷起 第一百二十二章 大败沙家坝 第二卷诸侯纷起 第一百二十三章 ..."
            }, 
            {
                "answer": "战国时期", 
                "passage": "战国时期孔子和孟子悉尼李商隐和杜牧甲骨文愿乘风破万里浪 甘面壁读十年书风流人物数当代 大好春光看今朝人之学问知能成就,犹骨象玉石切磋琢磨也"
            }, 
            {
                "answer": "", 
                "passage": "围魏救赵是谁的故事?围魏救赵发生在什么时期_用户5692140230_新浪博客,用户5692140230,"
            }, 
            {
                "answer": "", 
                "passage": "许攸建议袁绍派轻骑攻袭许都,迎接天子讨伐曹操,曹操首尾难顾,必败无疑(嗅到了围魏救赵的气息),袁绍又没有听从意见,许攸生气下投奔了曹操。...建安二十二年为丁酉年,发生汉魏时期最大的瘟疫,故称“丁酉大疫”,建安七子中王粲等五人罹难于此(仲宣怀远更凄凉,好可惜) ..."
            }, 
            {
                "answer": "战国时期", 
                "passage": "历史趣闻 www.lishiquwen.com 分享: [导读] 导读:“围魏救赵”这句成语指避实就虚、袭击敌人后方以迫使进攻之敌撤回的战术。故事发生在战国时期的魏国国都大梁即现在的开封。 ..."
            }, 
            {
                "answer": "", 
                "passage": "示例:恩利科的一篇日记中讲述了一则发生在教室里科罗西用墨水瓶砸弗朗蒂的故事。因为这则故事借老师之口批评了欺凌弱小者的不光彩行为,赞美了卡罗纳的勇敢、善良、宽厚。 ..."
            }, 
            {
                "answer": "战国时", 
                "passage": "围魏救赵原指战国时齐军用围攻魏国的方法,迫使魏国撤回攻赵部队而使赵国得救。[1] 后指袭击敌人后方的据点以迫使进攻之敌撤退的战术。现借指用包抄敌人的后方来迫使他..."
            }, 
            {
                "answer": "战国时期", 
                "passage": " 故事发生在战国时期的魏国国都大梁即现在的开封。因魏国国都在大梁所以魏国也称梁国,其国君魏惠王也称梁惠王。..."
            }, 
            {
                "answer": "", 
                "passage": "围魏救赵是谁的故事?围魏救赵发生在什么时期_用户5692140230_新浪博客,用户5692140230,"
            }
        ], 
        "question": "围魏救赵发生在哪个时期", 
        "id": "10002"
    }, 
    ,...,
]
```

load_data需要把数据处理成`(文章, 问题, 答案)`的形式

### Train

运行QA/train.py

### Evaluate

根据loss设置早停

```
Epoch 99/100
1000/1000 [==============================] - 814s 814ms/step - loss: 0.3919
```

### Predict

运行QA/predict.py

```python
qag.generate("1915年12月13日，时任中华民国大总统的袁世凯宣布废除共和政体，实行帝制，改国号为中华帝国，年号洪宪，自任皇帝；1916年3月22日，宣布废除帝制，重归共和，前后总共历时83天。")
```
根据问题生成问题并抽取答案


输出：
```
2022-05-07 19:25:51.911963: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcublas.so.10.0
('袁世凯当上中华帝国的黄帝是在', '1915')
```