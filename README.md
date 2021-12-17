# Chinese NLP (albert/electra with Keras)

## Named Entity Recognization

### Project Structure

```
./
├── NER
│   ├── __init__.py
│   ├── log                                     训练nohup日志
│   │   ├── albert.out
│   │   ├── albert_crf.out
│   │   ├── electra.out
│   │   ├── electra_crf.out
│   │   ├── electra_regulization.out
│   │   └── electra_tiny.out
│   └── train.py
├── README.md
├── albert_base_google_zh                       albert_base权重
│   ├── albert_config.json
│   ├── albert_model.ckpt.data-00000-of-00001
│   ├── albert_model.ckpt.index
│   ├── checkpoint
│   └── vocab.txt
├── albert_tiny_google_zh                       albert_tiny权重
│   ├── albert_config.json
│   ├── albert_model.ckpt.data-00000-of-00001
│   ├── albert_model.ckpt.index
│   ├── checkpoint
│   └── vocab.txt
├── chinese_electra_small_ex_L-24_H-256_A-4     electra_small权重
│   ├── electra_small_ex.data-00000-of-00001
│   ├── electra_small_ex.index
│   ├── electra_small_ex.meta
│   ├── small_ex_discriminator_config.json
│   ├── small_ex_generator_config.json
│   └── vocab.txt
├── data                                        数据集
│   ├── pulmonary.test
│   ├── pulmonary.train
│   └── sict_train.txt
├── electra_180g_base                           electra_base权重
│   ├── base_discriminator_config.json
│   ├── base_generator_config.json
│   ├── electra_180g_base.ckpt.data-00000-of-00001
│   ├── electra_180g_base.ckpt.index
│   ├── electra_180g_base.ckpt.meta
│   └── vocab.txt
├── environment.yaml                            conda环境配置文件
├── main.py
├── path.py                                     所有路径
├── requirements.txt
├── utils                                       bert4keras包（也可pip下）
│   ├── __init__.py
│   ├── backend.py
│   ├── layers.py
│   ├── models.py
│   ├── optimizers.py
│   ├── snippets.py
│   └── tokenizers.py
└── weights                                     权重文件
    ├── pulmonary_albert_ner.h5
    ├── pulmonary_electra_ner.h5
    └── pulmonary_electra_tiny_ner_crf.h5

9 directories, 48 files
```

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
- 字与标签之间用空格（"\ "）隔开
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
