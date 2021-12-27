#! -*- coding: utf-8 -*-
# 用CRF做中文分词（CWS, Chinese Word Segmentation）
# 数据集 http://sighan.cs.uchicago.edu/bakeoff2005/
# 最后测试集的F1约为96.1%

import re, os, json
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping

from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, train_file_path_Segmentation, weights_path, \
    MODEL_TYPE
from utils.backend import keras, K
from utils.models import build_transformer_model
from utils.tokenizers import Tokenizer
from utils.optimizers import Adam, extend_with_piecewise_linear_lr
from utils.snippets import sequence_padding, DataGenerator
from utils.snippets import open, ViterbiDecoder, to_array
from utils.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm

maxlen = 200
epochs = 100
num_labels = 4
batch_size = 256
bert_layers = 4
learning_rate = 1e-4  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1  # 必要时扩大CRF层的学习率

# bert配置
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)


def load_data(filename):
    """加载数据
    单条格式：[词1, 词2, 词3, ...]
    """
    D = []
    with open(filename, encoding = 'utf-8') as f:
        for l in f:
            D.append(re.split(' +', l.strip()))
    return D


# 标注数据
data = load_data(train_file_path_Segmentation)

# 保存一个随机序（供划分valid用）
if not os.path.exists('./random_order.json'):
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    json.dump(random_order, open('./random_order.json', 'w'), indent = 4)
else:
    random_order = json.load(open('./random_order.json'))

# 划分valid
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case = True)


class data_generator(DataGenerator):
    """数据生成器
    """
    
    def __iter__(self, random = False):
        """标签含义
        0: 单字词； 1: 多字词首字； 2: 多字词中间； 3: 多字词末字
        """
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if len(w_token_ids) == 1:
                        labels += [0]
                    else:
                        labels += [1] + [2] * (len(w_token_ids) - 2) + [3]
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                # [0, 1, 3, 1, 3, 1, 2, 3, 1, 3, 1, 3, 0]
                # 0表示单个字或填充字符，1表示多字的起始字符，2表示多字的中间字符，3表示多字的终止字符
                
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


"""
后面的代码使用的是bert类型的模型，如果你用的是albert，那么前几行请改为：
model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='albert',
)
output_layer = 'Transformer-FeedForward-Norm'
output = model.get_layer(output_layer).get_output_at(bert_layers - 1)
"""

model = build_transformer_model(
    config_path,
    checkpoint_path,
    model = MODEL_TYPE,
)
output_layer = 'Transformer-FeedForward-Norm'
output = model.get_layer(output_layer).get_output_at(bert_layers - 1)
output = Dense(num_labels)(output)
CRF = ConditionalRandomField(lr_multiplier = crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

AdamLR = extend_with_piecewise_linear_lr(Adam, name = 'AdamLR')

model.compile(
    loss = CRF.sparse_loss,
    optimizer = AdamLR(lr = learning_rate, lr_schedule = {
        1000: 1,
        2000: 0.1
    }),
    metrics = [CRF.sparse_accuracy]
)


class WordSegmenter(ViterbiDecoder):
    """基本分词器
    """
    
    def tokenize(self, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        words = []
        for i, label in enumerate(labels[1:-1]):
            if label < 2 or len(words) == 0:
                words.append([i + 1])
            else:
                words[-1].append(i + 1)
        return [text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1] for w in words]


segmenter = WordSegmenter(trans = K.eval(CRF.trans), starts = [0], ends = [0])


def simple_evaluate(data, tqdm_verbose = True):
    """简单的评测
    该评测指标不等价于官方的评测指标，但基本呈正相关关系，
    可以用来快速筛选模型。
    """
    total, right = 0., 0.
    
    if tqdm_verbose:
        for w_true in tqdm(data):
            w_pred = segmenter.tokenize(''.join(w_true))
            w_pred = set(w_pred)
            w_true = set(w_true)
            total += len(w_true)
            right += len(w_true & w_pred)
    
    else:
        for w_true in data:
            w_pred = segmenter.tokenize(''.join(w_true))
            w_pred = set(w_pred)
            w_true = set(w_true)
            total += len(w_true)
            right += len(w_true & w_pred)
    return right / total


def predict_to_file(in_file, out_file):
    """预测结果到文件，便于用官方脚本评测
    使用示例：
    predict_to_file('/root/icwb2-data/testing/pku_test.utf8', 'myresult.txt')
    官方评测代码示例：
    data_dir="/root/icwb2-data"
    $data_dir/scripts/score $data_dir/gold/pku_training_words.utf8 $data_dir/gold/pku_test_gold.utf8 myresult.txt > myscore.txt
    （执行完毕后查看myscore.txt的内容末尾）
    """
    fw = open(out_file, 'w', encoding = 'utf-8')
    with open(in_file, encoding = 'utf-8') as fr:
        for l in tqdm(fr):
            l = l.strip()
            if l:
                l = ' '.join(segmenter.tokenize(l))
            fw.write(l + '\n')
    fw.close()


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    
    def __init__(self):
        self.best_val_acc = 0
    
    def on_epoch_end(self, epoch, logs = None):
        trans = K.eval(CRF.trans)
        segmenter.trans = trans
        # print(segmenter.trans)
        acc = simple_evaluate(valid_data, tqdm_verbose = False)
        # 保存最优
        if acc >= self.best_val_acc:
            self.best_val_acc = acc
            save_file_path = "{}/{}_segmentation_tiny.h5".format(weights_path, MODEL_TYPE)
            model.save_weights(save_file_path)
        print('acc: %.5f, best acc: %.5f' % (acc, self.best_val_acc))


if __name__ == '__main__':
    
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(valid_data, batch_size)
    
    save_file_path = "{}/{}_segmentation_tiny.h5".format(weights_path, MODEL_TYPE)
    
    # save_model = ModelCheckpoint(save_file_path, monitor = 'val_loss', verbose = 0, save_best_only = True,
    #                              mode = 'min')
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1)  # 提前结束
    
    for i, item in enumerate(train_generator):
        print("\nbatch_token_ids shape:", item[0][0].shape)
        print("batch_segment_ids shape:", item[0][1].shape)
        print("batch_labels shape:", item[1].shape)
        if i == 4:
            break
    
    # batch_token_ids shape: shape: (128, 256)
    # batch_segment_ids shape: (128, 256)
    # batch_labels shape: (128, 256)
    
    model.fit(
        train_generator.forfit(),
        steps_per_epoch = len(train_generator),
        validation_data = valid_generator.forfit(),
        validation_steps = len(valid_generator),
        epochs = epochs,
        callbacks = [evaluator, early_stopping]
    )

else:
    save_file_path = "{}/{}_segmentation_tiny.h5".format(weights_path, MODEL_TYPE)
    model.load_weights(save_file_path)
    segmenter.trans = K.eval(CRF.trans)

