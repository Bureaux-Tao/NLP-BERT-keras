#! -*- coding: utf-8 -*-
# 用CRF做中文命名实体识别
# 数据集 http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# 实测验证集的F1可以到96.48%，测试集的F1可以到95.38%

import numpy as np
import os

from keras import regularizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.chdir(os.path.dirname(__file__))
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from path import train_file_path, test_file_path, BASE_MODEL_DIR, BASE_CONFIG_NAME, BASE_CKPT_NAME, event_type, \
    weights_path, MODEL_TYPE
from utils.backend import keras, K
from utils.models import build_transformer_model
from utils.tokenizers import Tokenizer
from utils.optimizers import Adam
from utils.snippets import sequence_padding, DataGenerator
from utils.snippets import open, ViterbiDecoder, to_array
from utils.layers import ConditionalRandomField
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Dropout
from keras.models import Model
from tqdm import tqdm

maxlen = 180
epochs = 300
batch_size = 64
bert_layers = 6
learning_rate = 1e-3  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 2000  # 必要时扩大CRF层的学习率
categories = set()

# bert配置
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)


def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding = 'utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'I':
                    d[-1][1] = i
            D.append(d)
    return D


# 标注数据
train_data = load_data(train_file_path)
test_data = load_data(test_file_path)
categories = list(sorted(categories))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case = True)


class data_generator(DataGenerator):
    """数据生成器
    """
    
    def __iter__(self, random = False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen = maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros(len(token_ids))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[start] = categories.index(label) * 2 + 1
                    labels[start + 1:end + 1] = categories.index(label) * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
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

# model = build_transformer_model(
#     config_path,
#     checkpoint_path,
#     model = MODEL_TYPE,
# )
# output_layer = 'Transformer-FeedForward-Norm'
# output = model.get_layer(output_layer).get_output_at(bert_layers - 1)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    model = MODEL_TYPE
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output


# output = Bidirectional(LSTM(64,
#                             return_sequences = True,
#                             dropout = 0.5,
#                             recurrent_dropout = 0.5,
#                             kernel_regularizer = keras.regularizers.l2(0.01)))(output)
output = TimeDistributed(Dense(len(categories) * 2 + 1))(output)
output = Dropout(0.5)(output)
# output = Dense(len(categories) * 2 + 1)(output)
CRF = ConditionalRandomField(lr_multiplier = crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
for layer in model.layers:
    layer.trainable = True
model.summary()

model.compile(
    loss = CRF.sparse_loss,
    optimizer = Adam(learning_rate),
    metrics = [CRF.sparse_accuracy]
)


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    
    def recognize(self, text):
        tokens = tokenizer.tokenize(text, maxlen = 512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], categories[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities]


NER = NamedEntityRecognizer(trans = K.eval(CRF.trans), starts = [0], ends = [0])


def evaluate(data, tqdm_verbose = False):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    if tqdm_verbose:
        for d in tqdm(data, ncols = 100):
            R = set(NER.recognize(d[0]))
            T = set([tuple(i) for i in d[1:]])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
    else:
        for d in data:
            R = set(NER.recognize(d[0]))
            T = set([tuple(i) for i in d[1:]])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    
    def __init__(self):
        self.best_val_f1 = 0
    
    def on_epoch_end(self, epoch, logs = None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        # print(NER.trans)
        f1, precision, recall = evaluate(train_data)
        # 保存最优
        # if f1 >= self.best_val_f1:
        #     self.best_val_f1 = f1
        #     save_file_path = "{}/{}_{}_ner.h5".format(weights_path, event_type, BASE_MODEL_DIR)
        #     model.save_weights(save_file_path)
        print(
            'train:  f1: %.5f, precision: %.5f, recall: %.5f' %
            (f1, precision, recall)
        )
        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f' %
            (f1, precision, recall)
        )


if __name__ == '__main__':
    save_file_path = "{}/{}_{}_tiny_ner_crf.h5".format(weights_path, event_type, MODEL_TYPE)
    
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)
    reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 3, verbose = 1)
    early_stopping = EarlyStopping(monitor = 'loss', patience = 10, verbose = 1)  # 提前结束
    save_model = ModelCheckpoint(save_file_path, monitor = 'loss', verbose = 1, save_best_only = True,
                                 mode = 'min')
    
    model.fit(
        train_generator.forfit(),
        steps_per_epoch = len(train_generator),
        epochs = epochs,
        callbacks = [evaluator, early_stopping, reduce_lr, save_model]
    )

else:
    save_file_path = "{}/{}_{}_tiny_ner_crf.h5".format(weights_path, event_type, MODEL_TYPE)
    model.load_weights(save_file_path)
    NER.trans = K.eval(CRF.trans)
