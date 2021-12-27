#! -*- coding:utf-8 -*-
# 句子对分类任务，LCQMC数据集
# val_acc: 0.887071, test_acc: 0.870320

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.callbacks import EarlyStopping, ModelCheckpoint

from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, train_file_path_Similarity, val_file_path_Similarity, \
    test_file_path_Similarity, MODEL_TYPE, weights_path, event_type
from utils.backend import keras, set_gelu, K
from utils.tokenizers import Tokenizer
from utils.models import build_transformer_model
from utils.optimizers import Adam, extend_with_piecewise_linear_lr
from utils.snippets import sequence_padding, DataGenerator
from utils.snippets import open
from keras.layers import Dropout, Dense

set_gelu('tanh')  # 切换gelu版本

# albert tiny

maxlen = 128
batch_size = 256
bert_layers = 3
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)


def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    with open(filename, encoding = 'utf-8') as f:
        for l in f:
            text1, text2, label = l.strip().split('\t')
            D.append((text1, text2, int(label)))
    return D


# 加载数据集
train_data = load_data(train_file_path_Similarity)
valid_data = load_data(val_file_path_Similarity)
test_data = load_data(test_file_path_Similarity)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case = True)


class data_generator(DataGenerator):
    """数据生成器
    """
    
    def __iter__(self, random = False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, maxlen = maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path = config_path,
    checkpoint_path = checkpoint_path,
    model = MODEL_TYPE,
    with_pool = True,
    return_keras_model = False,
)
# output_layer = 'Transformer-FeedForward-Norm'
# output = bert.get_layer(output_layer).get_output_at(bert_layers)

output = Dropout(rate = 0.1)(bert.model.output)
output = Dense(
    units = 2, activation = 'softmax', kernel_initializer = bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

AdamLR = extend_with_piecewise_linear_lr(Adam, name = 'AdamLR')

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = AdamLR(lr = 1e-4, lr_schedule = {
        1000: 1,
        2000: 0.1
    }),
    metrics = ['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis = 1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    
    def __init__(self):
        self.best_val_acc = 0.
    
    def on_epoch_end(self, epoch, logs = None):
        save_file_path = "{}/{}_similarity_max_acc.h5".format(weights_path, MODEL_TYPE)
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights(save_file_path)
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


if __name__ == '__main__':
    
    save_file_path = "{}/{}_similarity_min_loss.h5".format(weights_path, MODEL_TYPE)
    
    evaluator = Evaluator()
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1)  # 提前结束
    save_model = ModelCheckpoint(save_file_path, monitor = 'val_loss', verbose = 0, save_best_only = True,
                                 mode = 'min')
    
    for i, item in enumerate(train_generator):
        print("\nbatch_token_ids shape:", item[0][0].shape)
        print("batch_segment_ids shape:", item[0][1].shape)
        print("batch_labels shape:", item[1].shape)
        if i == 4:
            break
            
    # batch_token_ids shape: (256, 65)
    # batch_segment_ids shape: (256, 65)
    # batch_labels shape: (256, 1)
    
    # batch_token_ids shape: (256, 71)
    # batch_segment_ids shape: (256, 71)
    # batch_labels shape: (256, 1)
    
    # batch_token_ids shape: (256, 69)
    # batch_segment_ids shape: (256, 69)
    # batch_labels shape: (256, 1)
    
    model.fit(
        train_generator.forfit(),
        steps_per_epoch = len(train_generator),
        validation_data = valid_generator.forfit(),
        validation_steps = len(valid_generator),
        epochs = 100,
        callbacks = [evaluator, early_stopping, save_model]
    )
    
    model.load_weights(save_file_path)
    print(u'final min loss test acc: %05f\n' % (evaluate(test_generator)))
    
    save_file_path = "{}/{}_similarity_max_acc.h5".format(weights_path, MODEL_TYPE)
    model.load_weights(save_file_path)
    print(u'final max acc test acc: %05f\n' % (evaluate(test_generator)))

else:
    save_file_path = "{}/{}_similarity_min_loss.h5".format(weights_path, MODEL_TYPE)
    model.load_weights(save_file_path)
