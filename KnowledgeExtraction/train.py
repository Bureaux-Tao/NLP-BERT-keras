#! -*- coding:utf-8 -*-
# 三元组抽取任务，基于“半指针-半标注”结构
# 文章介绍：https://kexue.fm/archives/7161
# 数据集：http://ai.baidu.com/broad/download?dataset=sked
# 最优f1=0.82198
# 换用RoBERTa Large可以达到f1=0.829+
# 说明：由于使用了EMA，需要跑足够多的步数(5000步以上）才生效，如果
#      你的数据总量比较少，那么请务必跑足够多的epoch数，或者去掉EMA。

import json

import numpy as np
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from keras.utils import plot_model
from tqdm import tqdm

from path import *
from utils.backend import keras, K, batch_gather
from utils.layers import LayerNormalization
from utils.layers import Loss
from utils.models import build_transformer_model
from utils.optimizers import Adam, extend_with_exponential_moving_average
from utils.snippets import open, to_array
from utils.snippets import sequence_padding, DataGenerator
from utils.tokenizers import Tokenizer

maxlen = 128
batch_size = 128
learning_rate = 1e-4
# bert配置
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [(s, p, o)]}
    """
    D = []
    with open(filename, encoding = 'utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'text': l['text'],
                'spo_list': [(spo['subject'], spo['predicate'], spo['object'])
                             for spo in l['spo_list']]
            })
    return D


# 加载数据集
train_data = load_data(train_file_path_KE)
valid_data = load_data(val_file_path_KE)
predicate2id, id2predicate = {}, {}

with open(schemas_KE, encoding = 'utf-8') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case = True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """
    
    def __iter__(self, random = False):
        batch_token_ids, batch_segment_ids = [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(d['text'], maxlen = maxlen)
            # 整理三元组 {s: [(o, p)]}
            spoes = {}
            for s, p, o in d['spo_list']:
                s = tokenizer.encode(s)[0][1:-1]
                p = predicate2id[p]
                o = tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1
                # 随机选一个subject（这里没有实现错误！这就是想要的效果！！）
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)
                # 对应的object标签
                object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1
                # 构建batch
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_subject_labels = sequence_padding(
                        batch_subject_labels
                    )
                    batch_subject_ids = np.array(batch_subject_ids)
                    batch_object_labels = sequence_padding(batch_object_labels)
                    yield [
                              batch_token_ids,  # 输入文本(batch_size, max_len)
                              batch_segment_ids,  # BERT特产(batch_size, max_len)
                              batch_subject_labels,  # S的标签(batch_size, max_len, 2) 最后一维2:(是否头,是否尾)
                              batch_subject_ids,  # (batch_size, 2) S使用search函数在batch_token_ids的起始坐标、终止坐标
                              batch_object_labels  # O的标签(batch_size, max_len, schema, 2) 最后一维2:(是否头,是否尾)
                          ], None
                    
                    # subject_labels -> batch_token_ids
                    # object_labels -> batch_segment_ids
                    # subject_preds -> batch_subject_labels
                    # object_preds -> batch_subject_ids
                    # bert.model.output -> batch_object_labels
                    
                    batch_token_ids, batch_segment_ids = [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []


def extract_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    把预测的subject对应的embedding与bert输出的hidden-states 进行拼接
    """
    output, subject_ids = inputs  # output:(?, ?, 312)    subject_ids:(?, 2)
    # Embedding - Token 出来的每个字都是312维
    # batch_gather
    # seq = [1,2,3,4,5,6]
    # idxs = [2,3,4]
    # a = K.tf.batch_gather(seq, idxs)
    # [3 4 5]
    start = batch_gather(output, subject_ids[:, :1])  # 取头指针 (?, 1, 312)
    end = batch_gather(output, subject_ids[:, 1:])  # 取尾指针 (?, 1, 312)
    subject = K.concatenate([start, end], 2)  # (?, 1, 624)
    return subject[:, 0]  # (?, 624)


# 补充输入
subject_labels = Input(shape = (None, 2), name = 'Subject-Labels')
subject_ids = Input(shape = (2,), name = 'Subject-Ids')
object_labels = Input(shape = (None, len(predicate2id), 2), name = 'Object-Labels')

# 加载预训练模型
bert = build_transformer_model(
    config_path = config_path,
    checkpoint_path = checkpoint_path,
    return_keras_model = False,
    model = MODEL_TYPE
)

# 预测subject
output = Dense(
    units = 2, activation = 'sigmoid', kernel_initializer = bert.initializer
)(bert.model.output)
subject_preds = Lambda(lambda x: x ** 2)(output)
# 我们引入了Bert作为编码器，然后得到了编码序列bert.model.output，然后直接接一个输出维度是2的Dense来预测首尾

subject_model = Model(bert.model.inputs, subject_preds)

# 传入subject，预测object

# 为了取Transformer-11-FeedForward-Add做Conditional Layer Normalization
# 做Conditional Layer Normalization就不需要Transformer-11-FeedForward-Norm了
output = bert.model.layers[-2].get_output_at(-1)
# 把传入的s的首尾对应的编码向量拿出来,直接加到编码向量序列t中去
subject = Lambda(extract_subject)([output, subject_ids])
# 通过Conditional Layer Normalization将subject融入到object的预测中
output = LayerNormalization(conditional = True)([output, subject])
output = Dense(
    units = len(predicate2id) * 2,
    activation = 'sigmoid',
    kernel_initializer = bert.initializer
)(output)
output = Lambda(lambda x: x ** 4)(output)
object_preds = Reshape((-1, len(predicate2id), 2))(output)

object_model = Model(bert.model.inputs + [subject_ids], object_preds)


class TotalLoss(Loss):
    """subject_loss与object_loss之和，都是二分类交叉熵
    """
    
    def compute_loss(self, inputs, mask = None):
        subject_labels, object_labels = inputs[:2]
        subject_preds, object_preds, _ = inputs[2:]
        if mask[4] is None:
            mask = 1.0
        else:
            mask = K.cast(mask[4], K.floatx())
        # sujuect部分loss
        subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
        subject_loss = K.mean(subject_loss, 2)
        subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
        # object部分loss
        object_loss = K.binary_crossentropy(object_labels, object_preds)
        object_loss = K.sum(K.mean(object_loss, 3), 2)
        object_loss = K.sum(object_loss * mask) / K.sum(mask)
        # 总的loss
        return subject_loss + object_loss


subject_preds, object_preds = TotalLoss([2, 3])([
    subject_labels, object_labels, subject_preds, object_preds,
    bert.model.output
])

# 训练模型
train_model = Model(
    bert.model.inputs + [subject_labels, subject_ids, object_labels],
    [subject_preds, object_preds]
)

plot_model(train_model, to_file = './images/model.png', show_shapes = True)
train_model.summary()

AdamEMA = extend_with_exponential_moving_average(Adam, name = 'AdamEMA')
optimizer = AdamEMA(lr = learning_rate)
train_model.compile(optimizer = optimizer)


def extract_spoes(text):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen = maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen = maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    # 抽取subject
    subject_preds = subject_model.predict([token_ids, segment_ids])
    subject_preds[:, [0, -1]] *= 0
    start = np.where(subject_preds[0, :, 0] > 0.6)[0]
    end = np.where(subject_preds[0, :, 1] > 0.5)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    if subjects:
        spoes = []
        token_ids = np.repeat(token_ids, len(subjects), 0)
        segment_ids = np.repeat(segment_ids, len(subjects), 0)
        subjects = np.array(subjects)
        # 传入subject，抽取object和predicate
        object_preds = object_model.predict([token_ids, segment_ids, subjects])
        object_preds[:, [0, -1]] *= 0
        for subject, object_pred in zip(subjects, object_preds):
            start = np.where(object_pred[:, :, 0] > 0.6)
            end = np.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((mapping[subject[0]][0],
                              mapping[subject[1]][-1]), predicate1,
                             (mapping[_start][0], mapping[_end][-1]))
                        )
                        break
        return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )
    
    def __hash__(self):
        return self.spox.__hash__()
    
    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(val_file_path_KE, 'w', encoding = 'utf-8')
    pbar = tqdm()
    for d in data:
        R = set([SPO(spo) for spo in extract_spoes(d['text'])])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
            ensure_ascii = False,
            indent = 4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall


count_model_did_not_improve = 0


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    
    def __init__(self, patience = 1):
        super().__init__()
        self.best_val_f1 = 0.
        self.patience = patience
    
    def on_epoch_end(self, epoch, logs = None):
        global count_model_did_not_improve
        save_file_path = "{}/{}_{}.h5".format(weights_path, event_type, MODEL_TYPE)
        optimizer.apply_ema_weights()
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            count_model_did_not_improve = 0
            train_model.save_weights(save_file_path)
        else:
            count_model_did_not_improve += 1
            print("Early stop count " + str(count_model_did_not_improve) + "/" + str(self.patience))
            if count_model_did_not_improve >= self.patience:
                self.model.stop_training = True
                print("Epoch %05d: early stopping THR" % epoch)
        
        optimizer.reset_old_weights()
        print(
            'best f1: %.5f\n' % (self.best_val_f1)
        )


if __name__ == '__main__':
    
    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator(patience = 3)
    
    for i, item in enumerate(train_generator):
        print("\nbatch_token_ids shape:", item[0][0].shape)
        print("batch_segment_ids shape:", item[0][1].shape)
        print("batch_subject_labels shape:", item[0][2].shape)
        print("batch_subject_ids shape:", item[0][3].shape)
        print("batch_object_labels shape:", item[0][4].shape)
        if i == 4:
            break
    
    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch = len(train_generator),
        epochs = 50,
        callbacks = [evaluator]
    )

else:
    save_file_path = "{}/{}_{}.h5".format(weights_path, event_type, MODEL_TYPE)
    train_model.load_weights(save_file_path)
