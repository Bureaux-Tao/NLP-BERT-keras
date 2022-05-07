#! -*- coding: utf-8- -*-
# 用Seq2Seq做阅读理解构建
# 根据篇章先采样生成答案，然后采样生成问题
# 数据集同 https://github.com/bojone/dgcnn_for_reading_comprehension

import json, os
import numpy as np
from keras.callbacks import ModelCheckpoint

from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, train_file_path_web, train_file_path_sogou, \
    val_file_path_QA, MODEL_TYPE, weights_path, event_type
from utils.backend import keras, K
from utils.layers import Loss
from utils.models import build_transformer_model
from utils.tokenizers import Tokenizer, load_vocab
from utils.optimizers import Adam
from utils.snippets import sequence_padding, open
from utils.snippets import DataGenerator, AutoRegressiveDecoder
from utils.snippets import text_segmentate
from keras.models import Model
from tqdm import tqdm

# 基本参数
max_p_len = 128
max_q_len = 64
max_a_len = 16
batch_size = 256
epochs = 100

# bert配置
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)

# 标注数据
webqa_data = json.load(open(train_file_path_web, encoding = "utf-8"))
sogou_data = json.load(open(train_file_path_sogou, encoding = "utf-8"))

# 筛选数据
seps, strips = u'\n。！？!?；;，, ', u'；;，, '
data = []
for d in webqa_data + sogou_data:
    for p in d['passages']:
        if p['answer']:
            for t in text_segmentate(p['passage'], max_p_len - 2, seps, strips):
                if p['answer'] in t:
                    data.append((t, d['question'], p['answer']))

del webqa_data
del sogou_data

with open("./data/v.txt","a",encoding = "utf-8") as f1:
    for i in data:
        f1.write(i[0]+'\n')
        f1.write(i[1]+'\n')
        f1.write(i[2]+'\n\n')
    

# 保存一个随机序（供划分valid用）
if not os.path.exists(val_file_path_QA):
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    json.dump(random_order, open(val_file_path_QA, 'w', encoding = "utf-8"), indent = 4)
else:
    random_order = json.load(open(val_file_path_QA, encoding = "utf-8"))

# 划分valid
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path = dict_path,
    simplified = True,
    startswith = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case = True)


class data_generator(DataGenerator):
    """数据生成器
    """
    
    def __iter__(self, random = False):
        """单条样本格式：[CLS]篇章[SEP]答案[SEP]问题[SEP]
        """
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (p, q, a) in self.sample(random):
            p_token_ids, _ = tokenizer.encode(p, maxlen = max_p_len + 1)
            a_token_ids, _ = tokenizer.encode(a, maxlen = max_a_len)
            q_token_ids, _ = tokenizer.encode(q, maxlen = max_q_len)
            token_ids = p_token_ids + a_token_ids[1:] + q_token_ids[1:]
            segment_ids = [0] * len(p_token_ids)
            segment_ids += [1] * (len(token_ids) - len(p_token_ids))
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    
    def compute_loss(self, inputs, mask = None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application = 'unilm',
    keep_tokens = keep_tokens,  # 只保留keep_tokens中的字，精简原字表,
    model = MODEL_TYPE
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer = Adam(1e-4))
model.summary()


class QuestionAnswerGeneration(AutoRegressiveDecoder):
    """随机生成答案，并且通过beam search来生成问题
    """
    
    @AutoRegressiveDecoder.wraps(default_rtype = 'probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(model).predict([token_ids, segment_ids])
    
    def generate(self, passage, topk = 1, topp = 0.95):
        token_ids, segment_ids = tokenizer.encode(passage, maxlen = max_p_len)
        a_ids = self.random_sample([token_ids, segment_ids], 1,
                                   topp = topp)[0]  # 基于随机采样
        token_ids += list(a_ids)
        segment_ids += [1] * len(a_ids)
        q_ids = self.beam_search([token_ids, segment_ids],
                                 topk = topk)  # 基于beam search
        return (tokenizer.decode(q_ids), tokenizer.decode(a_ids))


qag = QuestionAnswerGeneration(
    start_id = None, end_id = tokenizer._token_end_id, maxlen = max_q_len
)


def predict_to_file(data, filename, topk = 1):
    """将预测结果输出到文件，方便评估
    """
    with open(filename, 'w', encoding = 'utf-8') as f:
        for d in tqdm(iter(data), desc = u'正在预测(共%s条样本)' % len(data)):
            q, a = qag.generate(d[0])
            s = '%s\t%s\t%s\n' % (q, a, d[0])
            f.write(s)
            f.flush()


count_model_did_not_improve = 0


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    
    def __init__(self, patience = 5):
        super().__init__()
        self.lowest = 1e10
        self.patience = patience
    
    def on_epoch_end(self, epoch, logs = None):
        global count_model_did_not_improve
        # 保存最优
        save_file_path = "{}/{}_{}.h5".format(weights_path, event_type, MODEL_TYPE)
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            
            # predict_to_file(valid_data, "./data/pred/ep_" + str(epoch + 1) + ".csv")
            
            model.save_weights(save_file_path)
            count_model_did_not_improve = 0
        else:
            count_model_did_not_improve += 1
            print("Early stop count " + str(count_model_did_not_improve) + "/" + str(self.patience))
            if count_model_did_not_improve >= self.patience:
                self.model.stop_training = True
                print("Epoch %05d: early stopping THR" % epoch)


if __name__ == '__main__':
    save_all_path = ("{}/QA/{}_{}_tiny".format(weights_path, event_type, MODEL_TYPE)) + "_ep{epoch:02d}.h5"
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)
    save_model = ModelCheckpoint(save_all_path, monitor = 'loss', verbose = 0, period = 1,
                                 save_weights_only = True, save_best_only = False)
    
    model.fit(
        train_generator.forfit(),
        steps_per_epoch = 1000,
        epochs = epochs,
        callbacks = [evaluator, save_model]
    )
