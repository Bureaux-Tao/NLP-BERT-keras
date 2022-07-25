import json
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.backend import keras, K
from utils.layers import Loss
from utils.models import build_transformer_model
from utils.tokenizers import Tokenizer, load_vocab
from utils.optimizers import Adam
from utils.optimizers import extend_with_weight_decay
from utils.optimizers import extend_with_gradient_accumulation
from utils.snippets import sequence_padding, open
from utils.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, train_file_path_multidialog, weights_path, \
    val_file_path_multidialog
import jieba

jieba.initialize()

maxlen = 512
batch_size = 16
steps_per_epoch = 1000
epochs = 9999

# nezha配置
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)


def corpus(path):
    """循环读取语料
    """
    while True:
        with open(path, 'r', encoding = 'utf-8') as f:
            for l in f:
                l = json.loads(l, encoding = 'utf-8')
                yield l


# 加载并精简词表
token_dict, keep_tokens = load_vocab(
    dict_path = dict_path,
    simplified = True,
    startswith = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
# 补充词表
compound_tokens = []
for l in open('./user_tokens.csv', encoding = 'utf-8'):
    token, count = l.strip().split('\t')
    if int(count) >= 10 and token not in token_dict:
        token_dict[token] = len(token_dict)
        compound_tokens.append([0])

# 建立分词器
tokenizer = Tokenizer(token_dict, do_lower_case = True, pre_tokenize = lambda s: jieba.cut(s, HMM = False))


# tokenizer = Tokenizer(dict_path, do_lower_case = True, pre_tokenize = lambda s: jieba.cut(s, HMM = False))


class data_generator(DataGenerator):
    """数据生成器
    """
    
    def __iter__(self, random = False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, texts in self.sample(random):
            token_ids, segment_ids = [tokenizer._token_start_id], [0]
            for i, text in enumerate(texts):
                ids = tokenizer.encode(text)[0][1:]
                if len(token_ids) + len(ids) <= maxlen:
                    token_ids.extend(ids)
                    segment_ids.extend([i % 2] * len(ids))
                else:
                    break
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉padding部分
    """
    
    def compute_loss(self, inputs, mask = None):
        y_true, y_pred = inputs
        y_mask = K.cast(mask[1], K.floatx())[:, 1:]
        y_true = y_true[:, 1:]  # 目标token_ids
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    model = 'nezha',
    application = 'lm',
    keep_tokens = keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    compound_tokens = compound_tokens,  # 要扩充的词表
)

output = CrossEntropy(1)([model.inputs[0], model.outputs[0]])

model = Model(model.inputs, output)
model.summary()

for layer in model.layers:
    layer.trainable = True

AdamW = extend_with_weight_decay(Adam, 'AdamW')
AdamWG = extend_with_gradient_accumulation(AdamW, 'AdamWG')
optimizer = AdamWG(
    lr = 2e-5,
    weight_decay_rate = 0.01,
    exclude_from_weight_decay = ['Norm', 'bias'],
    grad_accum_steps = 16
)
model.compile(optimizer = optimizer)


class ChatBot(AutoRegressiveDecoder):
    """基于随机采样对话机器人
    """
    
    @AutoRegressiveDecoder.wraps(default_rtype = 'probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        curr_segment_ids = np.ones_like(output_ids) - segment_ids[0, -1]
        segment_ids = np.concatenate([segment_ids, curr_segment_ids], 1)
        return model.predict([token_ids, segment_ids])[:, -1]
    
    def response(self, texts, topk = 5):
        token_ids, segment_ids = [tokenizer._token_start_id], [0]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:]
            token_ids.extend(ids)
            segment_ids.extend([i % 2] * len(ids))
        results = self.random_sample([token_ids, segment_ids], 1, topk)
        return tokenizer.decode(results[0])


chatbot = ChatBot(start_id = None, end_id = tokenizer._token_end_id, maxlen = 32)


class TestOutput(keras.callbacks.Callback):
    """保存模型权重
    """
    
    def __init__(self, dialog_list):
        self.dialog = dialog_list
    
    def on_epoch_end(self, epoch, logs = None):
        print('dialog:', self.dialog)
        print('answer:', chatbot.response(self.dialog).replace(" ", "") + "\n")


if __name__ == '__main__':
    test_output = TestOutput([u'我觉得巧克力好苦', u'那是你没吃到甜的', u'我不喜欢吃甜的'])
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1, mode = 'min')  # 提前结束
    save_model = ModelCheckpoint(weights_path + "/multi_dialog.h5", monitor = 'val_loss', verbose = 0,
                                 mode = 'min', save_weights_only = True, save_best_only = True)
    train_data = corpus(train_file_path_multidialog)
    train_generator = data_generator(train_data, batch_size)
    
    val_data = corpus(val_file_path_multidialog)
    val_generator = data_generator(val_data, batch_size)
    
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch = steps_per_epoch,
        validation_data = val_generator.forfit(),
        validation_steps = steps_per_epoch,
        epochs = epochs,
        callbacks = [test_output, early_stopping, save_model],
        verbose = 1
    )
