import json

import numpy as np
from tqdm import tqdm

from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, train_file_path_abstract, val_file_path_abstract, \
    test_file_path_abstract, weights_path, event_type, MODEL_TYPE
from utils.backend import keras, K
from utils.layers import Loss
from utils.models import build_transformer_model
from utils.tokenizers import Tokenizer, load_vocab
from utils.optimizers import Adam, extend_with_exponential_moving_average
from utils.snippets import sequence_padding, open
from utils.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 基本参数
maxlen = 256
batch_size = 64
epochs = 200

config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)


def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding = 'utf-8') as f:
        for l in f:
            d = json.loads(l)
            title = d["title"]
            content = d["abst"]
            D.append((title, content))
    return D


# 加载数据集
train_data = load_data(train_file_path_abstract)
valid_data = load_data(val_file_path_abstract)
test_data = load_data(test_file_path_abstract)

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
        def get_keys(d, value):
            return [k for k, v in d.items() if v == value]
        
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (title, content) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                content, title, maxlen = maxlen
            )
            ids = []
            for i, item in enumerate(token_ids):
                ids.append(get_keys(token_dict, item)[0])
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
    model = MODEL_TYPE,
    application = 'unilm',
    keep_tokens = keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    
    @AutoRegressiveDecoder.wraps(default_rtype = 'probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(model).predict([token_ids, segment_ids])
    
    def generate(self, text, topk = 1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen = max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk = topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id = None, end_id = tokenizer._token_end_id, maxlen = 32)
count_model_did_not_improve = 0


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    
    def __init__(self, patience = 5):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.
        self.patience = patience
    
    def on_epoch_end(self, epoch, logs = None):
        global count_model_did_not_improve
        # 保存最优
        save_file_path = "{}/{}_{}.h5".format(weights_path, event_type, MODEL_TYPE)
        metrics = self.evaluate(valid_data)  # 评测模型
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            model.save_weights(save_file_path)  # 保存模型
            count_model_did_not_improve = 0
        else:
            count_model_did_not_improve += 1
            print("Early stop count " + str(count_model_did_not_improve) + "/" + str(self.patience))
            if count_model_did_not_improve >= self.patience:
                self.model.stop_training = True
                print("Epoch %05d: early stopping THR" % epoch)
        metrics['best_bleu'] = self.best_bleu
        print('valid_data:', metrics)
    
    def evaluate(self, data, topk = 1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        pbar = tqdm()
        for title, content in data:
            total += 1
            title = ' '.join(title).lower()
            pred_title = ' '.join(autotitle.generate(content, topk)).lower()
            if pred_title.strip():
                scores = self.rouge.get_scores(hyps = pred_title, refs = title)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references = [title.split(' ')],
                    hypothesis = pred_title.split(' '),
                    smoothing_function = self.smooth
                )
                pbar.update()
                pbar.set_description(
                    'rouge_1: %.5f, rouge_2: %.5f, rouge_l: %.5f, bleu: %.5f' % (
                        rouge_1 / total, rouge_2 / total, rouge_l / total, bleu / total)
                )
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        pbar.close()
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }


if __name__ == '__main__':
    evaluator = Evaluator(patience = 5)
    train_generator = data_generator(train_data, batch_size)
    
    model.fit(
        train_generator.forfit(),
        steps_per_epoch = len(train_generator),
        epochs = epochs,
        callbacks = [evaluator]
    )
