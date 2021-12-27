from keras.layers import Dense

from path import *
from utils.layers import ConditionalRandomField, Model
from utils.models import build_transformer_model
from utils.snippets import ViterbiDecoder, to_array
from utils.tokenizers import Tokenizer
from utils.backend import K

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


class WordSegmenter(ViterbiDecoder):
    """基本分词器
    """
    
    def __init__(self, tokenizer, model, trans, starts, ends):
        self.tokenizer = tokenizer
        self.model = model
        super(WordSegmenter, self).__init__(trans = trans, starts = starts, ends = ends)
    
    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = self.tokenizer.rematch(text, tokens)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = self.model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        words = []
        for i, label in enumerate(labels[1:-1]):
            if label < 2 or len(words) == 0:
                words.append([i + 1])
            else:
                words[-1].append(i + 1)
        return [text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1] for w in words]


if __name__ == '__main__':
    tokenizer = Tokenizer(dict_path, do_lower_case = True)
    
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
    save_file_path = "{}/{}_segmentation_tiny.h5".format(weights_path, MODEL_TYPE)
    model.load_weights(save_file_path)
    
    segmenter = WordSegmenter(tokenizer, model, trans = K.eval(CRF.trans), starts = [0], ends = [0])
    
    txt = "张宝华当时是香港有线电视的时政记者，又就中央是否支持董建华连任特首一事向江泽民提问。江泽民面对记者的提问，先是风趣地以粤语“好啊”、“当然啦”回答，但随着记者提问为什么提前钦定董建华为特首，江泽民开始对“钦定”表露出不满情绪。他还称赞曾采访他的记者迈克·华莱士“比你们不知道高到哪里去了”。江泽民指责香港新闻界“你们有一个好，全世界跑到什么地方，你们比其他的西方记者跑得还快，但是问来问去的问题呀，都too simple, sometimes naive”。"
    
    print("/ ".join(segmenter.tokenize(txt)))
