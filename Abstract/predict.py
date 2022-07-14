import numpy as np

from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, MODEL_TYPE
from utils.backend import K
from utils.layers import Loss
from utils.models import build_transformer_model
from utils.tokenizers import Tokenizer, load_vocab
from utils.snippets import AutoRegressiveDecoder
from keras.models import Model

config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)

maxlen = 256

token_dict, keep_tokens = load_vocab(
    dict_path = dict_path,
    simplified = True,
    startswith = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case = True)


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

model.load_weights("../weights/Abstract_roformer_v2.h5")


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


def just_show(s):
    print(u'生成标题:', autotitle.generate(s))


just_show(u'根据动态组播代理的移动组播算法原理,提出一种移动组播协议,采用新的快速组播切换机制,以时间量度和距离量度为依据,动态选择组播代理。仿真结果表明,该协议避免组播转发树的频繁重构,降低组播切换延迟,具有次优的组播传输路径,可以与现有网络协议相融合。')
just_show(u'为方便医生做肺结节病历的研究，本文提出了基于嵌入Roformer预训练语言模型和一种基于全局指针思想的实体关系抽取模型，采用滑动平均的优化方法并启用快速梯度方法进行对抗训练，针对中文肺部CT病历数据集进行识别。本文所提出的模型并可以根据上下文语义，分析出父子关系，进而将其处理成结构化数据。实验结果表明，本模型相较传统方法抽取效果提升显著，F1值可达86.2%。')
just_show(u'Web服务具有平台无关性、动态性、开放性和松散耦合等特征,这给基于异构平台的应用集成带来极大便利,同时也使其自身面临许多独特的安全问题。Web服务的安全性对其发展前景产生重要的影响,也是目前Web服务并没有进入大规模应用阶段的主要原因之一。总结了Web服务存在的主要安全问题;概述了已有的Web服务安全标准;然后从消息层安全、Web服务安全策略、Web服务组合安全、身份与信任管理、Web服务访问控制、Web服务攻击与防御、安全Web服务开发等方面详细分析了目前有代表性的Web服务关键安全技术解决方案;结合已有的研究成果,讨论了Web服务安全未来的研究动向及面临的挑战。')

# 生成标题: 移动组播代理的移态组播协议
# 生成标题: 基于嵌入roformer预训练语言模型的肺结节病历抽取
# 生成标题: 基于异构平台的web服务安全研究