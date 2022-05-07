import numpy as np

from QA.train import tokenizer, max_q_len, max_p_len, model
from utils.snippets import AutoRegressiveDecoder

model.load_weights("/home/bureaux/Projects/keras4bert/weights/QA_albert.h5")


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

result = qag.generate("1915年12月13日，时任中华民国大总统的袁世凯宣布废除共和政体，实行帝制，改国号为中华帝国，年号洪宪，自任皇帝；1916年3月22日，宣布废除帝制，重归共和，前后总共历时83天。")

print(result)
