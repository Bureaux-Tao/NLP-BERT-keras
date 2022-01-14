import codecs

from utils.snippets import lowercase_and_normalize
from utils.tokenizers import Tokenizer

dict_path = './albert_base_google_zh/vocab.txt'
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        if self._do_lower_case:
            text = lowercase_and_normalize(text)
        
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += '\t' + ch + '\t'
            elif self._is_space(ch):
                spaced += '\t' + '[unused1]\t'
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        
        tokens = []
        for word in spaced.strip().split('\t'):
            tokens.extend(self._word_piece_tokenize(word))
        
        return tokens


tokenizer1 = OurTokenizer(token_dict)
txt = "我的iphone6s丢   了"
tokenizer2 = Tokenizer(dict_path, do_lower_case = True)
print(tokenizer1.tokenize(txt))
token_ids, segment_ids = tokenizer1.encode(txt, maxlen = 30)
print(token_ids)
print(segment_ids)

# print(tokenizer.tokenize(u'今天天气不错 '))


# ['[CLS]', '今', '天', '天', '气', '不', '错', '[unused1]', '[SEP]']

# def get_keys(d, value):
#     return [k for k, v in d.items() if v == value]
#
#
# ids = [101, 2225, 1046, 185, 4649, 858, 2861, 2805, 8020, 10560, 11057, 8169, 9409, 12653, 11403, 8021, 5401, 1744,
#        4510, 2512, 4028, 1447, 8024, 8974, 2399, 1139, 4495, 754, 5401, 1744, 6662, 3211, 3172, 2128, 6929, 2336, 3173,
#        1952, 2209, 5679, 8024, 3684, 689, 754, 7350, 5507, 5682, 1920, 2110, 1469, 6662, 3211, 3172, 2128, 6929, 2336,
#        4989, 1920, 2110, 102]
#
# for i, item in enumerate(ids):
#     print((i, get_keys(token_dict, item)[0]),end = "")
