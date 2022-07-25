##
import json
import os

from tqdm import tqdm

with open("./data/LCCC-base_train.json", 'r', encoding = 'utf-8') as f:
    m = json.load(f)
    print(len(m))
m
##
if os.path.exists("./data/train.jsonl"):
    # 存在，则删除文件
    os.remove("./data/train.jsonl")
if os.path.exists("./data/valid.jsonl"):
    # 存在，则删除文件
    os.remove("./data/valid.jsonl")
with open("./data/train.jsonl", 'a', encoding = 'utf-8') as f1:
    with open("./data/valid.jsonl", 'a', encoding = 'utf-8') as f2:
        for i in tqdm(range(len(m))):
            if i < int(len(m) * 0.8):
                f1.write(json.dumps(m[i], ensure_ascii = False) + "\n")
            else:
                f2.write(json.dumps(m[i], ensure_ascii = False) + "\n")