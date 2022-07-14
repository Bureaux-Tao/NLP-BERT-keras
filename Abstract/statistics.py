import json

import pandas as pd

source = []
target = []
with open("./data/all.jsonl", encoding = 'utf-8') as f:
    i = 0
    for l in f:
        l = json.loads(l)
        if len(l['target'].strip()) <= 256:
            i += 1
        source.append(len(l['source'].strip()))
        target.append(len(l['target'].strip()))
s = pd.DataFrame(source)
t = pd.DataFrame(target)
print(s.describe())
print(t.describe())
print(i / s.count())
print(s.count() - i)
