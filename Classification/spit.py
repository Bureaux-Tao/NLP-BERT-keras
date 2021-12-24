with open('./data/train_sentiment.txt', encoding = 'utf-8') as f:
    with open('./data/val_sentiment.txt', 'a', encoding = 'utf-8') as f1:
        lines = f.readlines()
        for i, item in enumerate(lines):
            if i > 13000:
                print(i, item)
                f1.writelines(item)
