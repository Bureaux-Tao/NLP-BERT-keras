from collections import Counter

with open('/home/bureaux/Projects/keras4bert/data/yidu_train.txt', encoding = 'utf-8') as f1:
    with open('/home/bureaux/Projects/keras4bert/data/yidu.train', 'a', encoding = 'utf-8') as f2:
        with open('/home/bureaux/Projects/keras4bert/data/yidu.validate', 'a', encoding = 'utf-8') as f3:
            lines = f1.readlines()
            count = 0
            sentense_len = []
            for i in lines:
                count += 1
                if i.strip('\n') == '。	O' or i.strip('\n') == '；	O' or i.strip('\n') == '？	O' or i.strip(
                        '\n') == '！	O':
                    sentense_len.append(count)
                    count = 0
            print(max(sentense_len))
            print(len(sentense_len))
            freq = dict(Counter(sentense_len))
            print(sorted(freq.items(), key = lambda d: d[0], reverse = True))
            print(sorted(freq.items(), key = lambda d: d[1], reverse = True))
            
            count = 0
            for i in lines:
                if count <= 7600:
                    item = i.strip('\n')
                    char, tag = item.split('\t')
                    if item.strip('\n') == '。	O' or item.strip('\n') == '；	O' or item.strip(
                            '\n') == '？	O' or item.strip('\n') == '！	O':
                        count += 1
                        f2.write(char + ' ' + tag + '\n\n')
                    else:
                        if len(tag.split('-')) == 1:
                            f2.write(char + ' ' + tag + '\n')
                        else:
                            tag_name, tag_bio = tag.split('-')
                            f2.write(char + ' ' + tag_bio + '-' + tag_name + '\n')
                else:
                    item = i.strip('\n')
                    char, tag = item.split('\t')
                    if item.strip('\n') == '。	O' or item.strip('\n') == '；	O' or item.strip(
                            '\n') == '？	O' or item.strip('\n') == '！	O':
                        count += 1
                        f3.write(char + ' ' + tag + '\n\n')
                    else:
                        if len(tag.split('-')) == 1:
                            f3.write(char + ' ' + tag + '\n')
                        else:
                            tag_name, tag_bio = tag.split('-')
                            f3.write(char + ' ' + tag_bio + '-' + tag_name + '\n')
            print('finish')

with open('/home/bureaux/Projects/keras4bert/data/yidu_test.txt', encoding = 'utf-8') as f4:
    with open('/home/bureaux/Projects/keras4bert/data/yidu.test', 'a', encoding = 'utf-8') as f5:
        lines = f4.readlines()
        for i in lines:
            item = i.strip('\n')
            char, tag = item.split('\t')
            if item.strip('\n') == '。	O' or item.strip('\n') == '；	O' or item.strip(
                    '\n') == '？	O' or item.strip('\n') == '！	O':
                f5.write(char + ' ' + tag + '\n\n')
            else:
                if len(tag.split('-')) == 1:
                    f5.write(char + ' ' + tag + '\n')
                else:
                    tag_name, tag_bio = tag.split('-')
                    f5.write(char + ' ' + tag_bio + '-' + tag_name + '\n')
