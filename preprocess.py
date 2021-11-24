import numpy as np
import re
import pandas as pd
from finetune_args import args


# 截断至最大长度
def truncate_sequences(maxlen, index, *sequences):
    sequences = [s for s in sequences if s]
    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(index)
        else:
            return sequences


def load_data(filename):
    """
    加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    data = open(filename, encoding='UTF-8')
    D = []
    for text in data.readlines():
        text_list = text.strip().split('\t')
        text1 = text_list[0]
        text2 = text_list[1]
        if len(text_list) > 2:
            label = int(text_list[2])
        else:
            label = -100
        D.append((text1, text2, label))
    return D


# 统计序列长度大于某个值的数量
def get_len(data):
    lens = []
    for line in data:
        lens.append(len(line[0] + line[1]))
    count = 0
    for i in lens:
        if i > 64:
            count += 1
    print(count)


def data_clean(x):
    # HTML标签
    x = re.sub(r'<[^>]+>', '', x)
    # 多余空格
    x = re.sub(r'\s', '', x)
    #

    # 表情
    # expression = ['/撇嘴', '/亲亲', '/发怒', '/:|-)', '/坏笑', '/睡', '/微笑', '/难过',
    #               '/可怜', '/折磨', '/得意', '/呲牙', '/调皮', '/流泪', '/发呆', '/抓狂',
    #               '/可爱', '/色', '/调皮', '/咒骂', '/大哭', '/傲慢', '/偷笑', '/:gift',
    #               '/尴尬', '/快哭了', '/表情', '/想', '/流汗', '/拥抱', '/咒骂', '/流泪',
    #               '-_-', '(⊙o⊙)']
    # for wbad in expression:
    #     x = x.replace(wbad, '')
    # 数字加标点符号开头的噪音
    pattern_1 = '^[0-9][.？，:！〈?|~】、～。^＂=<+：－%＞/"（(【).＝_#;…＃〉《”÷!,>＋\]；@）￥\[“]'
    x = re.sub(pattern_1, '', x)
    # 标点符号开头的噪音
    pattern_2 = '^[.？，！:〈?|~】、～。^＂=<+：－%＞/"（(【).＝_#;…＃〉《”÷!,>＋\]；@）￥\[“]'
    x = re.sub(pattern_2, '', x)

    return x


if __name__ == '__main__':
    fr = open(f'./data/train_dataset/STS-B/valid.data',encoding='UTF-8')
    fw = open('./data/train_dataset/STS-B/valid', 'w', encoding='UTF-8')
    for line in fr.readlines():
        text_list = line.strip().split('\t')
        text1 = text_list[0]
        text2 = text_list[1]
        label = str(0) if int(text_list[2]) < 4 else str(1)
        fw.write(text1 + '\t' + data_clean(text2) + '\t' + label + '\n')
    fr.close()
    fw.close()
