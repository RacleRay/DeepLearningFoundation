#-*- coding:utf-8 -*-
# author: Racle
# project: pointer-network


import glob
import re
from tqdm import tqdm

from config import train_data_path


pattern1 = re.compile(r'{{(.*)}}$')
def find_content(text):
    return re.search(pattern1, text).group(1)


pattern2 = re.compile(r'!\[.*?\]\(.*?\)')
def remove_pic_ref(text):
    return re.sub(pattern2, '', text)


pattern3 = re.compile(r'\(http.*?\)')
def remove_http_ref(text):
    return re.sub(pattern3, '', text)


# 小实验结果：分词的效果比不上分字的效果。
# tokenizer = pyhanlp.JClass("com.hankcs.hanlp.tokenizer.StandardTokenizer")

def gen_vocab_file():
    word_freq = {}
    vocab_file = r'./vocab/vocab_file'
    with open(vocab_file, 'w', encoding='utf-8') as t:
        filelist = glob.glob(train_data_path + '/*.txt')
        for f in tqdm(filelist):
            with open(f, 'r', encoding='utf-8') as f:
                text = f.readlines()
                summary = find_content(text[0])
                content = remove_http_ref(remove_pic_ref(find_content(text[1])))
            for w in list(summary + content):
                word_freq[w.strip()] = word_freq.setdefault(w, 0) + 1
        word_freq = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
        for w, num in tqdm(word_freq):
            t.write(w + '\t' + str(num) + '\n')


if __name__ == "__main__":
    gen_vocab_file()