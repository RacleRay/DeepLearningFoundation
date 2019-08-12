import codecs
import collections
from operator import itemgetter


# 生成词汇表

# 参数
MODE = "PTB"    # 将MODE设置为"PTB", "TRANSLATE_EN", "TRANSLATE_ZH"之一。

if MODE == "PTB":             # PTB数据处理
    RAW_DATA = "../datasets/PTB_data/ptb.train.txt"  # 训练集数据文件
    VOCAB_OUTPUT = "ptb.vocab"                         # 输出的词汇表文件
elif MODE == "TRANSLATE_ZH":  # 翻译语料的中文部分
    RAW_DATA = "../datasets/TED_data/train.txt.zh"
    VOCAB_OUTPUT = "zh.vocab"
    VOCAB_SIZE = 4000
elif MODE == "TRANSLATE_EN":  # 翻译语料的英文部分
    RAW_DATA = "../datasets/TED_data/train.txt.en"
    VOCAB_OUTPUT = "en.vocab"
    VOCAB_SIZE = 10000


# 对单词按词频排序
counter = collections.Counter()
with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

sorted_word_to_cnt = sorted(
    counter.items(), key=itemgetter(1), reverse=True
)


# 插入特殊符号
if MODE == "PTB":
    # 我们需要在文本换行处加入句子结束符"<eos>"，这里预先将其加入词汇表。
    sorted_words = ["<eos>"] + sorted_words
elif MODE in ["TRANSLATE_EN", "TRANSLATE_ZH"]:
    # 处理机器翻译数据时，除了"<eos>"以外，还需要将"<unk>"和句子起始符
    # "<sos>"加入词汇表，并从词汇表中删除低频词汇。
    sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
    if len(sorted_words) > VOCAB_SIZE:
        sorted_words = sorted_words[:VOCAB_SIZE]


# 保存词汇表文件
with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as fout:
    for word in sorted_words:
        fout.write(word + "\n")



# 生成训练文件

# 参数设置
MODE = "PTB_TRAIN"    # 将MODE设置为"PTB_TRAIN", "PTB_VALID", "PTB_TEST", "TRANSLATE_EN", "TRANSLATE_ZH"之一。

if MODE == "PTB_TRAIN":        # PTB训练数据
    RAW_DATA = "../datasets/PTB_data/ptb.train.txt"  # 训练集数据文件
    VOCAB = "ptb.vocab"                                 # 词汇表文件
    OUTPUT_DATA = "ptb.train"                           # 将单词替换为单词编号后的输出文件
elif MODE == "PTB_VALID":      # PTB验证数据
    RAW_DATA = "../datasets/PTB_data/ptb.valid.txt"
    VOCAB = "ptb.vocab"
    OUTPUT_DATA = "ptb.valid"
elif MODE == "PTB_TEST":       # PTB测试数据
    RAW_DATA = "../datasets/PTB_data/ptb.test.txt"
    VOCAB = "ptb.vocab"
    OUTPUT_DATA = "ptb.test"
elif MODE == "TRANSLATE_ZH":   # 中文翻译数据
    RAW_DATA = "../datasets/TED_data/train.txt.zh"
    VOCAB = "zh.vocab"
    OUTPUT_DATA = "train.zh"
elif MODE == "TRANSLATE_EN":   # 英文翻译数据
    RAW_DATA = "../datasets/TED_data/train.txt.en"
    VOCAB = "en.vocab"
    OUTPUT_DATA = "train.en"


# 按词汇表对将单词映射到编号
with codecs.open(VOCAB, 'r', 'uft-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

# 如果出现了不在词汇表内的低频词，则替换为"unk"
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]


# 对数据进行替换并保存结果
fin = codecs.open(RAW_DATA, "r", "utf-8")
fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')

for line in fin:
    words = line.strip().split() + ["<eos>"]  # 读取单词并添加<eos>结束符
    # 将每个单词替换为词汇表中的编号
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    fout.write(out_line)

fin.close()
fout.close()