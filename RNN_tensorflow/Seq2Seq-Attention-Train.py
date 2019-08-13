import tensorflow as tf


SRC_TRAIN_DATA = "./train.en"          # 源语言输入文件。
TRG_TRAIN_DATA = "./train.zh"          # 目标语言输入文件。
CHECKPOINT_PATH = "./attention_ckpt"   # checkpoint保存路径。

HIDDEN_SIZE = 1024                     # LSTM的隐藏层规模。
DECODER_LAYERS = 2                     # 解码器中LSTM结构的层数。这个例子中编码器固定使用单层的双向LSTM。
SRC_VOCAB_SIZE = 10000                 # 源语言词汇表大小。
TRG_VOCAB_SIZE = 4000                  # 目标语言词汇表大小。
BATCH_SIZE = 100                       # 训练数据batch的大小。
NUM_EPOCH = 5                          # 使用训练数据的轮数。
KEEP_PROB = 0.8                        # 节点不被dropout的概率。
MAX_GRAD_NORM = 5                      # 用于控制梯度膨胀的梯度大小上限。
SHARE_EMB_AND_SOFTMAX = True           # 在Softmax层和词向量层之间共享参数。

MAX_LEN = 50   # 限定句子的最大单词数量。
SOS_ID  = 1    # 目标语言词汇表中<sos>的ID。


# 读取训练数据并创建Dataset
def MakeDataset(file_path):
    """数据的格式为每行一句话，单词已经转化为单词编号"""
    dataset =tf.data.TextLineDataset(file_path)
    # 根据空格将单词编号切分开并放入一个一维向量。
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # 将字符串形式的单词编号转化为整数。
    dataset = dataset.map(
        lambda string: tf.string_to_number(string, tf.int32))
    # 统计每个句子的单词数量，并与句子内容一起放入Dataset中。
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    """从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行填充和
    batching操作"""
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    # ds[0][0]是源句子
    # ds[0][1]是源句子长度
    # ds[1][0]是目标句子
    # ds[1][1]是目标句子长度
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    def FilterLength(src_tuple, trg_tuple):
        """删除内容为空（只包含<EOS>）的句子和长度过长的句子"""
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(
            tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(
            tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok)

    dataset = dataset.filter(FilterLength)

    def MakeTrgInput(src_tuple, trg_tuple):
        """
        # 1.解码器的输入(trg_input)，形式如同"<sos> X Y Z"
        # 2.解码器的目标输出(trg_label)，形式如同"X Y Z <eos>"
        从文件中读到的目标句子是"X Y Z <eos>"的形式，我们需要从中生成"<sos> X Y Z"
        形式并加入到Dataset中。
        """
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))

    dataset = dataset.map(MakeTrgInput)

    dataset = dataset.shuffle(10000)

    padded_shapes = (
        (tf.TensorShape([None]),      # 源句子是长度未知的向量
         tf.TensorShape([])),         # 源句子长度是单个数字
        (tf.TensorShape([None]),      # 目标句子（解码器输入）是长度未知的向量
         tf.TensorShape([None]),      # 目标句子（解码器目标输出）是长度未知的向量
         tf.TensorShape([])))         # 目标句子长度是单个数字
    # 调用padded_batch方法进行batching操作。
    # padded to the maximum size of that dimension in each batch
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes, )
    return batched_dataset


# 定义翻译模型

class NMTModel(object):
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构。
        self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
          [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
           for _ in range(DECODER_LAYERS)])

        # 为源语言和目标语言分别定义词向量。
        self.src_embedding = tf.get_variable(
            "src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable(
            "trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
           self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
           self.softmax_weight = tf.get_variable(
               "weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable(
            "softmax_bias", [TRG_VOCAB_SIZE])

    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]

        # 将输入和输出单词编号转为词向量。
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        # 在词向量上进行dropout。
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        with tf.variable_scope("encoder"):
            # enc_outputs: 顶层LSTM在每一步的输出
            #            ([batch_size, max_time, HIDDEN_SIZE], [batch_size, max_time, HIDDEN_SIZE])
            # enc_state: ([batch_size, HIDDEN_SIZE], [batch_size, HIDDEN_SIZE])
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.enc_cell_fw, self.enc_cell_bw, src_emb, src_size,
                dtype=tf.float32)
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)

        with tf.variable_scope("decoder"):
            # BahdanauAttention是使用一个隐藏层的前馈神经网络
            # memory_sequence_length是一个维度为[batch_size]的张量，代表batch
            # 中每个句子的长度，Attention需要根据这个信息把填充位置的注意力权重设置为0
            attention_mechanisim = tf.contrib.seq2seq.AttentionWrapper(
                self.dec_cell, attention_mechanism,
                attention_layer_size=HIDDEN_SIZE
            )

            # 不使用编码器的输出来初始化输入，而完全依赖注意力作为信息来源
            dec_outputs, _ =  tf.nn.dynamic_rnn(
                attention_cell, trg_emb, trg_size, dtype=tf.float32)

        # 计算解码器每一步的log perplexity
        output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(trg_label, [-1]), logits=logits)

        # 在计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰
        # 模型的训练
        label_weights = tf.sequence_mask(
            trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(cost / tf.to_float(batch_size),
                             trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        return cost_per_token, train_op


# 训练过程和主函数
def run_epoch(session, cost_op, train_op, saver, step):
    # 训练一个epoch。
    # 重复训练步骤直至遍历完Dataset中所有数据。
    while True:
        try:
            # 运行train_op并计算损失值。训练数据在main()函数中以Dataset方式提供。
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print("After %d steps, per token cost is %.3f" % (step, cost))
            # 每200步保存一个checkpoint。
            if step % 200 == 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step

def main():
    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("nmt_model", reuse=None,
                           initializer=initializer):
        train_model = NMTModel()

    # 定义输入数据。
    data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    # 定义前向计算图。输入数据以张量形式提供给forward函数。
    cost_op, train_op = train_model.forward(src, src_size, trg_input,
                                            trg_label, trg_size)

    # 训练模型。
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)



if __name__ == "__main__":
    main()