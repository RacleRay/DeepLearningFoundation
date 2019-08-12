import numpy as np
import tensorflow as tf


TRAIN_DATA = "ptb.train"          # 训练数据路径。
EVAL_DATA = "ptb.valid"           # 验证数据路径。
TEST_DATA = "ptb.test"            # 测试数据路径。
HIDDEN_SIZE = 300                 # 隐藏层规模。
NUM_LAYERS = 2                    # 深层循环神经网络中LSTM结构的层数。
VOCAB_SIZE = 10000                # 词典规模。
TRAIN_BATCH_SIZE = 20             # 训练数据batch的大小。
TRAIN_NUM_STEP = 35               # 训练数据截断长度。

EVAL_BATCH_SIZE = 1               # 测试数据batch的大小。
EVAL_NUM_STEP = 1                 # 测试数据截断长度。
NUM_EPOCH = 5                     # 使用训练数据的轮数。
LSTM_KEEP_PROB = 0.9              # LSTM节点不被dropout的概率。
EMBEDDING_KEEP_PROB = 0.9         # 词向量不被dropout的概率。
MAX_GRAD_NORM = 5                 # 用于控制梯度膨胀的梯度大小上限。
SHARE_EMB_AND_SOFTMAX = True      # 在Softmax层和词向量层之间共享参数。


# 定义模型
class PTBModel():
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 网络结构
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 0
        lstm_cell = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
                output_keep_prob=dropout_keep_prob
            )
            for _ in range(NUM_LAYERS)
        ]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)

        # 初始化最初的状态，即全零的向量。这个量只在每个epoch初始化第一个batch
        # 时使用。
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        embedding = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE])

        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)

        # 定义输出列表。在这里先将不同时刻LSTM结构的输出收集起来，再一起提供给
        # softmax层。
        outputs = []
        state = self.initial_state
        with tf.vraiable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        # 将输出队列改成[batch*numsteps, hidden_size]的形状
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        # softmax: 共享转换矩阵
        if SHARE_EMB_AND_SOFTMAX:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable('weight', [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable('weight', [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias

        # 定义交叉熵损失函数和平均损失
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]),
            logits=logits
        )
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if not is_training: return

        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, trainable_variables),
            MAX_GRAD_NORM
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables)
        )


# 定义数据和训练过程

def run_epoch(session, model, batches, train_op, output_log, step):
    """使用给定的模型model在数据data上运行train_op并返回在全部数据上的perplexity值"""
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    # 训练一个epoch。
    for x, y in batches:
        # 在当前batch上运行train_op并计算损失值。交叉熵损失函数计算的就是下一个单
        # 词为给定单词的概率
        cost, state, _ = session.run(
            [model.cost, model.final_state, train_op],
            {model.input_data: x, model.targets: y,
             model.initial_state: state})
        total_costs += cost
        iters += model.num_steps
    # perplexity计算公式
    return step, np.exp(total_costs / iters)


def read_data(file_path):
    """从文件中读取数据，并返回包含单词编号的数组"""
    with open(file_path, "r") as fin:
        # 将整个文档读进一个长字符串。
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]  # 将读取的单词编号转为整数
    return id_list


def make_batches(id_list, batch_size, num_step):
    """生成batch输出"""
    num_batches = (len(id_list) - 1) // (batch_size * num_step)

    # 将数据整理成一个维度为[batch_size, num_batches * num_step]
    data = np.array(id_list[: num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches * num_step])
    # 沿着第二个维度将数据切分成num_batches个batch，存入一个数组，画图理解最佳
    data_batches = np.split(data, num_batches, axis=1)

    # RNN每一步输出所要预测的下一个单词
    label = np.array(id_list[1 : num_batches * batch_size * num_step + 1])
    label = np.reshape(label, [batch_size, num_batches * num_step])
    label_batches = np.split(label, num_batches, axis=1)

    # 返回一个长度为num_batches的数组，其中每一项包括一个data矩阵和一个label矩阵
    return list(zip(data_batches, label_batches))


# 主函数
def main():
    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("language_model",
                           reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    # 定义测试用的循环神经网络模型。它与train_model共用参数，但是没有dropout。
    with tf.variable_scope("language_model",
                           reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    # 训练模型。
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        train_batches = make_batches(
            read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
        eval_batches = make_batches(
            read_data(EVAL_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)
        test_batches = make_batches(
            read_data(TEST_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)

        step = 0
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            step, train_pplx = run_epoch(session, train_model, train_batches,
                                         train_model.train_op, True, step)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_pplx))

            _, eval_pplx = run_epoch(session, eval_model, eval_batches,
                                     tf.no_op(), False, 0)
            print("Epoch: %d Eval Perplexity: %.3f" % (i + 1, eval_pplx))

        _, test_pplx = run_epoch(session, eval_model, test_batches,
                                 tf.no_op(), False, 0)
        print("Test Perplexity: %.3f" % test_pplx)


if __name__ == "__main__":
    main()