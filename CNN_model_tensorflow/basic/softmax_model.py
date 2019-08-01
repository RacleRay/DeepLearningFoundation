#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import cifardata_load

CIFAR_DIR = "../cifar-10-python/cifar-10-batches-py/"

train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i)
                   for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)


# -----------------------------------------------------------------
# 构建计算图
# 输入
x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])  # [None]  [0, 3, 1, 2, ....]
# y作one-hot
y_one_hot = tf.one_hot(y, 10, dtype=tf.float32)

# 参数
w = tf.get_variable(
        'w', [x.get_shape()[-1], 10],
        initializer=tf.random_normal_initializer(0, 1))
b = tf.get_variable(
        'b', [10], 
        initializer=tf.constant_initializer(0.0))

# 计算
y_ = tf.matmul(x, w) + b    # [None, 3072] * [3072, 10] = [None, 10]
# p_y = tf.nn.softmax(y_)   # [None, 10]

# 定义损失函数
# L2
# loss = tf.reduce_mean(tf.square(y_one_hot - p_y)) 
# cross_entropy
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

# 计算准确率
predict = tf.argmax(y_, 1)                 # indeces
correct_prediction = tf.equal(predict, y)  # [true, false]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

# 计算图
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


# -----------------------------------------------------------------
init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100


with tf.Session() as sess:
    sess.run(init)
    all_test_acc_val = []
    
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, acc_val, _ = sess.run(
            [loss, accuracy, train_op], 
            feed_dict={
                x: batch_data, 
                y: batch_labels})
         
        if (i+1) % 500 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' 
                  % (i+1, loss_val, acc_val))
            
        if (i+1) % 5000 == 0:
            test_data = CifarData(test_filenames, False)
            for j in  range(test_steps):
                test_batch_data, test_batch_labels \
                        = test_data.next_batch(batch_size)
                test_acc_val = sess.run(
                        [accuracy], 
                        feed_dict = {
                            x: test_batch_data,
                            y: test_batch_labels
                        })
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test ] Step: %d, acc: %4.5f' % (i+1, test_acc))