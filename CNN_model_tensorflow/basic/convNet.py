#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from cifardata_load import CifarData


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
x_image = tf.reshape(x, [-1, 3, 32, 32])
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])  # RGB
y = tf.placeholder(tf.int64, [None])  # [None]  [0, 3, 1, 2, ....]

# conv1: 神经元图， feature_map, 输出图像
conv1 = tf.layers.conv2d(x_image,
                        32,                # output channel number
                        (3, 3),
                        padding='same',
                        activation=tf.nn.relu,
                        name='conv1')

# 16 * 16
pooling1 = tf.layers.max_pooling2d(conv1,
                                  (2, 2),
                                  (2, 2),
                                  name='pool1')

conv2 = tf.layers.conv2d(pooling1,
                         32,               # output channel number
                         (3,3),            # kernel size
                         padding = 'same',
                         activation = tf.nn.relu,
                         name = 'conv2')

# 8 * 8
pooling2 = tf.layers.max_pooling2d(conv2,
                                   (2, 2), # kernel size
                                   (2, 2), # stride
                                   name = 'pool2')

conv3 = tf.layers.conv2d(pooling2,
                         32,               # output channel number
                         (3,3),            # kernel size
                         padding = 'same',
                         activation = tf.nn.relu,
                         name = 'conv3')

# 4 * 4 * 32
pooling3 = tf.layers.max_pooling2d(conv3,
                                   (2, 2), # kernel size
                                   (2, 2), # stride
                                   name = 'pool3')

# [None, 4 * 4 * 32]
flatten = tf.contrib.layers.flatten(pooling3)
y_ = tf.layers.dense(flatten, 10)

# 定义损失函数
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


