#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from cifardata_load import CifarData


def load_data(cifar_path):
    train_filenames = [os.path.join(cifar_path, 'data_batch_%d' % i) 
                      for i in range(1, 6)]
    test_filenames = [os.path.join(cifar_path, 'test_batch')]

    train_data = CifarData(train_filenames, True)
    test_data = CifarData(test_filenames, False)

    return train_data, test_data


def gen_graph():
    # 重置默认的图
    tf.reset_default_graph()
    # 定义图的基本信息
    with tf.Graph().as_default() as graph_default:
        x = tf.placeholder(tf.float32, [None, 3072])
        y = tf.placeholder(tf.int64, [None])
        # [None], eg: [0,5,6,3]
        x_image = tf.reshape(x, [-1, 3, 32, 32])
        # 32*32
        x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

        # conv1: 神经元图， feature_map, 输出图像
        conv1_1 = tf.layers.conv2d(
            x_image,
            32,      # output channel number
            (3, 3),  # kernel size
            padding='same',
            activation=tf.nn.relu,
            name='conv1_1')
        conv1_2 = tf.layers.conv2d(
            conv1_1,
            32,      # output channel number
            (3, 3),  # kernel size
            padding='same',
            activation=tf.nn.relu,
            name='conv1_2')
        # 16 * 16
        pooling1 = tf.layers.max_pooling2d(
            conv1_2,
            (2, 2),  # kernel size
            (2, 2),  # stride
            name='pool1')

        conv2_1 = tf.layers.conv2d(
            pooling1,
            64,      # output channel number
            (3, 3),  # kernel size
            padding='same',
            activation=tf.nn.relu,
            name='conv2_1')
        conv2_2 = tf.layers.conv2d(
            conv2_1,
            64,      # output channel number
            (3, 3),  # kernel size
            padding='same',
            activation=tf.nn.relu,
            name='conv2_2')
        # 8 * 8
        pooling2 = tf.layers.max_pooling2d(
            conv2_2,
            (2, 2),  # kernel size
            (2, 2),  # stride
            name='pool2')

        conv3_1 = tf.layers.conv2d(
            pooling2,
            128,  # output channel number
            (3, 3),  # kernel size
            padding='same',
            activation=tf.nn.relu,
            name='conv3_1')
        conv3_2 = tf.layers.conv2d(
            pooling2,
            128,  # output channel number
            (3, 3),  # kernel size
            padding='same',
            activation=tf.nn.relu,
            name='conv3_2')
        # 4 * 4 * 32
        pooling3 = tf.layers.max_pooling2d(
            conv3_2,
            (2, 2),  # kernel size
            (2, 2),  # stride
            name='pool3')
        # [None, 4 * 4 * 128]
        flatten = tf.layers.flatten(pooling3)
        fc1 = tf.layers.dense(flatten, 1280, 
            kernel_initializer=tf.truncated_normal_initializer())
        fc2 = tf.layers.dense(fc1, 640,
            kernel_initializer=tf.truncated_normal_initializer())
        y_ = tf.layers.dense(fc2, 10)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
        # y_ -> sofmax
        # y -> one_hot
        # loss = ylogy_

        # indices
        predict = tf.argmax(y_, 1)
        # [1,0,1,1,1,0,0,0]
        correct_prediction = tf.equal(predict, y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

        with tf.name_scope('train_op'):
            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

        init = tf.global_variables_initializer()



if __name__ == '__main__':
    CIFAR_DIR = "../cifar-10-python/cifar-10-batches-py/"
    train_data, test_data = load_data(CIFAR_DIR)

    gen_graph()
    
    batch_size = 20
    train_steps = 10000
    test_steps = 100

    config = tf.ConfigProto(
        allow_soft_placement=True, # 系统自动选择运行cpu或者gpu
        log_device_placement=False # 是否需要打印设备日志
    )

    # train 10k: 73.4%
    with tf.Session(graph=graph_default, config=config) as sess:
        sess.run(init)
        for i in range(train_steps):
            batch_data, batch_labels = train_data.next_batch(batch_size)
            loss_val, acc_val, _ = sess.run(
                [loss, accuracy, train_op],
                feed_dict={
                    x: batch_data,
                    y: batch_labels})
            if (i+1) % 100 == 0:
                print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' 
                      % (i+1, loss_val, acc_val))
            if (i+1) % 1000 == 0:
                test_data = CifarData(test_filenames, False)
                all_test_acc_val = []
                for j in range(test_steps):
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