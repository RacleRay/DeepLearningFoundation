#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from cifardata_load import CifarData
from residual_block import residual_block


def load_data(cifar_path):
    train_filenames = [os.path.join(cifar_path, 'data_batch_%d' % i) 
                      for i in range(1, 6)]
    test_filenames = [os.path.join(cifar_path, 'test_batch')]

    train_data = CifarData(train_filenames, True)
    test_data = CifarData(test_filenames, False)

    return train_data, test_data


def res_net(x, num_residual_blocks, num_filter_base, class_num):
    """residual network implementation
    Args:
    - x: 输入
    - num_residual_blocks: 每一个convi_x(stage)的残差块的个数,eg: [3, 4, 6, 3]
                           最后一个将output_channel增加两倍
    - num_filter_base: residual_block的输入通道数目
    - class_num: 输出类别数
    """
    # 每一个stage视为一次采样
    num_subsampling = len(num_residual_blocks)
    layers = []
    # x: [None, width, height, channel] -> [width, height, channel]
    input_size = x.get_shape().as_list()[1:]

    with tf.variable_scope('conv0'):
        conv0 = tf.layers.conv2d(
                                x,
                                num_filter_base, (3, 3),
                                strides=(1, 1),
                                padding='same',
                                activation=tf.nn.relu,
                                name='conv0')
        layers.append(conv0)

    # eg:num_subsampling = 4, sample_id = [0,1,2,3]
    for sample_id in range(num_subsampling):
        for i in range(num_residual_blocks[sample_id]):
            with tf.variable_scope("conv%d_%d" % (sample_id, i)):
                conv = residual_block(layers[-1],
                                      num_filter_base * (2 ** sample_id))
                layers.append(conv)

    # 检查输出的shape
    multiplier = 2 ** (num_subsampling - 1)
    assert layers[-1].get_shape().as_list()[1:] \
        == [input_size[0] / multiplier,
            input_size[1] / multiplier,
            num_filter_base * multiplier]

    # global_pool 相当于是与特征图大小一样的kernal进行pooling
    # 输出[None, channel]
    with tf.variable_scope('fc'):
        # layer[-1].shape : [None, width, height, channel]
        # kernal_size: image_width, image_height
        global_pool = tf.reduce_mean(layers[-1], [1, 2])
        logits = tf.layers.dense(global_pool, class_num)
        layers.append(logits)
    return layers[-1]


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

        # res_net
        y_ = res_net(x_image, [2,3,2], 32, 10)

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

    # train 10k: 74.85%
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

