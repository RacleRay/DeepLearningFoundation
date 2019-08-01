#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


def inception_block(x,
                    output_channel_for_each_path,
                    name):
    """inception block implementation V2
    Args:
    - x: 输入，四通道
    - output_channel_for_each_path: 每个部分output_channel，eg: [10, 20, 5]
    - name: inception_block name
    """
    with tf.variable_scope(name):   # tf.variable_scope避免命名冲突，带name前缀
        with tf.variable_scope('Branch_0'):
            conv1_1 = tf.layers.conv2d(x,
                                    output_channel_for_each_path[0],
                                    (1, 1),
                                    strides = (1,1),
                                    padding = 'same',
                                    activation = tf.nn.relu,
                                    name = 'conv1_1')
        with tf.variable_scope('Branch_1'):
            conv3_3 = tf.layers.conv2d(x,
                                    2*output_channel_for_each_path[1]//3,
                                    (1, 1),
                                    strides = (1,1),
                                    padding = 'same',
                                    name = 'conv3_3a')
            conv3_3 = tf.layers.conv2d(conv3_3,
                                    output_channel_for_each_path[1],
                                    (3, 3),
                                    strides = (1,1),
                                    padding = 'same',
                                    activation = tf.nn.relu,
                                    name = 'conv3_3b')
        with tf.variable_scope('Branch_2'):
            conv5_5 = tf.layers.conv2d(x,
                                    2*output_channel_for_each_path[2]//3,
                                    (1, 1),
                                    strides = (1,1),
                                    padding = 'same',
                                    name = 'conv5_5a')
            conv5_5 = tf.layers.conv2d(conv5_5,
                                    output_channel_for_each_path[2],
                                    (3, 3),
                                    strides = (1,1),
                                    padding = 'same',
                                    name = 'conv5_5b')
            conv5_5 = tf.layers.conv2d(conv5_5,
                                    output_channel_for_each_path[2],
                                    (3, 3),
                                    strides = (1,1),
                                    padding = 'same',
                                    activation = tf.nn.relu,
                                    name = 'conv5_5c')
        with tf.variable_scope('Branch_3'):
            max_pooling = tf.layers.max_pooling2d(x,
                                                (3, 3),
                                                (2, 2),
                                                padding = 'same',
                                                name = 'max_pooling')
            max_pooling = tf.layers.conv2d(max_pooling,
                                    output_channel_for_each_path[2],
                                    (1, 1),
                                    strides = (1,1),
                                    padding = 'same',
                                    activation = tf.nn.relu,
                                    name = 'max_pooling_1x1')

    # 将max_pooling的shape，padding为其他组大小
    max_pooling_shape = max_pooling.get_shape().as_list()[1:]
    input_shape = x.get_shape().as_list()[1:]

    width_padding = (input_shape[0] - max_pooling_shape[0]) // 2
    height_padding = (input_shape[1] - max_pooling_shape[1]) // 2

    # rank of 'max_pooling' is 4
    padded_pooling = tf.pad(max_pooling,
                            [[0, 0],                           # 在第1维前后不pad
                             [width_padding, width_padding],   # 在第2维前后pad
                             [height_padding, height_padding], # 在第3维前后pad
                             [0, 0]])                          # 在第4维前后不pad

    # 通道维度上的concat
    concat_layer = tf.concat(
        [conv1_1, conv3_3, conv5_5, padded_pooling],
        axis = 3)
    return concat_layer






