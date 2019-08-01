#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


def residual_block(x, output_channel):
    """residual connection implementation"""
    input_channel = x.get_shape().as_list()[-1]

    # output_channel变为2倍，需要进行进行降采样
    if input_channel * 2 == output_channel:
        increase_dim = True
        strides = (2, 2)
    # output_channel不变，不需要需要进行进行降采样
    elif input_channel == output_channel:
        increase_dim = False
        strides = (1, 1)
    else:
        raise Exception("input channel can't match output channel")

    # 卷积分支
    conv1 = tf.layers.conv2d(
                            x,
                            output_channel, (3, 3),
                            strides=strides,
                            padding='same',
                            activation=tf.nn.relu,
                            name='conv1')
    conv2 = tf.layers.conv2d(
                            conv1,
                            output_channel, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation=tf.nn.relu,
                            name='conv2')

    # 恒等变化分支
    if increase_dim:
        # [None, image_width, image_height, channel] -> [,,,channel*2]
        # 先降采样
        # 说明: padding 设为same指的不是让输入和输出一样，而是说不丢失数据。
        #       设为valid是指如果有不足的部分，就丢弃掉。
        pooled_x = tf.layers.average_pooling2d(x, 
                                               (2, 2), 
                                               (2, 2), 
                                               padding='valid')
        # 需要将channel变为两倍，只在channel维度上前后补0
        padded_x = tf.pad(pooled_x,
                        [[0, 0], [0, 0], [0, 0], 
                         [input_channel // 2, input_channel // 2]])
    else:
        # 不变channel
        padded_x = x
    
    # element-wise相加
    output_x = conv2 + padded_x
    return output_x