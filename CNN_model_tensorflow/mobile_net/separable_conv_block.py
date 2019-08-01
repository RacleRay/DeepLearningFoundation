#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


def separable_conv_block(x,
                         output_channel_number,
                         name):
    """separable_conv block implementation
    Args:
    - x: 
    - output_channel_number: output channel of the final 1*1 conv layer.
    - name: 
    """
    with tf.variable_scope(name):
        input_channel = x.get_shape().as_list()[-1]
        # channel_wise_x: [channel1, channel2, ...]
        channel_wise_x = tf.split(x, input_channel, axis=3)
        
        output_channels = []
        for i in range(len(channel_wise_x)):
            output_channel = tf.layers.conv2d(channel_wise_x[i],
                                              1,
                                              (3, 3),
                                              strides = (1,1),
                                              padding = 'same',
                                              name = 'conv_%d' % i)
            output_channels.append(output_channel)

        concat_layer = tf.concat(output_channels, axis = 3)

        # 此部分在简化模型可以省略
        concat_layer = tf.layers.batch_normalization(concat_layer)
        concat_layer = tf.nn.relu(concat_layer)

        # final 1*1 conv layer
        conv1_1 = tf.layers.conv2d(concat_layer,
                                   output_channel_number,
                                   (1,1),
                                   strides = (1,1),
                                   padding = 'same',
                                   activation = tf.nn.relu,
                                   name = 'conv1_1')

        conv1_1 = tf.layers.batch_normalization(conv1_1)
        conv1_1 = tf.nn.relu(conv1_1)

    return conv1_1