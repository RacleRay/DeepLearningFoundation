#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from cifardata_load import CifarData
from variable_summary import variable_summary


def load_data(cifar_path):
    train_filenames = [os.path.join(cifar_path, 'data_batch_%d' % i) 
                      for i in range(1, 6)]
    test_filenames = [os.path.join(cifar_path, 'test_batch')]

    train_data = CifarData(train_filenames, True)
    test_data = CifarData(test_filenames, False)

    return train_data, test_data


def data_ang(data, batch_size):
    """数据增强"""
    x_image_arr = tf.split(x_image, num_or_size_splits=batch_size, axis=0)
    result_x_image_arr = []

    for x_single_image in x_image_arr:
        # x_single_image: [1, 32, 32, 3] -> [32, 32, 3]
        x_single_image = tf.reshape(x_single_image, [32, 32, 3])
        data_aug_1 = tf.image.random_flip_left_right(x_single_image)
        data_aug_2 = tf.image.random_brightness(data_aug_1, max_delta=63)
        data_aug_3 = tf.image.random_contrast(data_aug_2, lower=0.2, upper=1.8)
        x_single_image = tf.reshape(data_aug_3, [1, 32, 32, 3])
        result_x_image_arr.append(x_single_image)
    result_x_images = tf.concat(result_x_image_arr, axis=0)

    return result_x_images


def conv_wrapper(inputs, 
                 name,
                 is_training,
                 output_channel=32, 
                 kernel_size=(3,3), 
                 activation=tf.nn.relu, 
                 padding='same'):
    with tf.name_scope(name):
        conv2d = tf.layers.conv2d(inputs,
                                  filters=output_channel,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  activation=None,
                                  name=name+"-notBN",)
        # bn在训练和测试时不一样
        # 训练时均值是在batch中求，测试是在整个数据集上求（加权平均）
        bn = tf.layers.batch_normalization(conv2d, training=is_training)
        return activation(bn)


def pooling_wrapper(inputs, name):
    return tf.layers.max_pooling2d(inputs,
                                  pool_size=(2,2),
                                  strides=(2,2),
                                  name=name)


def vggNet(batch_size, output_class, is_data_ang=False):
    # 重置默认的图
    tf.reset_default_graph()
    # 定义图的基本信息
    with tf.Graph().as_default() as graph_default:
        x = tf.placeholder(tf.float32, [batch_size, 3072])
        y = tf.placeholder(tf.int64, [batch_size])
        is_training = tf.placeholder(tf.bool, []) 
        # [None], eg: [0,5,6,3]
        x_image = tf.reshape(x, [-1, 3, 32, 32])
        # 32*32
        x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

        if is_data_ang == True:
            x_image = data_ang(x_image, batch_size)
            
        # conv1: 神经元图， feature_map, 输出图像
        conv1_1 = conv_wrapper(x_image, 'conv1_1', is_training)
        conv1_2 = conv_wrapper(conv1_1, 'conv1_2', is_training)
        conv1_3 = conv_wrapper(conv1_2, 'conv1_3', is_training)
        # 16 * 16
        pooling1 = pooling_wrapper(conv1_3, 'pool1')

        conv2_1 = conv_wrapper(pooling1, 'conv2_1', is_training, output_channel=64)
        conv2_2 = conv_wrapper(conv2_1, 'conv2_2', is_training, output_channel=64)
        conv2_3 = conv_wrapper(conv2_2, 'conv2_3', is_training, output_channel=64)
        # 8 * 8
        pooling2 = pooling_wrapper(conv2_3, 'pool2')

        conv3_1 = conv_wrapper(pooling2, 'conv3_1', is_training, output_channel=128)
        conv3_2 = conv_wrapper(conv3_1,  'conv3_2', is_training, output_channel=128)
        conv3_3 = conv_wrapper(conv3_2, 'conv3_3', is_training, output_channel=128)
        # 4 * 4
        pooling3 = pooling_wrapper(conv3_3, 'pool3')

        # [None, 4 * 4 * 32]
        if is_training:
            droprate1 = 0.5
            droprate2 = 0.2
        else:
            droprate1 = 0
            droprate2 = 0

        fc1 = tf.layers.dense(flatten, 1280, 
                kernel_initializer=tf.truncated_normal_initializer())
        drop1 = tf.layers.dropout(fc1, rate=droprate1)
        fc2 = tf.layers.dense(drop1, 640,
                kernel_initializer=tf.truncated_normal_initializer())
        drop2 = tf.layers.dropout(fc2, rate=droprate2)
        y_ = tf.layers.dense(drop2, output_class)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
        # indices
        predict = tf.argmax(y_, 1)
        correct_prediction = tf.equal(predict, y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

        with tf.name_scope('train_op'):
            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

        with tf.name_scope('summary'):
            variable_summary(conv1_1, 'conv1_1')
            variable_summary(conv1_2, 'conv1_2')
            variable_summary(conv1_2, 'conv1_3')

            variable_summary(conv2_1, 'conv2_1')
            variable_summary(conv2_2, 'conv2_2')
            variable_summary(conv2_2, 'conv2_3')

            variable_summary(conv3_1, 'conv3_1')
            variable_summary(conv3_2, 'conv3_2')
            variable_summary(conv3_2, 'conv3_3')

        # x_image输入之前经过处理，先在还原
        # source_image = (x_image + 1) * 127.5
        # 处理后的图
        # inputs_summary = tf.summary.image('inputs_image', source_image)  

        loss_summary = tf.summary.scalar('loss', loss)
        # 'loss': <10, 1.1>, <20, 1.08>
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

        merged_summary = tf.summary.merge_all()
        merged_summary_test = tf.summary.merge([loss_summary, accuracy_summary])

        init = tf.global_variables_initializer()


if __name__ == '__main__':
    CIFAR_DIR = "../cifar-10-python/cifar-10-batches-py/"
    train_data, test_data = load_data(CIFAR_DIR)

    batch_size = 32
    train_steps = 10000
    test_steps = 100

    # 指定文件输出路径（自动创建文件名，不用指定文件名，只需要路径）
    LOG_DIR = '.'
    run_label = 'run_vgg_tensorboard'
    run_dir = os.path.join(LOG_DIR, run_label)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    train_log_dir = os.path.join(run_dir, 'train')
    test_log_dir = os.path.join(run_dir, 'test')
    if not os.path.exists(train_log_dir):
        os.mkdir(train_log_dir)
    if not os.path.exists(test_log_dir):
        os.mkdir(test_log_dir)
        
        
    model_dir = os.path.join(run_dir, 'model')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    saver = tf.train.Saver()

    # 需要恢复的模型文件名
    model_name = 'ckp-00000'
    model_path = os.path.join(model_dir, model_name)

    # 输出summary是比较耗时的，指定输出的步骤
    output_summary_every_steps = 100
    # 保存模型的间隔
    output_model_every_steps = 100

    config = tf.ConfigProto(
        allow_soft_placement=True, # 系统自动选择运行cpu或者gpu
        log_device_placement=False # 是否需要打印设备日志
    )

    vggNet(batch_size, 10, False)
    with tf.Session(graph=graph_default, config=config) as sess:
        sess.run(init)
        # 打开summary文件写入
        # If you pass a `Graph` to the constructor it is added to the event file.
        train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        test_writer = tf.summary.FileWriter(test_log_dir)

        # 在循环之前保存一组fixed_test_batch
        fixed_test_batch_data, fixed_test_batch_labels \
            = test_data.next_batch(batch_size)
        
        # 检查需要加载的模型是否存在，存在直接加载checkpoint继续训练
        if os.path.exists(model_path + '.index'):
            saver.restore(sess, model_path)
            print('model restored from %s' % model_path)
        else:
            print('model %s does not exist' % model_path)
        
        for i in range(train_steps):
            batch_data, batch_labels = train_data.next_batch(batch_size)
            # 根据步数选择是否输出summary
            eval_ops = [loss, accuracy, train_op]
            should_output_summary = ((i+1) % output_summary_every_steps == 0)
            if should_output_summary:
                eval_ops.append(merged_summary)
            # eval_ops_results接收结果
            eval_ops_results = sess.run(eval_ops,
                                        feed_dict={
                                            x: batch_data,
                                            y: batch_labels,
                                            is_training: True})
            loss_val, acc_val = eval_ops_results[0:2]
            
            if should_output_summary:
                # 写入train_summary
                train_summary_str = eval_ops_results[-1]
                # 指定相应的步数
                train_writer.add_summary(train_summary_str, i+1)
                # fixed_test_batc固定测试集，使得测试结果更具有可比性
                # 输出为一个list，所以在最后加【0】
                test_summary_str = sess.run([merged_summary_test],
                                            feed_dict={
                                                x: fixed_test_batch_data,
                                                y: fixed_test_batch_labels,
                                                is_training: False
                                            })[0]
                test_writer.add_summary(test_summary_str, i+1)
            
            if (i+1) % 100 == 0:
                print(('[Train] Step: %d, loss: %4.5f, acc: %4.5f' 
                      % (i+1, loss_val, acc_val)))
            if (i+1) % 1000 == 0:
                all_test_acc_val = []
                for j in range(test_steps):
                    test_batch_data, test_batch_labels \
                        = test_data.next_batch(batch_size)
                    test_acc_val = sess.run(
                        [accuracy],
                        feed_dict = {
                            x: test_batch_data, 
                            y: test_batch_labels,
                            is_training: False
                        })
                    all_test_acc_val.append(test_acc_val)
                test_acc = np.mean(all_test_acc_val)
                print(('[Test ] Step: %d, acc: %4.5f' % (i+1, test_acc)))     
            
            # 保存模型：saver默认只保存最近的5次结果
            if (i+1) % output_model_every_steps == 0:
                saver.save(sess, 
                           os.path.join(model_dir, 'ckp-%05d' % (i+1)))
                print('model saved to ckp-%05d' % (i+1))