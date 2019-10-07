# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import montage

batch_size = 100
z_dim = 100
# dataset = 'lfw_new_imgs'
dataset = 'celeba'

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(os.path.join('samples_' + dataset, 'wgan_' + dataset + '-60000.meta'))
saver.restore(sess, tf.train.latest_checkpoint('samples_' + dataset))

# 获取计算图generator
graph = tf.get_default_graph()
g = graph.get_tensor_by_name('generator/g/Tanh:0')
noise = graph.get_tensor_by_name('noise:0')
is_training = graph.get_tensor_by_name('is_training:0')

n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
gen_imgs = sess.run(g, feed_dict={noise: n, is_training: False})

gen_imgs = (gen_imgs + 1) / 2
imgs = [img[:, :, :] for img in gen_imgs]
gen_imgs = montage(imgs)
gen_imgs = np.clip(gen_imgs, 0, 1)
plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(gen_imgs)
plt.show()