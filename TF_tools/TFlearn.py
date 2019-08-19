import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist
# pip install tflearn


trainX, trainY, testX, testY = mnist.load_data(
    data_dir="../datasets/MNIST_data", one_hot=True)
# 将图像数据resize成卷积卷积神经网络输入的格式。
trainX = trainX.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# 构建神经网络。
net = input_data(shape=[None, 28, 28, 1], name='input')
net = conv_2d(net, 32, 5, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 64, 5, activation='relu')
net = max_pool_2d(net, 2)
net = fully_connected(net, 500, activation='relu')
net = fully_connected(net, 10, activation='softmax')

# 定义学习任务。指定优化器为sgd，学习率为0.01，损失函数为交叉熵
net = regression(net, optimizer='sgd', learning_rate=0.01,
                 loss='categorical_crossentropy')


model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=10,
          validation_set=([testX, testY]),
          show_metric=True)