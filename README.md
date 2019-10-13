# DeepLearningFoundation
Neural network algorithm、architecture and some materials.

## 第一部分 [使用numpy完成神经网络算法](NeuralNetwork.ipynb)
每个部分均给出代码及运行结果
> Table of Contents
> - 1  逻辑门
> - 2  常用函数
> - 3  梯度求解
> - 4  简单网络，数值求解
> - 5  BP算法公式版
>      - 5.1  乘法、加法操作的简单定义
>      - 5.2  主要layer的实现
>      - 5.3  BP算法初步双层网络
>      - 5.4  模型优化、参数调优
>        - 5.4.1  权重初始化实验
>        - 5.4.2  为方便实验，先定义一个多层全连接网络
>        - 5.4.3  MNIST权重初始化影响
>        - 5.4.4  超参数搜索
>           - 5.4.4.1  Trainer训练集成类
>           - 5.4.4.2  超参数的随机搜索
>        - 5.4.5  优化梯度更新算法
>           - 5.4.5.1  SGD
>           - 5.4.5.2  Momentum
>           - 5.4.5.3  Nesterov
>           - 5.4.5.4  Adagrad
>           - 5.4.5.5  Adadelta
>           - 5.4.5.6  RMSprop
>           - 5.4.5.7  Adam
>           - 5.4.5.8  Nadam(结合Nesterov的思路)
>           - 5.4.5.9  AdaBound(加入lr动态上下界)
>        - 5.4.6  求解算法对比
>           - 5.4.6.1  简单方程对比
>           - 5.4.6.2  MINIST数据集对比
>        - 5.4.7  batch_norm
>           - 5.4.7.1  新建包含BN和dropout的网络
>           - 5.4.7.2  对比有误BN的情况
>        - 5.4.8  过拟合：L2和Dropout
>           - 5.4.8.1  Trainer训练集成类（和前面相同）
>           - 5.4.8.2  Dropout
>           - 5.4.8.3  L2
> - 6  Convolution卷积
>      - 6.1  卷积层
>      - 6.2  Pooling层
>      - 6.3  简单ConvNet
>      - 6.4  可视化filter
>      - 6.5  使用学习好的filter
> - 7  Deep_convnet
>      - 7.1  浮点运算的影响
> - 8  [RNN](RNN-LSTM.ipynb)
> - 9  [LSTM](RNN-LSTM.ipynb)

**_BONUS_** : [Perceptron](Perceptron.ipynb)  
**_BONUS_** : [拓扑排序算法 -- 实现神经网络框架图计算逻辑](build_NN.ipynb)

## 第二部分 [CNN常用Model使用TensorFlow实现](CNN_model_tensorflow/)
使用TensorFlow1.6完成了常用CNN基本结构的构建。
> - Inception V3 naive
> - resNet naive
> - VGGNet naive
> - MobileNet naive

## 第三部分 [RNN使用TensorFlow实现](RNN_tensorflow/)
进行基本文本预处理，其中batch处理需要一些小技巧。embedding_lookup和softmax共享权重矩阵。使用perplexity评价LM。
Seq2Seq attention的实现。
> - LM
> - Seq2Seq attention
>   - dynamic_rnn:排除padding state传递影响，直接使用上一步state覆盖
>   - sequence_mask:排除padding在计算loss时的影响
>   - encoder bidirectional_rnn:attention需要主要到分别来自上下文的语义信息
>   - decoder while_loop:实现动态inference
>   - decoder:只继承attention之后的encoder outputs，不继承encoder state，decoder可以更自由的选择结构形式
>   - decoder:embedding和softmax共享权重

## 第四部分 [TensorFlow高层封装](TF_tools/)
> - Keras的两种建模方式
> - Slim的简单使用
> - Estimator自定义模型，使用Dataset对象

## 第五部分 [ML+DL算法模型](ML_Algorithm/)
> - [SVM](ML_Algorithm/SVM.py)
>   - 理解SVM算法原理的demo，核心求解目标：max(||w||) and min(b), s.t. y(wx + b) >= 1。
>   - soft margin的实现可以将边界向decision boundary增加一个容忍量
> - [VAE](VAE/)
>   - DL生成式模型，keras和pytorch版本
> - [GAN](GAN/)
>   - WGAN实现，CGAN在MNIST数据集上实验，ACGAN生成人脸实验
>   - DCGAN在Have_Fun repository里实现过。
>   - 就人脸生成任务而言，ACGAN效果相对更好
> - [Capsule Net](CapsNet/)
>   - 根据Capsule Net原论文，参考[实现1](https://github.com/naturomics/CapsNet-Tensorflow.git), [实现1](https://github.com/XifengGuo/CapsNet-Pytorch)，修改了部分模型计算过程，在MNIST数据集上实验。
