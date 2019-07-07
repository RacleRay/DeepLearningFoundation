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

**_BONUS_** : [Perceptron](Perceptron.ipynb)
