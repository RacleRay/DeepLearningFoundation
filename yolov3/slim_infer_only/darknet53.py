import tensorflow as tf

slim = tf.contrib.slim


# 定义darknet块：一个残差连接，一个同尺度卷积再加一个下采样卷积
def _darknet53_block(inputs, filters):
    shortcut = inputs
    inputs = slim.conv2d(inputs, filters, 1, stride=1,
                         padding='SAME')  #正常卷积
    inputs = slim.conv2d(inputs, filters * 2, 3, stride=1,
                         padding='SAME')  #正常卷积

    inputs = inputs + shortcut
    return inputs


def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    assert strides > 1

    inputs = _fixed_padding(inputs, kernel_size)  #外围填充0，好支持valid卷积
    inputs = slim.conv2d(inputs,
                         filters,
                         kernel_size,
                         stride=strides,
                         padding='VALID')

    return inputs


# 对指定输入填充0
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    #inputs 【b,h,w,c】  pad  b,c不变。h和w上下左右，填充0.kernel = 3 ，则上下左右各加一趟0
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]],
        mode=mode)
    return padded_inputs


#定义Darknet-53 模型.返回3个不同尺度的特征
def darknet53(inputs):
    inputs = slim.conv2d(inputs, 32, 3, stride=1, padding='SAME')  #正常卷积
    inputs = _conv2d_fixed_padding(
        inputs, 64, 3, strides=2)  #需要填充,并使用了'VALID' (-1, 208, 208, 64)

    inputs = _darknet53_block(inputs, 32)  #darknet块
    inputs = _conv2d_fixed_padding(inputs, 128, 3, strides=2)

    for i in range(2):
        inputs = _darknet53_block(inputs, 64)
    inputs = _conv2d_fixed_padding(inputs, 256, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 128)
    route_1 = inputs  #特征1 (-1, 52, 52, 128)

    inputs = _conv2d_fixed_padding(inputs, 512, 3, strides=2)
    for i in range(8):
        inputs = _darknet53_block(inputs, 256)
    route_2 = inputs  #特征2  (-1, 26, 26, 256)

    inputs = _conv2d_fixed_padding(inputs, 1024, 3, strides=2)
    for i in range(4):
        inputs = _darknet53_block(inputs, 512)  #特征3 (-1, 13, 13, 512)

    # 在原有的darknet53，还会跟一个全局池化。这里没有使用。所以其实是只有52层
    return route_1, route_2, inputs
