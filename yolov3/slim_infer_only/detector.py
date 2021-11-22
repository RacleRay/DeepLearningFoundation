import tensorflow as tf

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

#定义候选框，来自coco数据集
_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]


#yolo检测块
def _yolo_block(inputs, filters):
    inputs = slim.conv2d(inputs, filters, 1, stride=1, padding='SAME')
    inputs = slim.conv2d(inputs, filters * 2, 3, stride=1, padding='SAME')
    inputs = slim.conv2d(inputs, filters, 1, stride=1, padding='SAME')
    inputs = slim.conv2d(inputs, filters * 2, 3, stride=1, padding='SAME')
    inputs = slim.conv2d(inputs, filters, 1, stride=1, padding='SAME')
    route = inputs
    inputs = slim.conv2d(inputs, filters * 2, 3, stride=1, padding='SAME')
    return route, inputs


#检测层
def _detection_layer(inputs, num_classes, anchors, img_size):
    print(inputs.get_shape())
    num_anchors = len(anchors)  # 候选框个数
    predictions = slim.conv2d(inputs,
                              num_anchors * (5 + num_classes),
                              1,
                              stride=1,
                              normalizer_fn=None,
                              activation_fn=None,
                              biases_initializer=tf.zeros_initializer())

    # 三个尺度的形状分别为：[1, 13, 13, 3*(5+c)]、[1, 26, 26, 3*(5+c)]、[1, 52, 52, 3*(5+c)]
    shape = predictions.get_shape().as_list()
    print("shape", shape)

    grid_size = shape[1:3]  #HW
    dim = grid_size[0] * grid_size[1]  #每个格子所包含的像素
    bbox_attrs = 5 + num_classes

    #把h和w展开成dim
    predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])

    # 缩放（多个pixel视为grid）
    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
    # 将候选框的尺寸值同比例缩小
    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

    box_centers, box_sizes, confidence, classes = tf.split(
        predictions, [2, 2, 1, num_classes], axis=-1)

    box_centers = tf.nn.sigmoid(box_centers)  # 每个grid内相对位置的预测
    confidence = tf.nn.sigmoid(confidence)

    grid_x = tf.range(grid_size[0], dtype=tf.float32)  #定义网格索引0,1,2...n
    grid_y = tf.range(grid_size[1], dtype=tf.float32)  #定义网格索引0,1,2,...m
    a, b = tf.meshgrid(grid_x, grid_y)  #生成网格矩阵

    x_offset = tf.reshape(a, (-1, 1))  #展开 一共dim个
    y_offset = tf.reshape(b, (-1, 1))

    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)  #连接 [dim,2]
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

    box_centers = box_centers + x_y_offset  #box_centers为0-1，x_y为具体网格的索引，相加后，就是真实位置(0.1+4=4.1，第4个网格里0.1的偏移)
    box_centers = box_centers * stride  #真实尺寸像素点

    anchors = tf.tile(anchors, [dim, 1])
    box_sizes = tf.exp(box_sizes) * anchors  #计算缩放情况下的边长：exp结果永远为正
    box_sizes = box_sizes * stride  #真实边长

    detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)
    classes = tf.nn.sigmoid(classes)
    predictions = tf.concat([detections, classes], axis=-1)

    # 三个尺度的形状分别为：[1, 507（13*13*3）, 5+c]、[1, 2028, 5+c]、[1, 8112, 5+c]
    print(predictions.get_shape())
    return predictions
