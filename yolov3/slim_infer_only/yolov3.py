import tensorflow as tf
from detector import *
from darknet53 import *

slim = tf.contrib.slim


def _upsample(inputs, out_shape):
    # tf.image.resize_bilinear结果较差
    inputs = tf.image.resize_nearest_neighbor(inputs,
                                              (out_shape[1], out_shape[2]))
    inputs = tf.identity(inputs, name='upsampled')
    return inputs


#定义yolo函数
def yolo_v3(inputs,
            num_classes,
            is_training=False,
            data_format='NHWC',
            reuse=False):

    assert data_format == 'NHWC'

    img_size = inputs.get_shape().as_list()[1:3]  # 获得输入图片大小

    inputs = inputs / 255  # 归一化

    #定义批量归一化参数
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,
    }

    #定义yolo网络
    with slim.arg_scope([slim.conv2d, slim.batch_norm], data_format=data_format, reuse=reuse):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            biases_initializer=None,
                            activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):

            with tf.variable_scope('darknet-53'):
                # route_1 (-1, 52, 52, 128)
                # route_2 (-1, 26, 26, 256)
                # inputs (-1, 13, 13, 512)
                route_1, route_2, inputs = darknet53(inputs)

            with tf.variable_scope('yolo-v3'):
                # inputs (-1, 13, 13, 1024)  route (-1, 13, 13, 512)
                route, inputs = _yolo_block(inputs, 512)
                detect_1 = _detection_layer(inputs, num_classes, _ANCHORS[6:9], img_size)  # 使用大检测框
                detect_1 = tf.identity(detect_1, name='detect_1')

                inputs = slim.conv2d(route, 256, 1, stride=1, padding='SAME')  #(-1, 13, 13, 256)
                upsample_size = route_2.get_shape().as_list()
                inputs = _upsample(inputs, upsample_size)  #(-1, 26, 26, 256)
                inputs = tf.concat([inputs, route_2], axis=3)  #(-1, 26, 26, 512)

                #inputs (-1, 26, 26, 512)   route (-1, 26, 26, 256)
                route, inputs = _yolo_block(inputs, 256)
                detect_2 = _detection_layer(inputs, num_classes, _ANCHORS[3:6], img_size)
                detect_2 = tf.identity(detect_2, name='detect_2')

                inputs = slim.conv2d(route, 128, 1, stride=1, padding='SAME') #(-1, 26, 26, 128)
                upsample_size = route_1.get_shape().as_list()
                inputs = _upsample(inputs, upsample_size)  #(-1, 52, 52, 128)
                inputs = tf.concat([inputs, route_1], axis=3)  #(-1, 52, 52, 256)

                _, inputs = _yolo_block(inputs, 128)  # inputs (-1, 52, 52, 256)

                detect_3 = _detection_layer(inputs, num_classes, _ANCHORS[0:3], img_size)
                detect_3 = tf.identity(detect_3, name='detect_3')

                # 返回了3个尺度。每个尺度里又包含3个结果(-1, 10647（ 507 +2028 + 8112）, 5+c)
                # 三个尺度的形状分别为：[1, 507（13*13*3）, 5+c]、[1, 2028, 5+c]、[1, 8112, 5+c]
                detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
                detections = tf.identity(detections, name='detections')
                return detections


'''--------Test the scale--------'''
if __name__ == "__main__":
    tf.reset_default_graph()
    import cv2
    data = cv2.imread('timg.jpg')
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = cv2.resize(data, (416, 416))
    data = tf.cast(tf.expand_dims(tf.constant(data), 0), tf.float32)

    detections = yolo_v3(data, 3, data_format='NHWC')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(detections).shape)
