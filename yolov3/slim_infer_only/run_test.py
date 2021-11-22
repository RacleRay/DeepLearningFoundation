import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from yolov3 import yolo_v3
from nms import non_max_suppression, detections_boxes


# 加载权重
def load_weights(var_list, weights_file):
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # 找到卷积项
        if 'Conv' in var1.name.split('/')[-2]:
            # 找到BN参数项
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # 加载批量归一化参数
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                i += 4 # 已经加载了4个变量，指针移动4
            elif 'Conv' in var2.name.split('/')[-2]:
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                i += 1

            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            # 加载权重
            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


# 将结果显示在图片上
def draw_boxes(boxes, img, cls_names, detection_size):
    draw = ImageDraw.Draw(img)

    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)
            print('{} {:.2f}%'.format(cls_names[cls], score * 100),box[:2])

def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


# 加载数据集标签名称
def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names


def main(size, conf_threshold, iou_threshold, input_img, output_img, class_names,
         weights_file):
    tf.reset_default_graph()
    img = Image.open(input_img)
    img_resized = img.resize(size=(size, size))
    classes = load_coco_names(class_names)

    inputs = tf.placeholder(tf.float32, [None, size, size, 3])

    with tf.variable_scope('detector'):
        detections = yolo_v3(inputs, len(classes), data_format='NHWC') # 定义网络结构
        load_ops = load_weights(tf.global_variables(scope='detector'), weights_file) # 加载权重

    boxes = detections_boxes(detections)
    with tf.Session() as sess:
        sess.run(load_ops)
        detected_boxes = sess.run(boxes, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})

    filtered_boxes = non_max_suppression(detected_boxes,
                                         confidence_threshold=conf_threshold,
                                         iou_threshold=iou_threshold)

    draw_boxes(filtered_boxes, img, classes, (size, size))

    img.save(output_img)
    img.show()


if __name__ == '__main__':
    imgsize = 416
    input_img ='data/timg.jpg'
    output_img = 'data/out.jpg'
    class_names = 'data/coco.names'
    weights_file = 'model/yolov3.weights'
    conf_threshold = 0.5 #置信度阈值
    iou_threshold = 0.4  #重叠区域阈值

    main(imgsize, conf_threshold, iou_threshold,
         input_img, output_img, class_names,
         weights_file)