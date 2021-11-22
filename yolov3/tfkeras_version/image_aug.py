import cv2
import numpy as np


class ImgAugment(object):
    def __init__(self, w, h, jitter):
        self._jitter = jitter
        self._w = w
        self._h = h

    def imread(self, img_file, boxes):
        image = cv2.imread(img_file)
        boxes_ = np.copy(boxes)
        if self._jitter:  #是否要增强数据
            image, boxes_ = make_jitter_on_image(image, boxes_)

        image, boxes_ = resize_image(image, boxes_, self._w, self._h)
        return image, boxes_


def make_jitter_on_image(image, boxes):
    h, w, _ = image.shape

    ### scale the image
    scale = np.random.uniform() / 10. + 1.
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    ### translate the image
    max_offx = (scale - 1.) * w
    max_offy = (scale - 1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)

    image = image[offy:(offy + h), offx:(offx + w)]

    ### flip the image
    flip = np.random.binomial(1, .5)
    if flip > 0.5:
        image = cv2.flip(image, 1)
        is_flip = True
    else:
        is_flip = False

    aug_pipe = _create_augment_pipeline()
    image = aug_pipe.augment_image(image)

    # fix object's position and size
    new_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = int(x1 * scale - offx)
        x2 = int(x2 * scale - offx)

        y1 = int(y1 * scale - offy)
        y2 = int(y2 * scale - offy)

        if is_flip:
            xmin = x1
            x1 = w - x2
            x2 = w - xmin
        new_boxes.append([x1, y1, x2, y2])
    return image, np.array(new_boxes)


def resize_image(image, boxes, desired_w, desired_h):
    h, w, _ = image.shape

    # resize the image to standard size
    image = cv2.resize(image, (desired_h, desired_w))
    image = image[:, :, ::-1]

    # fix object's position and size
    new_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = int(x1 * float(desired_w) / w)
        x1 = max(min(x1, desired_w), 0)
        x2 = int(x2 * float(desired_w) / w)
        x2 = max(min(x2, desired_w), 0)

        y1 = int(y1 * float(desired_h) / h)
        y1 = max(min(y1, desired_h), 0)
        y2 = int(y2 * float(desired_h) / h)
        y2 = max(min(y2, desired_h), 0)

        new_boxes.append([x1, y1, x2, y2])
    return image, np.array(new_boxes)


def _create_augment_pipeline():
    from imgaug import augmenters as iaa
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    aug_pipe = iaa.Sequential(
        [
            sometimes(iaa.Affine()),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images

                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    #iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    #iaa.Grayscale(alpha=(0.0, 1.0)),
                    #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return aug_pipe
