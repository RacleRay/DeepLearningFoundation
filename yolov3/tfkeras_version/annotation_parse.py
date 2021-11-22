import os
import numpy as np
from xml.etree.ElementTree import parse
np.random.seed(1337)


class PascalVocXmlParser(object):
    def __init__(self):
        pass

    def get_fname(self, annotation_file):
        root = self._root_tag(annotation_file)
        return root.find("filename").text

    def get_width(self, annotation_file):
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'width' in elem.tag:
                return int(elem.text)

    def get_height(self, annotation_file):
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'height' in elem.tag:
                return int(elem.text)

    def get_labels(self, annotation_file):
        root = self._root_tag(annotation_file)
        labels = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            labels.append(t.find("name").text)
        return labels

    def get_boxes(self, annotation_file):
        root = self._root_tag(annotation_file)
        bbs = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            box_tag = t.find("bndbox")
            x1 = box_tag.find("xmin").text
            y1 = box_tag.find("ymin").text
            x2 = box_tag.find("xmax").text
            y2 = box_tag.find("ymax").text
            box = np.array([
                int(float(x1)),
                int(float(y1)),
                int(float(x2)),
                int(float(y2))
            ])
            bbs.append(box)
        bbs = np.array(bbs)
        return bbs

    def _root_tag(self, fname):
        tree = parse(fname)
        root = tree.getroot()
        return root

    def _tree(self, fname):
        tree = parse(fname)
        return tree


class Annotation(object):
    def __init__(self, filename):
        self.fname = filename
        self.labels = []
        self.coded_labels = []
        self.boxes = None

    def add_object(self, x1, y1, x2, y2, name, code):
        self.labels.append(name)
        self.coded_labels.append(code)

        if self.boxes is None:
            self.boxes = np.array([x1, y1, x2, y2]).reshape(-1, 4)
        else:
            box = np.array([x1, y1, x2, y2]).reshape(-1, 4)
            self.boxes = np.concatenate([self.boxes, box])


def parse_annotation(ann_fname, img_dir, labels_naming=[]):
    parser = PascalVocXmlParser()
    fname = parser.get_fname(ann_fname)

    annotation = Annotation(os.path.join(img_dir, fname))

    labels = parser.get_labels(ann_fname)
    boxes = parser.get_boxes(ann_fname)

    for label, box in zip(labels, boxes):
        x1, y1, x2, y2 = box
        if label in labels_naming:
            annotation.add_object(x1,
                                  y1,
                                  x2,
                                  y2,
                                  name=label,
                                  code=labels_naming.index(label))
    return annotation.fname, annotation.boxes, annotation.coded_labels


def get_unique_labels(files):
    parser = PascalVocXmlParser()
    labels = []
    for fname in files:
        labels += parser.get_labels(fname)
        labels = list(set(labels))
    labels.sort()
    return labels


if __name__ == '__main__':
    import glob

    PROJECT_ROOT = os.path.dirname(__file__)  #获取当前目录
    LABELS = ['0', "1", "2", "3", '4', '5', '6', '7', '8', "9"]

    ann_dir = os.path.join(PROJECT_ROOT, "data", "ann", "*.xml")
    img_dir = os.path.join(PROJECT_ROOT, "data", "img")
    train_ann_fnames = glob.glob(ann_dir)  #获取该路径下的xml文件
    for fname in train_ann_fnames:
        train_anns = parse_annotation(fname, img_dir, labels_naming=LABELS)
        print(train_anns[0], train_anns[1], train_anns[2])