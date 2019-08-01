#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os
import tensorflow as tf


CIFAR_DIR = "../cifar-10-python/cifar-10-batches-py/"


class CifarData:
	"""数据处理"""
	def __init__(self, filenames, need_shuffle, is_binary=False):
        """filenames: 区分训练和测试
           need_shuffle：训练集需要打乱，测试集不需要
           is_binary：返回两类，还是多类"""
        all_data = []
        all_labels = []
        
        if is_binary == True:
            for filename in filenames:
                data, labels = self.load_data(filename)
                for item, label in zip(data, labels):
                    if label in [0, 1]:     # 进入logitic二分类
                        all_data.append(item)
                        all_labels.append(label)
        else:
            for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)           # 全部数据导入，不同于二分类
            all_labels.append(labels)
        
        self._data = np.vstack(all_data)    # 纵向
        self._data = self._data / 127.5 - 1 # 归一化，转换到[-1, 1]之间，sigmoid
        self._labels = np.hstack(all_labels)# 横向
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0                 # 指针
        if self._need_shuffle:
            self._shuffle_data()


    @staticmethod
    def load_data(filename):
        """read data from data file."""
        with open(filename, "rb") as f:
            data = pickle.load(f, encoding="bytes")
            return data[b"data"], data[b"labels"]
    

    def _shuffle_data(self):
        """随机打乱数据"""
        # [0,1,2,3,4,5] -> [5,3,2,4,0,1]
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]


    def next_batch(self, batch_size):
        """返回batch_size大小的batch"""
        end_indicator = self._indicator + batch_size
        
        # batch_size小于_num_examples
        if end_indicator > self._num_examples:
            if self._need_shuffle:   # _need_shuffle真，可以循环使用
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("没有样本了")
        
        # batch_size大于_num_examples
        if end_indicator > self._num_examples:
            raise Exception("batch size大于num_examples") 
        
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator

        return batch_data, batch_labels