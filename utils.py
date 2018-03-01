# -*- coding: utf-8 -*-
# @Author: denglei
# @Date:   2018-03-01 12:10:25
# @Last Modified by:   denis
# @Last Modified time: 2018-03-01 16:36:09

import numpy as np


def sigmoid(z):

	return 1 / (1 + np.exp(-z))


def softmax(x):
	"""
	计算多标签的softmax概率值
	利用softmax(x) = softmax(x + c)，保证不会出现数值错误
	"""
	x = np.array(x)
	if len(x.shape) > 1:
		max_x = np.max(x, axis=1, keepdims=True)
		class_prob = np.exp(x - max_x) 
		return class_prob / np.sum(class_prob, axis=1, keepdims=True)
	else:
		max_x = np.max(x)
		class_prob = np.exp(x - max_x)
		return class_prob / np.sum(class_prob) 
