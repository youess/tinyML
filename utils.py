# -*- coding: utf-8 -*-
# @Author: denglei
# @Date:   2018-03-01 12:10:25
# @Last Modified by:   denglei
# @Last Modified time: 2018-03-01 12:10:52

import numpy as np


def sigmoid(z):

	return 1 / (1 + np.exp(-z))