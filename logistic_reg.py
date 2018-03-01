# -*- coding: utf-8 -*-
# @Author: denglei
# @Date:   2018-03-01 11:45:13
# @Last Modified by:   denglei
# @Last Modified time: 2018-03-01 15:16:23


"""
Logistic regression经常用来做二分分类任务

训练数据集X， m x n Matrix, 通过一个线性函数h

z = h(X; theta)

得到一个值之后，我们可以通过sigmoid函数将其转换到0-1区间，有点类似概率

a = g(z) = 1 / (1 + exp(-z))

定义代价方程

1个样本： j = -a^y * (1 - a)^(1 - y), 如果y=1, 则j=a, 当估计a越接近1，那么j也是越小的； 
			如果y=0, 则j=1-a, 同理也是当a越接近真实的y, j都是减小的。
m个样本： J(theta) = multiply j over 1 -> m;

转换成对数形式的； J(theta) = -sum( y*log(a) + (1-y) * log(1-a) )

由于构建的代价函数是凸的，所以可以通过梯度降低优化方法： chain rule

dJ/da =  y/a - (1-y)/(1 - a) = y/a - (1-y)/(1-a)

da/dz = a * (1 - a)

dz/dtheta = -x

dJ/dtheta = dJ/da * da/dz * dz/dtheta
	= ( y/a - (1-y)/(1-a) ) * ( a*(1-a) ) * (-x)
	= ( y * (1-a) - (1-y)*a ) * (-x)
	= ( y - a*y - a + a*y ) * (-x)
	= ( y - a ) * (-x)
	= (a - y) * x


# 添加正则项，对参数进行限制

J_norm2 = lambda * sum(theta^2)       # 除去intercept的参数

J = J_theta + J_norm2

dJ/dtheta = (a-y)*x + 2 * lambda * theta

"""


import numpy as np
import utils as u
from scipy.optimize import fmin_l_bfgs_b


class BinaryLogisticReg(object):

	loss = None
	theta = None

	def __init__(self, **params):
		self.max_iter = params.get('max_iter', 500)
		self.debug = params.get('debug', False)
		self.fit_intercept = params.get('fit_intercept', True)
		self.learning_rate = params.get('learning_rate', 1e-3)
		self.lambdaa = params.get('lambda', 1)

	def compute_loss_gradient(cls, x, y, theta, lambdaa):
		"""
		计算损失以及theta的梯度
		"""
		# linear combination
		z = np.dot(x, theta.T)       # mx1

		# sigmoid
		a = u.sigmoid(z)             # mx1
		# print(np.array(a[:, -1].tolist()))

		# loss
		loss = -np.sum( y*np.log(a) + (1-y)*np.log(1-a) )

		# add regularizer
		reg_loss = lambdaa * np.sum(theta[1:]**2)
		loss += reg_loss

		# graident by samples
		theta_grad = np.dot((a - y).T, x)           # 1xn
		# add regularizer
		reg_theta_grad = 2 * lambdaa * theta
		theta_grad += reg_theta_grad

		return loss, theta_grad

	def fit_raw(self, x, y):
		"""
		x and y should be numpy array
		and y should be 0, 1 array, x should be normalized
		"""
		
		if self.fit_intercept:
			x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

		m, n = x.shape         # sample num, feature num

		if y.ndim == 1:
			y = y.reshape(m, 1)
		
		theta = np.random.rand(1, n)

		for ii in range(self.max_iter):
			loss, theta_grad = self.compute_loss_gradient(x, y, theta, self.lambdaa)
			# print(loss, end=' ')
			theta -= self.learning_rate * theta_grad

			if self.debug and (ii+1) % 10 == 0:
				print("Iteration: {}, Loss: {:.3f}".format(ii+1, loss))
		self.loss = loss
		self.theta = theta

	def fit(self, x, y):
		"""
		Using optimize framework to optimize the loss function and grad function
		to get best parameters
		"""
		
		if self.fit_intercept:
			x = np.concatenate((np.ones(shape=(x.shape[0], 1)), x), axis=1)

		init_theta = np.zeros(x.shape[1])

		def target_func(param, *args):

			x, y = args
			theta = param.reshape(1, x.shape[1])
			loss, _ = self.compute_loss_gradient(x, y, theta, self.lambdaa)
			return loss

		def grad_func(param, *args):
			x, y = args
			theta = param.reshape(1, x.shape[1])
			_, grad = self.compute_loss_gradient(x, y, theta, self.lambdaa)
			return grad

		opt_res = fmin_l_bfgs_b(target_func, x0=init_theta,
								args=(x, y), fprime=grad_func, maxiter=self.max_iter)
		self.theta, self.loss = opt_res[0], opt_res[1]

	def predict(self, new_x, threshold=.5):
		a = self.predict_proba(new_x)
		return np.array(a >= threshold, dtype=np.int32)

	def predict_proba(self, new_x):
		assert self.loss is not None
		if self.fit_intercept:
			new_x = np.concatenate((np.ones((new_x.shape[0], 1)), new_x), axis=1)
		z = np.dot(new_x, self.theta.T)
		a = u.sigmoid(z)
		return a
