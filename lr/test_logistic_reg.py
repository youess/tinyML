# -*- coding: utf-8 -*-
# @Author: denglei
# @Date:   2018-03-01 12:24:14
# @Last Modified by:   denglei
# @Last Modified time: 2018-03-01 15:16:55


import numpy as np
from logistic_reg import BinaryLogisticReg
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression


np.random.seed(2018)


print("########################Test on Binary Logistic Regression########################")
m, n = 200, 2
x = np.random.rand(m, n)
y = np.array(np.random.rand(m, 1) > .5, dtype=np.int32)


my_clf = BinaryLogisticReg(max_iter=500, debug=False)
my_clf.fit_raw(x, y)
my_a = my_clf.predict(x)
print(my_clf.theta, my_clf.loss)
print('My score: \n', confusion_matrix(y, my_a), accuracy_score(y, my_a))

clf = BinaryLogisticReg()
clf.fit(x, y)
my_a = clf.predict(x)
print(clf.theta, clf.loss)
print('My score2: \n', confusion_matrix(y, my_a), accuracy_score(y, my_a))

print("################# Compared with sklearn ########################")
clf = LogisticRegression()
clf.fit(x, y[:, 0])
sk_a = clf.predict(x)

print("Sklearn score: \n", confusion_matrix(y, sk_a), accuracy_score(y, sk_a))


print("########################Test on Multiple Classes Logistic Regression########################")