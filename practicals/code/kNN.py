# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:38:52 2019

@author: s151385
"""
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
import math

breast_cancer = load_breast_cancer()
# add subfolder that contains all the function implementations
# to the system path so we can import them
import sys
sys.path.append('code/')

# the actual implementation is in linear_regression.py,
# here we will just use it to fit a model
from linear_regression import *

# load the dataset
# same as before, but now we use all features
X_train = breast_cancer.data[:400, :]
y_train = breast_cancer.target[:400, np.newaxis]
X_test = breast_cancer.data[400:, :]
y_test = breast_cancer.target[400:, np.newaxis]

k = 5
for i in range(0,len(X_test[:,0])):
    dist_sum = np.zeros([len(X_train[:,0]),1])
    for j in range(0,len(X_train[:,0])):
        dist = np.square(X_train[j,:]-X_test[i,:])
        dist_sum[j] = math.sqrt(sum(dist))
    dist_sum_sort, y_train_sort = zip(*sorted(zip(dist_sum, y_train)))
    class_votes = 0
    for p in range(0,k):
        class_votes = class_votes + y_train_sort[k][0]
        
def classification(Xtrain,Ytrain,Xtest,Ytest,k):
    for i in range(0,len(Xtest[:,0])):
        dist_sum = np.zeros([len(Xtrain[:,0])],1)
        dist_sum, Ytest = zip(*sorted(zip(dist_sum, Ytest)))
        for j in range(0,len(Xtrain[0,:])):
            dist = np.square(Xtrain[j,:]-Xtest[i,:])
            dist_sum[j] = math.sqrt(np.sum(dist))
    return()