# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:38:52 2019

@author: s151385
"""
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
# =============================================================================
# breast_cancer = load_breast_cancer()
# X_train = breast_cancer.data[:400, :]
# y_train = breast_cancer.target[:400, np.newaxis]
# X_test = breast_cancer.data[400:, :]
# y_test = breast_cancer.target[400:, np.newaxis]
# =============================================================================
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
import math

     
def classification(X_train,y_train,X_test,y_test,k):
    # kNN classification
    # input: X_train: feature space used for training
    #       X_test: feature space used for testing
    #       y_train: ground truth classification for train dataset
    #       y_test: ground truth classification for test dataset
    #       k: number of nearest neighbor used
    # output: AUC of ROC as performance measure and predicted classification of test set
    
    predicted_y = np.zeros([len(X_test[:,0]),1])
    for i in range(0,len(X_test[:,0])):
        dist_sum = np.zeros([len(X_train[:,0]),1])
        for j in range(0,len(X_train[:,0])):
            dist = np.square(X_train[j,:]-X_test[i,:])
            dist_sum[j] = math.sqrt(sum(dist))
        dist_y = np.concatenate((dist_sum,y_train),axis=1)
        dist_y_sort = dist_y[dist_y[:,0].argsort()]
        if sum(dist_y_sort[0:k,1]) >= k/2:
            predicted_y[i] = 1
        else:
            predicted_y[i] = 0
    
    #CM(y_test,predicted_y)
    roc = ROC(y_test,predicted_y)
    
    return(roc,predicted_y)
    
def regression(X_train,y_train,X_test,y_test,k):
    import matplotlib.pyplot as plt
    # kNN classification
    # input: X_train: feature space used for training
    #       X_test: feature space used for testing
    #       y_train: ground truth classification for train dataset
    #       y_test: ground truth classification for test dataset
    #       k: number of nearest neighbor used
    # output: AUC of ROC as performance measure and predicted classification of test set
    reg = np.zeros([len(X_test[:,0]),1])
    for i in range(0,len(X_test[:,0])):
        dist_sum = np.zeros([len(X_train[:,0]),1])
        for j in range(0,len(X_train[:,0])):
            dist = np.square(X_train[j,:]-X_test[i,:])
            dist_sum[j] = math.sqrt(sum(dist))
        dist_y = np.concatenate((dist_sum,y_train),axis=1)
        dist_y_sort = dist_y[dist_y[:,0].argsort()]
        reg[i] = sum(dist_y_sort[0:k,1])/k
    return(reg)

### Some code Emiel wrote during his BEP, nice for viewing results
def CM(true,predicted):
    from sklearn.metrics import confusion_matrix
    import itertools
    import matplotlib.pyplot as plt
    import numpy as np
    cm = confusion_matrix(true, predicted)
    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    classes = {'benign':0, 'melanoma':1}
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def ROC(true,predictions): 
    from sklearn.metrics import roc_curve, auc 
    import matplotlib.pyplot as plt    
    fpr= dict()
    tpr= dict()
    roc_auc =dict()
    fpr,tpr, _ = roc_curve(true,predictions)
    roc_auc = auc(fpr, tpr)  
# =============================================================================
#     plt.figure()
#     plt.plot(fpr, tpr, lw=2,
#              label='ROC curve  (area = {f:.2f})'.format( f=roc_auc))
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     plt.show()
# =============================================================================
    return(roc_auc)