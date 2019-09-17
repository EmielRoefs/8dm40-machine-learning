import numpy as np

def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta

def predict(X, y, beta):
    """
    Predict target scores using multi linear regression
    :param X: Input data matrix
    :param y: Target vector
    :param beta: Estimated coefficient vector for the linear regression
    :return: MSE
    """
    
    # add colum of ones for intercept 
    # same as in the least squares
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    
    # predict target values 
    predict = np.dot(X,beta)
    
    return predict

def MSEevaluation(y, predict):
    """
    Evaluate predictions the mean sqaured error
    :param y: original values
    :param predict: predicted values 
    :return: the mean squared error of the predicted values 
    """
    
    n = len(y)                                                                  # number of samples 
    square = 0 
    for i in range(0, n-1):
        square += (y[i]- predict[i])**2
        
    MSE = square/n
    
    return MSE
     
