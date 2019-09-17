import numpy as np

def weightedlsq(X,y):
    """
    Least squares linear regression with weight implementation
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression with weight implementation
    """
    
    # create matrix of weights 
    nsamples = X.shape[0]
    nfeatures = X.shape[1]
    
    w_matrix = np.zeros((nsamples, nfeatures))
    for i in range(0,nfeatures):
        feature = X[:,i]
        for idx in range(0, len(feature)):
            w = np.count_nonzero(feature == feature[idx])
            w_matrix[idx, i] = w
            
    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    w_matrix = np.concatenate((ones, w_matrix), axis=1)
            
    # add weights to the input data matrix
    w_X = X*w_matrix         
   
    # calculate the coefficients
    w_beta = np.dot(np.linalg.inv(np.dot(w_X.T, w_X)), np.dot(w_X.T, y))

    return w_beta

