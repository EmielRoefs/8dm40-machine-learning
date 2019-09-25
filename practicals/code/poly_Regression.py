import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(n=100, degree=1, noise=1, factors=None):
    # Generates a dataset by adding random noise to a randomly
    # generated polynomial function.
    
    x = np.random.uniform(low=-1, high=1, size=n)
    
    factors = np.random.uniform(0, 10, degree+1)
    
    y = np.zeros(x.shape)
    
    for idx in range(degree+1):
        y += factors[idx] * (x ** idx)

    # add noise
    y += np.random.normal(-noise, noise, n)
    
    return x, y

# load generated data
np.random.seed(0)

X, y = generate_dataset(n=100, degree=4, noise=1.5)

plt.plot(X, y, 'r.', markersize=12)

X = X[:, np.newaxis]

X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]


#%%

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))
    
import matplotlib.pyplot as plt

def PolyReg_plot(X=X, y=y):
    X_plot = np.linspace(-1.1, 1.1, 500)[:, None]
    plt.scatter(X.ravel(), y, color='red')
    axis = plt.axis()
    for degree in [1, 3, 5]:
        y_plot = PolynomialRegression(degree).fit(X, y).predict(X_plot)
        plt.plot(X_plot.ravel(), y_plot, label='degree={}'.format(degree))
    plt.legend(loc='best');
    plt.show()

PolyReg_plot(X=X_train, y=y_train)

#%%

from sklearn.model_selection import GridSearchCV

def gridsearch(X=X, y=y):
    param_grid = {'polynomialfeatures__degree': np.arange(21)}
    grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
    grid.fit(X,y);
    return grid.best_params_

gridsearch(X=X, y=y)

#%%
import sklearn
from sklearn.model_selection import validation_curve

def learningCurve(X=X, y=y):
    degree = np.arange(0, 21)
    train_score, val_score = validation_curve(PolynomialRegression(), X, y,
                                              'polynomialfeatures__degree', degree, cv=7)
    plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
    plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
    plt.legend(loc='best')
    plt.ylim(0, 1)
    plt.xlabel('degree')
    plt.ylabel('score');
    plt.show()

learningCurve(X=X, y=y)

