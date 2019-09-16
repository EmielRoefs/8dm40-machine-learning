import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from math import *

def visualize_gauss(X, Y):
    plt.figure(figsize=(15,15))
    for feature in range(0,30):
        Xfeat = X[:, feature]
        X_mal = [] 
        X_ben = []
        for i in range(569):
            if Y[i] == 0: 
                X_mal.append(Xfeat[i])
            elif Y[i] == 1:
                X_ben.append(Xfeat[i])
        mu_ben = np.mean(X_ben) 
        sd_ben = np.std(X_ben)
        mu_mal = np.mean(X_mal) 
        sd_mal = np.std(X_mal)
        x = np.linspace(min(mu_ben, mu_mal) - 3*max(sd_ben, sd_mal), max(mu_ben, mu_mal) + 3*max(sd_ben, sd_mal), 100)
        plt.subplot(5, 6, feature+1)
        plt.plot(x, stats.norm.pdf(x, mu_mal, sd_mal))
        plt.plot(x, stats.norm.pdf(x, mu_ben, sd_ben))
        plt.title('feature {}'.format(feature+1))
        plt.xlabel('x')
        plt.ylabel('N(x)')
    plt.tight_layout()
    plt.show()
