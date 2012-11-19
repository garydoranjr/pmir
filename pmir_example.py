#!/usr/bin/env python
"""
A simple example of using MIClusterRegress with clustering
and regression classes from scikits-learn
"""
import numpy as np
from scipy.io import loadmat
from sklearn.svm import NuSVR
from sklearn.metrics import r2_score

from pmir import PruningMIR

if __name__ == '__main__':
    exset = loadmat('thrombin.mat', struct_as_record=False)['thrombin']

    # Construct bags
    all_labels = exset[:, 0]
    X = exset[:, 1:-1]
    bags = []
    values = []
    for label in np.unique(all_labels.flat):
        indices = np.nonzero(all_labels == label)
        bags.append(X[indices])
        values.append(float(exset[indices, -1][0, 0]))
    y = np.array(values)

    # Fit bags, predict labels, and compute simple MSE
    regress = NuSVR(kernel='rbf', gamma=0.1, nu=0.2, C=1.0)
    pmir = PruningMIR(regress)
    pmir.fit(bags, y)
    y_hat = pmir.predict(bags)
    print 'R^2: %f' % r2_score(y, y_hat)
