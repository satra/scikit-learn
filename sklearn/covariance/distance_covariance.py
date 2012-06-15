"""Distance Covariance: distance covariance estimation
"""

# Author: Satrajit Ghosh <satra@mit.edu>
# License: BSD Style
# Copyright: MIT

import warnings
import operator
import sys
import time

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .empirical_covariance_ import (empirical_covariance,
                                    EmpiricalCovariance, log_likelihood)

from ..utils import ConvergenceWarning
from ..linear_model import lars_path
from ..linear_model import cd_fast
from ..cross_validation import check_cv, cross_val_score
from ..externals.joblib import Parallel, delayed

###############################################################################
# Brownian-distance covariance estimator

def dcov(X, Y=None, unbiased=True):
    n = X.shape[0]
    a = squareform(pdist(X))
    a_mean = a.mean()
    A = a - a.mean(axis=1)[:, None] - a.mean(axis=0)[None, :] + a_mean
    if Y is not None:
        b = squareform(pdist(Y))
        b_mean = b.mean()
        B = b - b.mean(axis=1)[:, None] - b.mean(axis=0)[None, :] + b_mean
        V = np.mean(A * B)
    else:
        V = np.mean(A * A)
    if unbiased:
        T2 = a_mean
        if Y is not None:
            T2 *= b_mean
        V = (n ** 2) / ((n - 1) * (n - 2)) * (V - T2 / (n - 1))
    return V


def bdcov(X, assume_centered=False, unbiased=True):
    """Estimate the brownian distance covariance

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
      Data from which to compute the covariance estimate

    assume_centered: boolean
      If True, data are not centered before computation.
      Usefull to work with data whose mean is significantly equal to
      zero but is not exactly zero.
      If False, data are centered before computation.

    Returns
    -------
    distance_cov: array-like, shape (n_features, n_features)
      Brownian distance covariance

    References
    ----------
    """
    X = np.asarray(X)
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        if not assume_centered:
            X = X - X.mean()
        return np.atleast_2d((X ** 2).mean()), 0.
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))
        warnings.warn("Only one sample available. "\
                      "You may want to reshape your data array")
        n_samples = 1
        n_features = X.size
    else:
        n_samples, n_features = X.shape


    a = pdist(X)
    A

    return distance_cov

