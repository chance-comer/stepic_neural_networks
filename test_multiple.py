# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 09:23:39 2018

@author: Nataliya
"""

import numpy as np

ar1 = np.array([1, 2, 3], ndmin = 2)
ar2 = np.array([[1, 2], [2, 3], [3, 4]], ndmin = 2)

def func(a):
    return a

func(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)"""
    return sigmoid(x) * (1 - sigmoid(x))

deltas = np.array([[1,2,3],[4,5,6],[1,1,1],[2,2,2]])
sums = np.array([[1,2],[4,5],[1,1],[2,2]])
weights = np.array([[0.4, 0.5],[0.2, 0.1],[0.5, 0.65]])

def get_error(deltas, sums, weights):
    """
    compute error on the previous layer of network
    deltas - ndarray of shape (n, n_{l+1})
    sums - ndarray of shape (n, n_l)
    weights - ndarray of shape (n_{l+1}, n_l)
    """
    # here goes your code
    return sum([((np.array((1 - sigmoid(sums[i]))*sigmoid(sums[i]), ndmin = 2)).T * weights.T).dot(np.array(deltas[i],ndmin =2).T) for i in np.arange(len(sums))])/len(sums)


def get_error_2(deltas, sums, weights):
    """
    compute error on the previous layer of network
    deltas - ndarray of shape (n, n_{l+1})
    sums - ndarray of shape (n, n_l)
    weights - ndarray of shape (n_{l+1}, n_l)
    """
    # here goes your code
    return ((weights.T.dot(deltas.T)) * sigmoid_prime(sums).T).mean(axis = 0)

d = get_error(deltas, sums, weights)
