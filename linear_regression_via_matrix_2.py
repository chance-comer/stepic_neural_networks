# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:10:01 2018

@author: kazantseva
"""

import numpy as np

def compute_coefs(x, y):
  b = []
  transped = np.transpose(x)
  b = np.array(np.dot(transped, x))
  b = np.array(np.linalg.inv(b))
  b = np.dot(b, transped)
  b = np.dot(b, y)
  return b

data = np.loadtxt('boston_houses.csv', delimiter = ',', skiprows=1)

y = data[:, 0]
x = data[:, 1:]

x = np.insert(x, 0, [1] * len(y), axis = 1)

b = compute_coefs(x, y)

print(" ".join(str(i) for i in b))