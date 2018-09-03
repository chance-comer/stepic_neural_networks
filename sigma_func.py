# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 16:27:30 2018

@author: kazantseva
"""

import numpy as np
import matplotlib.pyplot as plt

w = np.array([100000000000000000, 120, -100])
x = np.array([0, 1, 1])

def sigma(x):
  return 1 / (1 + np.exp(-x))

arg = np.transpose(w).dot(x)

res_sigma = sigma(arg)

der = (res_sigma ) * res_sigma * (1 - res_sigma) * x