# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:30:49 2018

@author: kazantseva
"""

import numpy as np

sample = np.array([1, 2, 3], ndmin = 2)

z = map(lambda x: x.sum(), sample)
x = np.fromiter(z, dtype=np.bool) 

sample_2 = np.insert(sample, 0, values = [1] * len(sample), axis = 1)

print(x)

def train_on_single_example(example, y):
    # your code goes here
    estimate = 1
    error = y - estimate    
    return np.abs(error)

r = train_on_single_example([1, 2, 3], 1)
  
