# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:08:40 2018

@author: kazantseva
"""

from urllib.request import urlopen
import numpy as np
import ssl as s

filename = "boston_houses.csv"
f = open(filename)
data = np.loadtxt(f, skiprows = 1, delimiter=',')
means = data.mean(axis=0)

x = np.array([[1, 60], [1, 50], [1, 75]])
y = np.array([10, 7, 12])

b = np.dot(np.transpose(x), x)

c = np.linalg.inv(b)

d = np.dot(c, np.transpose(x))

e = np.dot(d, y)
#b = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x)), y)