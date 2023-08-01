#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 13:25:13 2022

@author: yingshuyang
"""



import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from math import pi
from scipy.interpolate import interp1d
from numpy.linalg import multi_dot




def THzProfile(path):
    text = path
    with open(text, 'r') as f:
        lines = f.readlines()
        x = [float(line.split()[0]) for line in lines]
        y = [float(line.split()[1]) for line in lines]
    # x[:] = [x*1e-12 for x in x]
    x = np.array(x)
    y = np.array(y)
    return x,y


def THzProfile2(path,t_pos,end_time,N):
    text = path
    with open(text, 'r') as f:
        lines = f.readlines()
        x = [float(line.split()[0]) for line in lines]
        y = [float(line.split()[1]) for line in lines]
    x[:] = [x*1e-12+t_pos for x in x]
    x = np.array(x)
    x_prolong1 = np.linspace(0,t_pos,50)
    x_prolong2 = np.linspace(x[-1]+t_pos,end_time,50)
    y_prolong1 = np.linspace(0,0,50)
    y_prolong2 = np.linspace(0,0,50)
    t_grid = np.hstack((x_prolong1,x,x_prolong2))
    E_t = np.hstack((y_prolong1,y,y_prolong2))
    f = interpolate.interp1d(t_grid, E_t)
    # xnew = np.arange(0,end_time,0.05e-12)
    xnew = np.linspace(0, end_time,N)
    ynew = f(xnew) 
    return ynew,xnew






t1,E1 = THzProfile('/Users/yingshuyang/pythonfiles/TransferMatrixMethod/A_YIG_Fitting/experimental_data/1_T_Ref copy.txt')
t2,E2 = THzProfile('/Users/yingshuyang/pythonfiles/TransferMatrixMethod/A_YIG_Fitting/experimental_data/2_T_Sample copy.txt')



E11,t11 = THzProfile2('/Users/yingshuyang/pythonfiles/TransferMatrixMethod/A_YIG_Fitting/experimental_data/1_T_Ref.txt',20e-12,40e-12,500)
E22,t22 = THzProfile2('/Users/yingshuyang/pythonfiles/TransferMatrixMethod/A_YIG_Fitting/experimental_data/2_T_Sample.txt',20e-12,40e-12,500)


fig = plt.figure('test')


ax1 = plt.subplot(111)
# ax1.plot(t11,E11)
# ax1.plot(t22,E22)

ax1.plot(t1,E1)
ax1.plot(t2,E2)
