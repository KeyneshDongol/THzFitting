#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:34:49 2023

@author: yingshuyang
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial
import json
import time
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from math import pi
from matplotlib.gridspec import GridSpec
from Functions.Read_material import Material
from Functions.TMM import SpecialMatrix
from Functions import exp_pulse, fourier, tools
from scipy.optimize import minimize, LinearConstraint
from Functions.mainfunction import E_TMM, Error_func,  noise_remove

from matplotlib import cm




if __name__ == '__main__':

    def E_TMM_new(layers, to_find, omega, eps0, mu, d, f_in,sub_layer, echoes_removed):  
        epsilon = layers

        '''Transfer matrix'''
        TMM = SpecialMatrix(omega, mu, epsilon, d)
        _, _, T_0inf = TMM.Transfer_Matrix()
        T_0s  = TMM.Transfer_Matrix_special_0N(sub_layer)
        T_sinf  = TMM.Transfer_Matrix_special_Ninf(sub_layer)
    
        '''Tranmission & reflection coefficients'''
        t_0s        = TMM.Transmission_coeff(T_0s)
        t_sinf      = TMM.Transmission_coeff(T_sinf)
        t           = TMM.Transmission_coeff(T_0inf)
        t_noecho    = np.multiply(t_0s, t_sinf)
    
        '''Remove echo or not'''    
        if echoes_removed[0]==True:
            f_inf_R      = f_in * t_noecho 
        else:
            f_inf_R      = f_in*t
            
        '''Transmitted wave in freq domain'''
        trans = fourier.ift(f_inf_R)    
        return trans, f_inf_R



    
    '''inputs'''
    f = open(Path.cwd()/'inputs.json')
    config = json.load(f)
    to_find = list(config['to_find'].values()) 
    input_num = list(config['input'].values())
    mat_data = config['layers']
    echoes_removed =  list(config['Echoes'].values())

    '''Material and Geometry of the sample'''
    mu = 12.57e-7
    eps0 = 8.85e-12
    d = [i['thickness'] for i in mat_data]       #list
    eps_data = [i['eps_data'] for i in mat_data] #list 
    is_known = [i['is_known'] for i in mat_data] #list
    is_substrate = [i['is_substrate'] for i in mat_data] #list
    sub_layer = np.where(is_substrate)[0][0]
    
    '''Experimental data'''
    pulse_res = config['pulse_res']  
    n, tmin, tmax, tpos = pulse_res.values()
    pulse_path = input_num[0]
    exp_in_pulse = Path.cwd()/'experimental_data'/pulse_path[0]  # current working directory
    exp_out_pulse = Path.cwd()/'experimental_data'/pulse_path[1] # current working directory
    t_grid, E_air_in, E_sample_out = exp_pulse.fitted_pulse(exp_in_pulse, exp_out_pulse, tmin, tmax, tpos, d, n) # data reading (propagated through air)
    omega, E_air_f = fourier.ft(t_grid, E_air_in)
    E_exp_f = fourier.ft(t_grid, E_sample_out)[1]





    
    def calculate_error(n, k):
        eps_Air = np.array([1] * len(omega))
        index_n = np.array([n] * len(omega))
        index_k = np.array([k] * len(omega))
        epsS = (index_n + 1j * index_k) ** 2
        layers = [eps_Air * eps0, epsS * eps0, eps_Air * eps0, eps_Air * eps0, eps_Air * eps0, eps_Air * eps0]
    
        # Replace with actual E_TMM_new function call
        E_theo_t, E_theo_f = E_TMM_new(layers, to_find, omega, eps0, mu, d, E_air_f,sub_layer,echoes_removed)
        
        error = np.sum(np.abs(E_theo_f-E_exp_f)**2)
        return error
    
    # Set up the ranges for n and k
    n_values = np.linspace(1, 3, 300)  # 25 steps from 0 to 6 for refractive index
    k_values = np.linspace(0, 0.3, 300)  # 25 steps from 0 to 1 for extinction coefficient
    
    # Prepare a grid for error values
    error_values = np.zeros((len(n_values), len(k_values)))
    
    # Calculate the error function for each combination of n and k
    for i, n in enumerate(n_values):
        for j, k in enumerate(k_values):
            error_values[i, j] = calculate_error(n, k)
 
    




    
    
    # Create a figure for the plots
    fig = plt.figure(figsize=(18, 8))
    
    # 3D plot of error function - larger subplot
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2, projection='3d')
    N, K = np.meshgrid(n_values, k_values)
    surf = ax1.plot_surface(N, K, error_values.T, cmap=cm.viridis)
    ax1.set_xlabel('Refractive Index (n)')
    ax1.set_ylabel('Extinction Coefficient (k)')
    ax1.set_zlabel('Error Function')
    ax1.set_title('3D Error Function Surface Plot')
    plt.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Common colormap for the 2D plots
    cmap = cm.viridis
    
    # 2D Projection along n-axis - smaller subplot
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    for k_index, k in enumerate(k_values):
        color = cmap(k_index / len(k_values))
        ax2.plot(n_values, error_values[:, k_index], color=color)
    ax2.set_xlabel('Refractive Index (n)')
    ax2.set_ylabel('Error Function')
    ax2.set_title('Error Projection Along n-axis')
    
    # 2D Projection along k-axis - smaller subplot
    ax3 = plt.subplot2grid((2, 3), (1, 2))
    for n_index, n in enumerate(n_values):
        color = cmap(n_index / len(n_values))
        ax3.plot(k_values, error_values[n_index, :], color=color)
    ax3.set_xlabel('Extinction Coefficient (k)')
    ax3.set_ylabel('Error Function')
    ax3.set_title('Error Projection Along k-axis')
    
    plt.tight_layout()
    plt.show()