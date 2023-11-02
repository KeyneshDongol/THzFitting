#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:49:50 2023

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

from Functions.Read_material import Material
from Functions.TMM import SpecialMatrix
from Functions import exp_pulse, fourier, tools
from scipy.optimize import minimize, LinearConstraint

# returns the theoretically transmitted pulse (in both time and frequency domain)
def E_TMM(layers, to_find, omega, eps0, mu, d, f_in,sub_layer, echoes_removed, unknown):  
    epsilon = []
    for l in layers:
        if type(l) == str:  
            if to_find[0] == True:                                        # permittivity
                epsilon.append(unknown*eps0)
            if to_find[1] == True:                                        # plasma and damping
                eps = Material(omega).drude(unknown[0], unknown[1]);  eps= 1*eps;  eps[0] = eps[1]
                epsilon.append(eps*eps0)
            if to_find[2] == True:                                        # n and k
                epsilon.append(Material(omega).epsilon(unknown)*eps0)
        else:
            epsilon.append(l)

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






if __name__ == '__main__':
    
    '''inputs'''
    f = open(Path.cwd()/'inputs1.json')
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

    '''Inputing the permittivities'''
    layers = []
    for j, k in zip(eps_data, is_known):
        if k == True:     # if known, find eps
            if type(j[0]) == str: ##
                layers.append(Material(omega).known_nk(j[0], j[1])*eps0)
            else:
                drude = Material(omega).drude(j[0], j[1]);     drude = 1*drude;  drude[0] = drude[1]
                layers.append(drude*eps0)
        else:
            layers.append('unknown')
            if to_find[0] == True:     # permittivity(real and imaginary)
                if type(j[0]) == str:
                    unknown = Material(omega).known_nk(j[0], j[1])
                else:
                    drude = Material(omega).drude(j[0], j[1]);     drude = 1*drude;  drude[0] = drude[1]
                    unknown = drude
            if to_find[1] == True:     # plasma and damping 
                unknown = np.array(j)
            if to_find[2] == True:     # n and k
                unknown = Material(omega).read_nk(j[0], j[1])
    
    
    '''Theoretical calculated transmitted pulse '''
    E_Theory = partial(E_TMM, layers, to_find, omega, eps0, mu, d, E_air_f,sub_layer,echoes_removed)    
    E_theo_t_new, E_theo_f_new = E_Theory(unknown)

    freq = omega*1e-12/2*pi
    time = t_grid*1e12

    # Define global style settings
    fontsize = 5
    title_fontsize = 6
    linewidth = 1
    linewidth2 = 2
    
    # Section 1: Minimization plot - time
    plt.figure('Minimization Plot - Time', figsize=(8.6/2.54, 5/2.54), dpi=200)
    
    # Subplot 1
    plt.subplot(121)
    plt.title('Time domain', fontsize=title_fontsize)
    plt.plot(time, E_air_in, linewidth=linewidth, label='E_air_in', color='lightblue')
    plt.plot(time, E_sample_out, linewidth=linewidth2, label='E_sample_out', color='tab:green')
    plt.plot(time, E_theo_t_new, linewidth=linewidth, label='E_theory_out', color='brown')
    plt.xlabel('Time (ps)', fontsize=fontsize)
    plt.ylabel('E(t) arb.units', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    # Subplot 2
    plt.subplot(122)
    plt.title('Freq. domain', fontsize=title_fontsize)
    plt.plot(freq, np.abs(E_exp_f), linewidth=linewidth2, label='E_sample_out', color='tab:green')
    plt.plot(freq, np.abs(E_theo_f_new), linewidth=linewidth, label='E_theory_fit', color='brown')
    plt.xlabel('Freq.(THz)', fontsize=fontsize)
    plt.ylabel('E($\omega$) arb.units', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0,20)    
    plt.tight_layout()
    
    






