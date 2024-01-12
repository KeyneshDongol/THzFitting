#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:50:02 2023

@author: yingshuyang
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial
import json
import time
import cmath
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from math import pi
from matplotlib.gridspec import GridSpec
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

''' error function without penalty term'''
# def Error_func(layers, to_find, omega, eps0, mu, d, E_air_f, E_exp_f,sub_layer,echoes_removed, unknown):
#     if to_find[0] == True:
#         unknown = unknown[:len(unknown)//2] + 1j*unknown[len(unknown)//2:]
#     if to_find[2] == True:
#         unknown = np.array(np.array_split(unknown, 2))    
#     E_theo_f = E_TMM(layers, to_find, omega, eps0, mu, d, E_air_f, sub_layer,echoes_removed,unknown)[1]    
#     return np.sum(np.abs(E_theo_f - E_exp_f))


''' error function with penalty term'''
def Error_func(layers, to_find, omega,eps0, mu, d, E_air_f, E_exp_f, sub_layer, echoes_removed, unknown):
    if to_find[0] == True:
        unknown = unknown[:len(unknown)//2] + 1j*unknown[len(unknown)//2:]
    if to_find[2] == True:
        unknown = np.array(np.array_split(unknown, 2))

    # Additional penalty term to enforce positive unknown[1]
    penalty = 0.0
    if to_find[2] == True:
        negative_k = np.where(unknown[1] < 0)[0]
        if len(negative_k) > 0:
            penalty = np.sum(np.exp(-unknown[1][negative_k]))

    E_theo_f = E_TMM(layers, to_find, omega, eps0, mu, d, E_air_f, sub_layer, echoes_removed, unknown)[1]
    
    # log_diff = np.log(np.abs(E_theo_f)) - np.log(np.abs(E_exp_f))
    # phase_diff = np.angle(E_theo_f) - np.angle(E_exp_f)

    # result = np.sum(log_diff**2) #+ phase_diff**2)
    
    return np.sum(np.abs(E_theo_f - E_exp_f)**2) #+ penalty



''' error function with penalty term'''
def Error_func2(layers, to_find, omega,eps0, mu, d, E_air_f, E_exp_f, sub_layer, echoes_removed, unknown):
    if to_find[0] == True:
        unknown = unknown[:len(unknown)//2] + 1j*unknown[len(unknown)//2:]
    if to_find[2] == True:
        unknown = np.array(np.array_split(unknown, 2))

    # Additional penalty term to enforce positive unknown[1]
    penalty = 0.0
    if to_find[2] == True:
        negative_k = np.where(unknown[1] < 0)[0]
        if len(negative_k) > 0:
            penalty = np.sum(np.exp(-unknown[1][negative_k]))

    E_theo_f = E_TMM(layers, to_find, omega, eps0, mu, d, E_air_f, sub_layer, echoes_removed, unknown)[1]
    
    
    '''Transfer function theory'''
    eps_Air = np.array([1] * len(omega))
    layers1 = [eps_Air * eps0, eps_Air * eps0, eps_Air * eps0, eps_Air * eps0, eps_Air * eps0, eps_Air * eps0]
    E_air_theo_t, E_air_theo_f = E_TMM_new(layers1, to_find, omega, eps0, mu, d, E_air_f,sub_layer,echoes_removed)

    T_theo_f = E_theo_f/E_air_theo_f
    
    return np.sum(np.abs(T_theo_f - E_exp_f)**2) #+ penalty

'''result smoothing'''

def noise_remove(res, omega, window_size, prominence_threshold1, prominence_threshold2):
    def find_all_peaks(data, prominence_threshold):
        peaks, _ = find_peaks(data, prominence=prominence_threshold)
        if data[0] > data[1]:  # Check if the first point is a peak
            peaks = np.insert(peaks, 0, 0)
        if data[-1] > data[-2]:  # Check if the last point is a peak
            peaks = np.append(peaks, len(data) - 1)
        return peaks

    def moving_average(data, window_size):
        # Pad the data at the beginning and end to avoid edge effects
        padded_data = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
        
        # Apply the moving average filter
        smoothed_data = np.convolve(padded_data, np.ones(window_size)/window_size, mode='valid')
        
        return smoothed_data
    # Split and extract the data
    result = np.array(np.array_split(res['x'], 2))
    
    # Find all peaks
    pos_peaks1 = find_all_peaks(result[0], prominence_threshold1)
    neg_peaks1 = find_all_peaks(-result[0], prominence_threshold1)
    pos_peaks2 = find_all_peaks(result[1], prominence_threshold2)
    neg_peaks2 = find_all_peaks(-result[1], prominence_threshold2)
    
    # Remove both positive and negative peaks
    result[0][pos_peaks1] = 100; result[0][neg_peaks1] = 50
    result[1][pos_peaks2] = 100; result[1][neg_peaks2] = 50
    
    # Apply moving average smoothing to the modified data
    smoothed_data1 = moving_average(result[0], window_size)
    smoothed_data2 = moving_average(result[1], window_size)
    
    # Create interpolation functions
    interp_func1 = interp1d(omega, smoothed_data1, kind='cubic')
    interp_func2 = interp1d(omega, smoothed_data2, kind='cubic')
    
    # Create a finer grid of x-coordinates for the interpolated data
    fine_x = np.linspace(omega[0], omega[-1], 500)
    
    # Interpolate the smoothed data on the finer grid using fine_x
    interpolated_data1 = interp_func1(fine_x)
    interpolated_data2 = interp_func2(fine_x)
    
    return fine_x, smoothed_data1, smoothed_data2, interpolated_data1, interpolated_data2, result




