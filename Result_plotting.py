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
from Functions.mainfunction import E_TMM, Error_func,  noise_remove




# if __name__ == '__main__':
    
'''inputs'''
f = open(Path.cwd()/'results.json')
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

t_grid_raw, E_air_in_raw = exp_pulse.read_pulse(exp_in_pulse) 
t_grid_raw, E_sample_out_raw = exp_pulse.read_pulse(exp_out_pulse) 




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



# =============================================================================
# Plottings dresults calculated fiel
# =============================================================================

freq = omega*1e-12/2*pi
time = t_grid*1e12

# Define global style settings
fontsize = 5
title_fontsize = 5
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






# =============================================================================
# Plottings of raw data
# =============================================================================




# Define the style settings
figsize = (8.6/2.54, 8.6/2.54)  # figure size in inches
linewidth = 1  # line width for the plots
tick_labelsize = 5  # size of the ticks labels
label_fontsize = 5  # font size for the x and y labels
title_fontsize = 5  # font size for the titles
colors = ['blue', 'green']  # colors for the plots

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=200)

# First subplot for raw data
ax1.plot(t_grid_raw, E_air_in_raw, label='E_air_in_raw', linewidth=linewidth, color=colors[0])
ax1.plot(t_grid_raw, E_sample_out_raw, label='E_sample_out_raw', linewidth=linewidth, color=colors[1])
ax1.set_xlabel('Time (ps)', fontsize=label_fontsize)
ax1.set_ylabel('Electric Field', fontsize=label_fontsize)
ax1.tick_params(axis='both', labelsize=tick_labelsize)
ax1.legend(fontsize=label_fontsize)
ax1.set_title('Raw data', fontsize=title_fontsize)

# Second subplot for processed raw data
ax2.plot(time, E_air_in, label='E_air_in_processed', linewidth=linewidth, color=colors[0])
ax2.plot(time, E_sample_out, label='E_sample_out_processed', linewidth=linewidth, color=colors[1])
ax2.set_xlabel('Time (ps)', fontsize=label_fontsize)
ax2.set_ylabel('Electric Field', fontsize=label_fontsize)
ax2.tick_params(axis='both', labelsize=tick_labelsize)
ax2.legend(fontsize=label_fontsize)
ax2.set_title('Processed raw data', fontsize=title_fontsize)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure if needed
plt.savefig('processed_data_plots.png', dpi=300)

# Show the plot
plt.show()








